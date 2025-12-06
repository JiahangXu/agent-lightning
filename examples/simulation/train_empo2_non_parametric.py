import os
import re
import time
import argparse
import asyncio
import wandb
import random
import subprocess
import numpy as np

from datasets import Dataset
from omegaconf import OmegaConf
from typing import List
from rich.console import Console
from dotenv import load_dotenv

import agentlightning as agl

from examples.simulation.empo2_agent import reset_memory
from examples.simulation.utils import vllm_server, run_cmd, kill_process_on_port

agl.configure_logger()
console = Console()

TOTAL_ITERATIONS = 50
VLLM_PORT = 12316
LLM_PROXY_PORT = 12358
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tip_probs = 1.0

async def empo2_algorithm(*, train_dataset, store: agl.LightningStore):
    algo_marker = "[bold red][Algo][/bold red]"

    llm_proxy = agl.LLMProxy(port=LLM_PROXY_PORT, store=store)

    # First launch the vLLM server
    with vllm_server(MODEL_NAME, VLLM_PORT) as server_address:
        # Update the model list of the LLM proxy and start it
        model_list: List[agl.ModelConfig] = [
            {
                "model_name": MODEL_NAME,
                "litellm_params": {
                    "model": f"hosted_vllm/{MODEL_NAME}",
                    "api_base": server_address,
                },
            }
        ]
        console.print(f"{algo_marker} Updating model list and restarting LLM proxy: {model_list}")
        llm_proxy.update_model_list(model_list)
        llm_proxy.restart()
    
        resources: agl.NamedResources = {"main_llm": llm_proxy.as_resource()}
        resource_update = await store.add_resources(resources)

        for iteration in range(TOTAL_ITERATIONS):
            console.print(f"\n{algo_marker} Starting iteration {iteration}")

            # Create tasks for runners to run, associating them with the proxy address
            rollouts: List[agl.Rollout] = []
            for t in train_dataset:
                t["global_steps"] = iteration
                t["train_mode"] = "on-policy-with-tips" if random.random() < tip_probs else "on-policy"
                rollouts.append(await store.enqueue_rollout(input=t, mode="train", resources_id=resource_update.resources_id))

            console.print(f"{algo_marker} Enqueued {len(rollouts)} rollouts")

            # Wait for the tasks to complete
            completed_rollouts: List[agl.Rollout] = []
            while True:
                completed_rollouts = await store.wait_for_rollouts(
                    rollout_ids=[rollout.rollout_id for rollout in rollouts],
                    timeout=0.0,  # Timeout must be a very small value to avoid blocking the store server
                )
                if len(completed_rollouts) >= len(rollouts):
                    console.print(f"{algo_marker} Received all {len(rollouts)} rollouts")
                    break
                console.print(
                    f"{algo_marker} Received {len(completed_rollouts)} rollouts, waiting for more..."
                )
                await asyncio.sleep(5.0)

            logs = {}
            for rollout in completed_rollouts:
                console.print(f"{algo_marker} Received Result: {rollout}")
                if rollout.status != "succeeded":
                    raise RuntimeError(f"Rollout {rollout.rollout_id} did not succeed. Status: {rollout.status}")
                spans = await store.query_spans(rollout.rollout_id)

                # Logs LLM spans for debugging and inspection here
                # await log_llm_span(spans)

                # The algorithm records the final reward for sorting
                
                final_reward = agl.find_final_reward(spans)
                assert final_reward is not None, "Expected a final reward from the client."
                console.print(f"{algo_marker} Final reward: {final_reward}")

                variation_idx = rollout.input["variation_idx"]
                if variation_idx not in logs:
                    logs[variation_idx] = []

                logs[variation_idx].append(final_reward)
                
            wandb_logs = {}
            for variation_idx, rewards in logs.items():
                arr = np.array(rewards)
                wandb_logs[f"episode_return_each/variant_{variation_idx}_min"] = float(np.min(arr))
                wandb_logs[f"episode_return_each/variant_{variation_idx}_mean"] = float(np.mean(arr))
                wandb_logs[f"episode_return_each/variant_{variation_idx}_max"] = float(np.max(arr))

            all_rewards = [reward for rewards in logs.values() for reward in rewards]
            if len(all_rewards) > 0:
                all_arr = np.array(all_rewards)
                wandb_logs["episode_return_all/min"] = float(np.min(all_arr))
                wandb_logs["episode_return_all/mean"] = float(np.mean(all_arr))
                wandb_logs["episode_return_all/max"] = float(np.max(all_arr))

            wandb.log(wandb_logs, step=iteration)

@agl.algo
async def empo2_algorithm_usable_in_trainer(*, train_dataset, store: agl.LightningStore):
    return await empo2_algorithm(train_dataset=train_dataset, store=store)

async def log_llm_span(spans: List[agl.Span]) -> None:
    """Logs the LLM related spans that records prompts and responses."""
    for span in spans:
        if "chat.completion" in span.name:
            console.print(f"[bold green][LLM][/bold green] Span {span.span_id} ({span.name}): {span.attributes}")

def get_config(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if "variables" in cfg:
        del cfg["variables"]
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="scienceworld2")
    parser.add_argument("--algorithm", type=str, default="grpo")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_workers", type=int, default=40, help="Number of workers for training")
    parser.add_argument("--task_num", type=int, default=25, help="ScienceWorld Task number to inject as env var")
    parser.add_argument("--_background", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Restart Ray cluster cleanly
    kill_process_on_port(4747)
    run_cmd("pkill -f AgentLightning")

    # set environment variable before loading configs
    if args.env == "scienceworld":
        os.environ["TASK_NUM"] = str(args.task_num)

    # Load configs
    agent_config_path = f"examples/simulation/envs/env_config/{args.env}.yaml"
    env_prefix = re.sub(r"\d+$", "", args.env)
    if args.debug:
        trainer_config_path = f"examples/simulation/run/{env_prefix}/debug/{args.algorithm}.yaml"
    else:
        trainer_config_path = f"examples/simulation/run/{env_prefix}/{args.algorithm}.yaml"
    agent_config = get_config(agent_config_path)

    # Load datasets
    dataset_dir = f"examples/simulation/task_data/scienceworld/single_data/{str(args.task_num)}"
    train_files = f"{dataset_dir}/train.parquet"
    train_data = Dataset.from_parquet(train_files).select(range(args.n_workers))

    # Initialize agent
    kill_process_on_port(8000)
    kill_process_on_port(8001)

    subprocess.Popen(
        f"nohup python algorithms/empo2/server_bert.py > logs/bert_{args.task_num}.log 2>&1 &",
        shell=True
    )
    subprocess.Popen(
        f"nohup python algorithms/empo2/server_mem.py > logs/mem_{args.task_num}.log 2>&1 &",
        shell=True
    )

    NUM_MEMORY = 5
    time.sleep(3)
    reset_memory(NUM_MEMORY)

    run_id = f"{int(time.time())}"
    load_dotenv()
    wandb.init(entity=os.getenv("WANDB_ENTITY"), project="AGL-EMPO2-Non-Parametric2", name=f"task_{args.task_num}_tips_probs_{tip_probs}", resume=None, id=run_id)
    
    trainer = agl.Trainer(n_workers=args.n_workers, algorithm=empo2_algorithm_usable_in_trainer)

    from empo2_agent import EMPO2Agent
    agent = EMPO2Agent(agent_config)
    trainer.fit(agent, train_data)