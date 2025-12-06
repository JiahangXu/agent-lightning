import re
import copy
import random
import numpy as np
import requests
import logging
from typing import Any, Dict

from envs import make_env_manager
from captioners.add_instruction import add_chat_instruction, add_single_tips, add_chat_tips, add_chat_all_tips
from captioners.gather_chats import gather_chats
from agentlightning import (
    LLM, 
    NamedResources, 
    Rollout,
    configure_logger,
    emit_reward, 
    operation
)
from agentlightning.utils.otel import make_link_attributes

from simulation_agent import SimulationAgent

configure_logger()
logger = configure_logger(name=__name__, level=logging.ERROR)

def do_compress(text):
    url = "http://127.0.0.1:8000/key_cal/"
    headers = {"Content-Type": "application/json"}  # 明确指定 JSON 格式
    data = {"text": text}
    response = requests.post(url, json=data, headers=headers)  # 使用 json 参数
    return response.json()

url_mem = "http://127.0.0.1:8001/mem/"

def retrieve_memory(idx, key):
    response = requests.post(url_mem, json={
        "key": key,
        "idx": idx
    })
    count, data = response.json()
    return count, data

def reset_memory(mem_list_num):
    requests.post(url_mem, json={
        "key": [],
        "idx": mem_list_num,  # 用于初始化多个 memory slot
        "content": "Reset"
    })

def add_memory(idx, key, content, score):
    requests.post(url_mem, json={
        "key": key,
        "idx": idx,
        "content": content,
        "score": score
    })

class EMPO2Agent(SimulationAgent):
    def __init__(self, config, trained_agents: str | None = None) -> None:
        super().__init__(config=config, trained_agents=trained_agents)

    def _get_tip_obs(self, obs, tips):
        obs_type = self.config.captioner.obs_type

        if obs_type == "single":
            return add_single_tips(obs, tips)
        elif obs_type == "chat":
            return add_chat_tips(obs, tips)
        else:
            raise ValueError(f"Unsupported obs_type '{obs_type}' for _get_tip_obs (expected 'single')")

    def _get_all_tip_obs(self, obs, tip_list):
        obs_type = self.config.captioner.obs_type
        if obs_type == "chat":
            return add_chat_all_tips(obs, tip_list)
        else:
            raise ValueError(f"Unsupported obs_type '{obs_type}' for _get_tip_obs (expected 'single')")

    def _get_tip_generation_prompt(self, chat_obs):
        return add_chat_instruction(chat_obs, "tip")

    async def rollout_async(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        rollout_id = rollout.rollout_id
        global_steps = task["global_steps"]
        logger.info(f"[Rollout {rollout_id}] Task: {task}")

        reward_scale = float(self.config["reawrd_scale"])

        # Setup LLM + agent
        llm: LLM = resources.get("main_llm")
        print("Training with model:", llm.model, "on endpoint:", llm.endpoint)
        self.agent = self._build_agent(llm, 1.0 if rollout.mode == "train" else 0.4)

        if rollout.mode == "train":
            train_mode = task["train_mode"]
        else:
            train_mode = "on-policy"

        if rollout.mode == "train" and (train_mode == "off-policy" or train_mode == "on-policy-with-tips"):
            use_tips = True 
        else:
            use_tips = False

        variation_idx = task["variation_idx"]

        try:
            # Setup environment
            self.env = make_env_manager(self.config.env_name, task, self.config, render_mode=None)
            obs, pure_env_obs, infos = self.env.reset()
            episode_reward, done = 0.0, False

            pure_obs_for_mem = []
            history_actions_for_mem = []
            tip_list = []

            step_count = 0
            while not done:
                if use_tips:
                    text = gather_chats(obs)
                    key = np.array(do_compress(text)['key']).reshape(-1, ).tolist()
                    count, mem_list = retrieve_memory(variation_idx, key)
                else:
                    count, mem_list = 0, []

                ret_tips, intrinsic_reward = "", 0.0

                if use_tips:
                    if count > 0:
                        ret_tips = "Here are some memories you collected in your previous exploration:\n"
                        for mem in mem_list:
                            ret_tips += mem+"\n"

                        tip_list.append(ret_tips)
                        intrinsic_reward = 0.1 / (count+1)
                    else:
                        tip_list.append("")
                        intrinsic_reward = 0.1

                try:
                    if count > 0:
                        tip_obs = self._get_tip_obs(obs, ret_tips)
                        instructed_obs = self._get_instructed_obs(tip_obs, sep="")
                    else:
                        instructed_obs = self._get_instructed_obs(obs)

                    # Main agent step
                    with operation(step_count=step_count):
                        result = await self.agent._model_client.create(instructed_obs)
                    output = result.content
                    logger.info(f"[LLM output]: {output}")

                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Error during training rollout: {e}", exc_info=True)
                    break

                # Environment step
                pure_obs_for_mem.append([copy.deepcopy(obs), None])
                obs, pure_env_obs, executed_action, is_valid, step_reward, terminated, truncated, info = self.env.step(output, use_reasoning=self.config.captioner.type=="cot")
                history_actions_for_mem.append(executed_action)
                
                if rollout.mode == "train":
                    step_reward = reward_scale * step_reward
                emit_reward(
                    {
                        "extrinsic_reward": step_reward,
                        "intrinsic_reward": intrinsic_reward,
                    },
                    primary_key="extrinsic_reward",
                    attributes=make_link_attributes({"step_count": str(step_count)}),
                )

                episode_reward += float(step_reward)
                done = np.logical_or(terminated, truncated)

                step_count += 1

            if self.config.captioner.obs_type == "chat" and self.config.save_rollout:
                filename = f"empo2_rollouts/variant_{variation_idx}/step_{global_steps}/{rollout_id}_{round(episode_reward, 1)}_use_tip_{use_tips}.json"
                if use_tips:
                    _rollout = self._get_all_tip_obs(obs, tip_list)
                else:
                    _rollout = obs
                self._save_chat_rollout(_rollout, filename)

            if rollout.mode == "train":
                self.env.obs_type = "chat"
                self.env.prompt_builder.max_history = -1
                chat_obs = self.env.get_obs()
                chat_obs.pop()
                tip_obs = self._get_tip_generation_prompt(chat_obs)

                self.agent._model_client.max_tokens = 128
                result = await self.agent._model_client.create(tip_obs)
                tips = result.content
                logger.info(f"Tips: {tips}")
                
                content = re.search(r'<tip>(.*?)</tip>', tips, re.IGNORECASE | re.DOTALL)
                if content:
                    tips = content.group(1).strip()

                    #! Fill the ret and tip
                    for i in range(len(pure_obs_for_mem)):
                        max_score = 100 * reward_scale
                        pure_obs_for_mem[i][1] = tips + f'; At that timestep, the specific action your took was {history_actions_for_mem[i]}; Eventually you got the score {round(episode_reward, 1)}/{int(max_score)}.'

                    #! Generate the tips and save the mem
                    for i in range(len(pure_obs_for_mem)):
                        text = gather_chats(pure_obs_for_mem[i][0])
                        key = np.array(do_compress(text)['key']).reshape(-1, ).tolist()
                        content = pure_obs_for_mem[i][1]
                        score = episode_reward
                        add_memory(variation_idx, key, content, round(score, 1))

            if self.config.use_success_rate:
                return self.env.get_success_score() * reward_scale
            else:
                return episode_reward
        
        finally:
            if self.env is not None:
                self.env.close()