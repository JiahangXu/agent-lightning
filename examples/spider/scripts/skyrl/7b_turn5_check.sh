#!/bin/bash

set -e

export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export DATA_DIR=data
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="skyrl_check_7B_turn5_truncate8192_1011"
export PROJECT_NAME=AgentLightning

echo "Starting training script..."

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/skyrl_sql_653.parquet \
    data.val_files=${DATA_DIR}/test_dev_500.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    trainer.n_gpus_per_node=${N_GPUS} \
    agentlightning.port=9997 \
    data.train_batch_size=256 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.truncation='error' \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.default_local_dir=/mnt/teamdrive/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=50 $@

cd /scratch/amlt_code/ && python train_nvidia.py
# PYTHONPATH=$PYTHONPATH:../.. VERL_API_BASE=http://localhost:9997/ python sql_agent2.py \
#     --litsqlagent.trained-agents write \
#     --trainer.n-workers 16 \
#     --trainer.daemon true \
#     --litsqlagent.val-temperature 0 \
#     --litsqlagent.max-turns 5 \
#     --litsqlagent.table-info-truncate 8192 \
#     --litsqlagent.execution-truncate 8192