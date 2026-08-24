XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
scripts/posttrain/prepare_pi05_dataset.py \
--dataset-root data/RoboDojo_lerobot_v21_video \
--repo-id RoboDojo-fill_pen_holder-arx_x5-joint \
--task fill_pen_holder \
--mode video


cd /share/mingyang/RoboDojo/XPolicyLab/policy/Pi_05

OPENPI_FINETUNE_MODE=action_expert_lora \
OPENPI_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-sim-arx_x5-joint-0" \
OPENPI_BATCH_SIZE=64 \
OPENPI_NUM_WORKERS=8 \
OPENPI_NUM_TRAIN_STEPS=3000 \
OPENPI_LEARNING_RATE=1e-5 \
OPENPI_WARMUP_STEPS=500 \
OPENPI_DECAY_STEPS=3000 \
OPENPI_DECAY_LR=1e-6 \
OPENPI_WEIGHT_DECAY=1e-10 \
OPENPI_CLIP_GRADIENT_NORM=1.0 \
OPENPI_SAVE_INTERVAL=300 \
OPENPI_LOG_INTERVAL=150 \
OPENPI_FSDP_DEVICES=2 \
OPENPI_WANDB_ENABLED=0 \
bash train.sh RoboDojo put_bottles_into_dustbin arx_x5 joint 0 0,1,2,3

OPENPI_FINETUNE_MODE=action_expert_lora OPENPI_BATCH_SIZE=64 OPENPI_NUM_WORKERS=8 OPENPI_NUM_TRAIN_STEPS=12000 OPENPI_LEARNING_RATE=1e-5 OPENPI_WARMUP_STEPS=500 OPENPI_DECAY_STEPS=12000 OPENPI_DECAY_LR=1e-6 OPENPI_WEIGHT_DECAY=1e-10 OPENPI_CLIP_GRADIENT_NORM=1.0 OPENPI_SAVE_INTERVAL=300 OPENPI_LOG_INTERVAL=150 OPENPI_FSDP_DEVICES=2 OPENPI_WANDB_ENABLED=0 OPENPI_RESUME=1 bash train.sh RoboDojo put_bottles_into_dustbin arx_x5 joint 0 0,1

40min/ksteps

# evaluate
CUDA_VISIBLE_DEVICES=2,3 bash scripts/robodojo.sh eval     --policy-dir XPolicyLab/policy/Pi_05     --policy-env openpi     --ckpt RoboDojo-fill_egg_holder-arx_x5-joint-0-full   --env-cfg arx_x5     --action-type joint     --eval-num 50     --task fill_egg_holder


# evaluate
CUDA_VISIBLE_DEVICES=4,5 bash scripts/robodojo.sh eval     --policy-dir XPolicyLab/policy/Pi_05     --policy-env openpi     --ckpt RoboDojo-put_bottles_into_dustbin-arx_x5-joint-0   --env-cfg arx_x5     --action-type joint     --eval-num 1000     --task put_bottles_into_dustbin \
--action-noise-viz \
--noise-viz-method umap \
--noise-viz-k 5

CUDA_VISIBLE_DEVICES=6,7 bash scripts/robodojo.sh eval     --policy-dir XPolicyLab/policy/Pi_05     --policy-env openpi     --ckpt RoboDojo-fill_egg_holder-arx_x5-joint-0 --env-cfg arx_x5     --action-type joint     --eval-num 50     --task fill_egg_holder \
--action-noise-viz \
--noise-viz-method umap \
--noise-viz-k 5

CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/robodojo.sh benchmark     --policy-dir XPolicyLab/policy/Pi_05     --policy-env openpi     --ckpt RoboDojo-put_bottles_into_dustbin-arx_x5-joint-0   --env-cfg arx_x5     --action-type joint     --eval-num native  --gpu-ids 4,5,6,7 --policy-gpu-ids 4,6 --env-gpu-ids 5,7 --dimension long_horizon,open \
--action-noise-viz \
--noise-viz-method umap \
--noise-viz-k 7


OPENPI_PARAMETER_DTYPE=bfloat16 \
OPENPI_FINETUNE_MODE=full \
OPENPI_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-sim-arx_x5-joint-0" \
OPENPI_BATCH_SIZE=16 \
OPENPI_NUM_WORKERS=1 \
OPENPI_NUM_TRAIN_STEPS=15000 \
OPENPI_LEARNING_RATE=1e-5 \
OPENPI_WARMUP_STEPS=1500 \
OPENPI_DECAY_STEPS=15000 \
OPENPI_DECAY_LR=1e-6 \
OPENPI_WEIGHT_DECAY=1e-10 \
OPENPI_CLIP_GRADIENT_NORM=1.0 \
OPENPI_SAVE_INTERVAL=500 \
OPENPI_LOG_INTERVAL=500 \
OPENPI_FSDP_DEVICES=4 \
OPENPI_WANDB_ENABLED=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash train.sh RoboDojo fill_egg_holder arx_x5 joint 0 4,5,6,7 #--overwrite
 
 OPENPI_PARAMETER_DTYPE=bfloat16：避免 full 模式参数保持 float32。                                                                                             
  - OPENPI_EMA_DECAY=None：关闭默认 EMA，减少一份完整模型参数副本。
  - 默认全局 batch size 是 256，4090 建议从 16 或 32 开始逐步增加。
  - OPENPI_FSDP_DEVICES=4 已经是当前 4 卡下的最大参数切分。

OPENPI_FINETUNE_MODE=action_expert_lora OPENPI_BATCH_SIZE=64 OPENPI_NUM_WORKERS=8 OPENPI_NUM_TRAIN_STEPS=10000 OPENPI_LEARNING_RATE=1e-5 OPENPI_WARMUP_STEPS=500 OPENPI_DECAY_STEPS=10000 OPENPI_DECAY_LR=1e-6 OPENPI_WEIGHT_DECAY=1e-10 OPENPI_CLIP_GRADIENT_NORM=1.0 OPENPI_SAVE_INTERVAL=300 OPENPI_LOG_INTERVAL=150 OPENPI_FSDP_DEVICES=2 OPENPI_WANDB_ENABLED=0 OPENPI_RESUME=1 bash train.sh RoboDojo fill_pen_holder arx_x5 joint 0 0,1,2,3


OPENPI_SHARDING_STRATEGY=full_shard \
OPENPI_CPU_OFFLOAD=1 \
OPENPI_PARAMETER_DTYPE=bfloat16 \
OPENPI_FINETUNE_MODE=full \
OPENPI_BATCH_SIZE=8 \
OPENPI_NUM_WORKERS=1 \
OPENPI_NUM_TRAIN_STEPS=1000 \
OPENPI_LEARNING_RATE=1e-5 \
OPENPI_WARMUP_STEPS=500 \
OPENPI_DECAY_STEPS=1000 \
OPENPI_DECAY_LR=1e-6 \
OPENPI_WEIGHT_DECAY=1e-10 \
OPENPI_CLIP_GRADIENT_NORM=1.0 \
OPENPI_SAVE_INTERVAL=100 \
OPENPI_LOG_INTERVAL=100 \
OPENPI_FSDP_DEVICES=2 \
OPENPI_WANDB_ENABLED=0 \
OPENPI_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-sim-arx_x5-joint-0" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
bash train.sh RoboDojo fill_pen_holder arx_x5 joint 0


OPENPI_PARAMETER_DTYPE=float32 OPENPI_FINETUNE_MODE=full OPENPI_BATCH_SIZE=16 OPENPI_NUM_WORKERS=4 OPENPI_NUM_TRAIN_STEPS=120000 OPENPI_LEARNING_RATE=5e-6 OPENPI_WARMUP_STEPS=3000 OPENPI_DECAY_STEPS=120000 OPENPI_DECAY_LR=5e-7 OPENPI_WEIGHT_DECAY=1e-10 OPENPI_CLIP_GRADIENT_NORM=1.0 OPENPI_SAVE_INTERVAL=5000 OPENPI_LOG_INTERVAL=100 OPENPI_FSDP_DEVICES=4 OPENPI_WANDB_ENABLED=0 OPENPI_RESUME=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OPENPI_SHARDING_STRATEGY=no_shard bash train.sh RoboDojo fill_egg_holder arx_x5 joint 0 4,5,6,7

rsync -avh -P -e "ssh -p 2370 -i ~/.ssh/mecs.id" --info=progress2 --partial *tar.gz mingyang@36.103.234.242:~