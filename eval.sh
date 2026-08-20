#!/bin/bash

sleep 2400
PID= 135827  # 替换为目标进程的 PID
while true; do
    if ! ps -p $PID > /dev/null; then
        echo "进程已结束"
        sleep 10
        break
    fi
    sleep 60
done

OPENPI_FINETUNE_MODE=action_expert_lora \ 
OPENPI_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-sim-arx_x5-joint-0" \
OPENPI_BATCH_SIZE=64 \
OPENPI_NUM_WORKERS=8 \
OPENPI_NUM_TRAIN_STEPS=1000 \
OPENPI_LEARNING_RATE=1e-5 \
OPENPI_WARMUP_STEPS=500 \
OPENPI_DECAY_STEPS=1000 \
OPENPI_DECAY_LR=1e-6 \
OPENPI_WEIGHT_DECAY=1e-10 \
OPENPI_CLIP_GRADIENT_NORM=1.0 \
OPENPI_SAVE_INTERVAL=100 \
OPENPI_LOG_INTERVAL=50 \
OPENPI_FSDP_DEVICES=2 \
OPENPI_WANDB_ENABLED=0 \
bash train.sh RoboDojo put_bottles_into_dustbin arx_x5 joint 0 0,1,2,3