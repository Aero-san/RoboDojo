XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
scripts/posttrain/prepare_pi05_dataset.py \
--dataset-root data/RoboDojo_lerobot_v21_video \
--repo-id RoboDojo-stack_bowls-arx_x5-joint \
--task stack_bowls \
--mode video


cd /share/mingyang/RoboDojo/XPolicyLab/policy/Pi_05


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
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0,1,2,3