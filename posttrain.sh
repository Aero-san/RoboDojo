#训练wcm
GPUS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/posttrain/run_wcm.sh

# wcm+rltoken
scripts/posttrain/run_pi05_rltoken.sh \
    --wcm-checkpoint outputs/wcm/robodojo_pi05/deploy.pt \
    --output outputs/posttrain/pi05_wcm_actor.pt \
    --objective wcm_actor

# 直接微调behavior cloning+ wcm
bash scripts/posttrain/finetune_pi05_with_wcm.sh


# 评测
POSTTRAIN_MODE=wcm_actor \
POSTTRAIN_CHECKPOINT=$PWD/outputs/posttrain/pi05_wcm_actor.pt \
bash scripts/robodojo.sh eval \
   --policy-dir XPolicyLab/policy/Pi_05 \
   --policy-env openpi \
   --task stack_bowls \
   --ckpt RoboDojo-sim-arx_x5-joint-2 \
   --action-type joint \
   --eval-num 1 \
   --no-video


# evaluate recap
CUDA_VISIBLE_DEVICES=4 AAC_ENABLED=0 bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Pi_05 \
    --policy-env openpi \
    --task push_T \
    --ckpt "$(cat outputs/recap/push_T/latest_policy.txt)" \
    --env-cfg arx_x5 \
    --action-type joint \
    --eval-num 10


# rltoken+recap+wcm
RLTOKEN_RECAP_ROLLOUT_EPISODES=40 \
RLTOKEN_ROLLOUT_ENVS_PER_WORKER=4 \
RLTOKEN_ROLLOUT_GPUS=0,1,2,3,4,5,6,7 \
INITIAL_WCM_CHECKPOINT=$PWD/outputs/rltoken_recap/push_t/iteration_01/wcm/deploy.pt \
RLTOKEN_ACTOR_MODE=direct \
OUTPUT_ROOT=$PWD/outputs/rltoken_recap_direct \
bash scripts/posttrain/run_pi05_rltoken_recap.sh \
--task push_T \
--base-policy-checkpoint $PWD/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-2/59999 \
--wcm-train-gpus 0,1,2,3,4,5,6,7 \
--actor-train-gpus 0,1,2,3,4,5,6,7 \
--encoder-resume /share/mingyang/RoboDojo/outputs/rltoken_recap/push_t/iteration_01/encoder.pt \
--actor-mode direct 


# normal evaluate 
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/robodojo.sh eval   \
--policy-dir XPolicyLab/policy/Pi_05   \
--task stack_bowls   \
--ckpt checkpoints/RoboDojo-sim-arx_x5-joint-2   \
--policy-env openpi   \
--eval-num 1 \
--action-type joint \
--save-video

# evaluate wcm rltoken actor
POSTTRAIN_MODE=wcm_actor \
POSTTRAIN_CHECKPOINT="$(cat outputs/rltoken_recap_direct/push_t/latest_actor.txt)" \
bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Pi_05 \
    --policy-env openpi \
    --task push_T \
    --ckpt RoboDojo-sim-arx_x5-joint-2 \
    --env-cfg arx_x5 \
    --action-type joint \
    --eval-num 10

# world critic model + recap on pi05
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_GPUS=0,1,2,3 \
WCM_TRAIN_GPUS=0,1,2,3 \
NUM_TRAIN_STEPS=500 \
RECAP_VALUE_VIDEO_EPISODES=3 \
RECAP_VALUE_VIDEO_GPU=2 \
TASK_NAME=put_bottles_into_dustbin \
DEMO_ROOT=$PWD/data/RoboDojo_lerobot_v21_video \
RECAP_ITERATIONS=20 \
RECAP_ROLLOUT_EPISODES=10 \
RECAP_MAX_DEMO_EPISODES=100 \
INITIAL_WCM_CHECKPOINT=/share/mingyang/RoboDojo/outputs/wcm/robodojo_pi05/deploy.pt \
OPENPI_FSDP_DEVICES=2 \
OPENPI_BATCH_SIZE=32 \
WCM_PER_DEVICE_BATCH_SIZE=16 \
POLICY_GPU=0 \
ENV_GPU=1 \
bash scripts/posttrain/run_pi05_recap.sh --task put_bottles_into_dustbin --initial-policy-checkpoint $PWD/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-put_bottles_into_dustbin-arx_x5-joint-0/11999

recap 10000steps/9hrs