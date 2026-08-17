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


TASK_NAME=push_T \
DEMO_ROOT=$PWD/data/RoboDojo_lerobot_v21_video \
INITIAL_POLICY_CHECKPOINT=$PWD/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-2/59999 \
RECAP_ITERATIONS=3 \
RECAP_ROLLOUT_EPISODES=50 \
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
WCM_TRAIN_GPUS=0,1,2,3,4,5,6,7 \
OPENPI_FSDP_DEVICES=2 \
OPENPI_BATCH_SIZE=32 \
WCM_PER_DEVICE_BATCH_SIZE=16 \
POLICY_GPU=0 \
ENV_GPU=1 \
bash scripts/posttrain/run_pi05_recap.sh --task push_T --initial-policy-checkpoint $PWD/XPolicyLab/policy/Pi_05/checkpoints/RoboDojo-sim-arx_x5-joint-2/59999

CUDA_VISIBLE_DEVICES=4 AAC_ENABLED=0 bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Pi_05 \
    --policy-env openpi \
    --task push_T \
    --ckpt "$(cat outputs/recap/push_T/latest_policy.txt)" \
    --env-cfg arx_x5 \
    --action-type joint \
    --eval-num 10

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

  File "/share/mingyang/RoboDojo/third_party/curobo/curobo/_src/robot/parser/parser_urdf.py", line 54, in __init__
    self._robot = yourdfpy.URDF.load(
                  ^^^^^^^^^^^^^^^^^^^
  File "/share/mingyang/miniconda3/envs/RoboDojo/lib/python3.11/site-packages/yourdfpy/urdf.py", line 958, in load
    raise ValueError("{} is not a file".format(fname_or_file))
ValueError: /home/gmy/robodojo-work/RoboDojo/Assets/Robots/x5/X5A.urdf is not a file



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/robodojo.sh eval     --policy-dir XPolicyLab/policy/Pi_05     --policy-env openpi     --ckpt RoboDojo-sim-arx_x5-joint-0     --env-cfg arx_x5     --action-type joint     --eval-num 10     --task pour_by_language

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/robodojo.sh eval   \
--policy-dir XPolicyLab/policy/Pi_05   \
--task stack_bowls   \
--ckpt checkpoints/RoboDojo-sim-arx_x5-joint-2   \
--policy-env openpi   \
--eval-num 1 \
--action-type joint \
--save-video

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

