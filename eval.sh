#!/bin/bash

PID=1685026 # 替换为目标进程的 PID
while true; do
    if ! ps -p $PID > /dev/null; then
        echo "进程已结束"
        sleep 10
        break
    fi
    sleep 60
done

AAC_ENABLED=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/robodojo.sh benchmark  \
--policy-dir XPolicyLab/policy/Pi_05 \
--policy-env openpi \
--ckpt RoboDojo-sim-arx_x5-joint-0 \
--action-type joint  \
--eval-num native \
--seed 0 \
--dimension open,long-horizon \
--gpu-ids 0,1,2,3,4,5,6,7