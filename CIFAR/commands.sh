cd /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main
conda activate pt110
nohup ./run.sh  > tmp.txt 2>&1 &
unset COMET_OPTIMIZER_ID
export COMET_OPTIMIZER_ID=$(comet optimize opt.json)
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py  --epochs 40 --val_rat 0.1 --dls_act linear --norm_loss --tuning > tmp1.txt 2>&1 &
comet upload /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CometData_tuning4/*.zip > tmp.txt 2>&1 &

ps -ef|grep train_tuning|grep -v grep|cut -c 9-15|xargs kill -9
tmux attach -t 8


# {
#     "algorithm": "bayes",                
#     "parameters": {
#         "dls_start": {"type": "float", "min": 0.0, "max": 0.3},
#         "dls_end":{"type": "float", "min": 0.0, "max": 0.3},      
#         "dls_coe0": {"type": "float", "min": 0.0, "max": 10.0}
#     },
#     "spec": {
#     "maxCombo": 10000,
#     "metric": "val_acc_SumDatasets",
#         "objective": "maximize"
#     }
# }