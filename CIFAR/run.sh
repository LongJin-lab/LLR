# Table I, II
CUDA_VISIBLE_DEVICES=0 comet optimize train_tuning.py configs.json > tuning0.txt 2>&1 &
sleep 1s
CUDA_VISIBLE_DEVICES=1 comet optimize train_tuning.py configs.json > tuning1.txt 2>&1 &
sleep 2s
CUDA_VISIBLE_DEVICES=2 comet optimize train_tuning.py configs.json > tuning2.txt 2>&1 &
sleep 3s
CUDA_VISIBLE_DEVICES=3 comet optimize train_tuning.py configs.json > tuning3.txt 2>&1 &
sleep 2.5s
CUDA_VISIBLE_DEVICES=4 comet optimize train_tuning.py configs.json > tuning4.txt 2>&1 &
sleep 4s
CUDA_VISIBLE_DEVICES=5 comet optimize train_tuning.py configs.json > tuning5.txt 2>&1 &
sleep 3s
CUDA_VISIBLE_DEVICES=6 comet optimize train_tuning.py configs.json > tuning6.txt 2>&1 &
sleep 2s
CUDA_VISIBLE_DEVICES=7 comet optimize train_tuning.py configs.json > tuning7.txt 2>&1 &
sleep 1s
CUDA_VISIBLE_DEVICES=8 comet optimize train_tuning.py configs.json > tuning8.txt 2>&1 &
sleep 2s
CUDA_VISIBLE_DEVICES=9 comet optimize train_tuning.py configs.json > tuning9.txt 2>&1 



# Fig. 3
CUDA_VISIBLE_DEVICES=0 python train_tuning.py --model resnet50 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset SVHN_base  --seed 1234 --norm_loss --plot_grad > tuning0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_tuning.py --model resnet50 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset SVHN_cutout  --seed 1234 --norm_loss --plot_grad > tuning1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_tuning.py --model mobilenetv2_x0_5 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset SVHN_base  --seed 1234 --norm_loss --plot_grad > tuning2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_tuning.py --model mobilenetv2_x0_5 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset SVHN_cutout  --seed 1234 --norm_loss --plot_grad > tuning3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python train_tuning.py --model resnet18 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR10_base  --seed 1234 --norm_loss --plot_grad > tuning0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_tuning.py --model resnet18 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR10_cutout  --seed 1234 --norm_loss --plot_grad > tuning1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python train_tuning.py --model repvgg_a1 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR10_base  --seed 1234 --norm_loss --plot_grad > tuning6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python train_tuning.py --model repvgg_a1 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR10_cutout  --seed 1234 --norm_loss --plot_grad > tuning7.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python train_tuning.py --model resnet18 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR100_base  --seed 1234 --norm_loss --plot_grad > tuning0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_tuning.py --model resnet18 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR100_cutout  --seed 1234 --norm_loss --plot_grad > tuning1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python train_tuning.py --model repvgg_a1 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR100_base  --seed 1234 --norm_loss --plot_grad > tuning6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python train_tuning.py --model repvgg_a1 --setting singlerun --plot_surf --dls_coe0 1.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_start 1.0 --dls_end 0.9 --dataset CIFAR100_cutout  --seed 1234 --norm_loss --plot_grad > tuning7.txt 2>&1 &




# Fig. 2
CUDA_VISIBLE_DEVICES=2 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 10.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act linear --dls_start 0.8 --dls_end 0.7 --dataset CIFAR100_cutout  --seed 1234 --norm_loss > tuning2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=8 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 10.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act linear --dls_start 0.3 --dls_end 0.2 --dataset CIFAR100_cutout  --seed 1234 --norm_loss > tuning8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=9 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 10.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act linear --dls_start 0.1 --dls_end 0.0 --dataset CIFAR100_cutout  --seed 1234 --norm_loss > tuning9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 10.0 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act linear --dls_start -0.1 --dls_end -1.0 --dataset CIFAR100_cutout  --seed 1234 --norm_loss > tuning5.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.1 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.2 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.3 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.4 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.5 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.6 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.7 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning7.txt 2>&1 &
CUDA_VISIBLE_DEVICES=8 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.8 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=9 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.9 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout  --calm_epoch 50 --seed 1234 > tuning9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.99 --dls_coe1 1.0 --dls_coe2 1.0 --epoch 200 --dls_act logexp --dataset CIFAR100_cutout --calm_epoch 50 --seed 1234 > tuning5.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_Identity_set_singlerun_coe0_0.0coe1_Nonecoe2_NoneNone_None3None1.02023-08-15-22:05:45/199_None_None_0.0_None_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,31].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.6_0.53None1.0None2023-08-18-20:27:15/199_0.6_0.5_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.2_0.13None1.0None2023-08-18-20:27:15/199_0.2_0.1_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.5_0.43None1.0None2023-08-18-20:27:15/199_0.5_0.4_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.8_0.73None1.0None2023-08-18-20:27:16/199_0.8_0.7_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.9_0.83None1.0None2023-08-18-20:27:16/199_0.9_0.8_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=6 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.01.0_0.93None1.0None2023-08-18-20:27:16/199_1.0_0.9_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=7 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.7_0.63None1.0None2023-08-18-20:27:16/199_0.7_0.6_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=8 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.3_0.23None1.0None2023-08-18-20:27:16/199_0.3_0.2_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=9 python plot_1D.py --surf_file /media/bdc/clm/DeformingTheLossSurface/Sparse-Sharpness-Aware-Minimization-main/logs/CIFAR100_cutout_bsz128_epoch200_resnet18_lr0.05_sgd_seed1234act_linear_set_singlerun_coe0_10.0coe1_1.0coe2_1.00.1_0.03None1.0None2023-08-18-20:27:16/199_0.1_0.0_10.0_1.0_states_xignore=biasbn_xnorm=filter.h5_[-0.2,0.2,51].h5 > tuning0.txt 2>&1 &






# CUDA_VISIBLE_DEVICES=0 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 1.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 3.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning1.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 5.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 7.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 9.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning4.txt 2>&1 &
# # CUDA_VISIBLE_DEVICES=5 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 0.1 --epoch 200 --dls_act sech > tuning4.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 0.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning6.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=7 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 2.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning7.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=8 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 4.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning8.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=9 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 6.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning9.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python train_tuning.py --setting singlerun_calmdown --plot_surf --dls_coe0 8.5 --dls_coe1 0.15 --epoch 200 --dls_act sech --dataset CIFAR100_cutout --plus_cutout 1.0 --seed 1234 --norm_loss > tuning5.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python train_tuning.py --setting singlerun --plot_surf --dls_coe0 0.1 --epoch 200 --dls_act sech --dataset CIFAR100_cutout > tuning4.txt 2>&1 &


