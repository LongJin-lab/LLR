import os
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import random
import errno
from random import randint

# from sympy import sec 

def gpu_info(GpuNum):
    gpu_status = os.popen('nvidia-smi -i '+str(GpuNum)+' | grep %').read().split('|')
    # print('gpu_status', gpu_status)
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split(
        '   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

def get_A(B_ab):
    A_ab = np.array([0]*len(B_ab))
    A_ab[0] = A_ab[0]+1
    A_ab = ' '+str(A_ab).replace('[', ' ').replace(']', ' ')+' '
    return A_ab


def SearchAndExe(Gpus, cmd, interval):
    prefix = 'CUDA_VISIBLE_DEVICES='
    foundGPU = 0
    while foundGPU==0:  # set waiting condition

        for u in Gpus: 
            gpu_power, gpu_memory = gpu_info(u)      
            cnt = 0   
            first = 0
            second = 0   
            empty = 1
            print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
            for i in range(12):
                gpu_power, gpu_memory = gpu_info(u)   
                print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
                if gpu_memory > 2000 or gpu_power > 150: # running
                    empty = 0
                time.sleep(interval)
            if empty == 1:
                foundGPU = 1
                break
            
    if foundGPU == 1:
        prefix += str(u)
        cmd = prefix + ' '+ cmd
        print('\n' + cmd)
        os.system(cmd)
    
def rand_port():
    r = ' '
    r += str(random.randint(1, 5))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += " "
    return r

def add_sp(str):
    return ' '+str+' '  

GPUS = [0]

cmd = ' python3 train.py --dataset CIFAR10_cutout --model resnet18 --val_rat 0.1 --num_workers 4 --epochs 200 > base1.txt 2>&1 & '
SearchAndExe(GPUS, cmd, interval=3)

cmd = ' python3 train.py --dataset CIFAR10_cutout --model resnet18 --val_rat 0.1 --num_workers 4 --epochs 200 > base2.txt 2>&1 & '
SearchAndExe(GPUS, cmd, interval=3)

# cmd = ' python3 train.py --dataset CIFAR100_cutout --model resnet18 --val_rat 0.1 --num_workers 4 --epochs 200 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = ' python3 train.py --dataset CIFAR100_cutout --model resnet18 --val_rat 0.1 --dls_act logexp --dls_coe0 0.0 --dls_coe1 3.0  --num_workers 4 --epochs 200 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = ' python3 train.py --dataset CIFAR100_cutout --model resnet18 --val_rat 0.1 --dls_act logexp --dls_coe0 0.0 --dls_coe1 3.0  --num_workers 4 --swa --epochs 200 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = ' python3 train.py --dataset CIFAR100_cutout --model resnet18 --val_rat 0.1 --dls_act atan --dls_coe0 0.0 --dls_coe1 3.0  --num_workers 4 --epochs 200 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = ' python3 train.py --dataset CIFAR100_cutout --model resnet18 --val_rat 0.1 --dls_act atan --dls_coe0 0.0 --dls_coe1 3.0  --num_workers 4 --swa --epochs 200 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

#auto
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet18  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --swa  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet18 --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# #auto
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet50  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet50 --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet34  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet34 --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)


# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model wideresnet34x10  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model wideresnet34x10 --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# #auto
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model wideresnet28x10  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model wideresnet28x10 --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 1 5 --dls_act atan  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)



# #auto
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet152  --datadir /media2/datasets/data/cifar10/ --val_rat 0.  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet152 --datadir /media2/datasets/data/cifar100/ --val_rat 0.  > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)


# cmd = 'nohup python3 train.py --dataset CIFAR10_auto --model resnet18  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.01 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_auto --model resnet18  --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.05 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = 'nohup python3 train.py --dataset CIFAR10_auto --model resnet50  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.01 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_auto --model resnet50  --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.05 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)


# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet18  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.01 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet18  --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.05 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)
# cmd = 'nohup python3 train.py --dataset CIFAR10_base --model resnet50  --datadir /media2/datasets/data/cifar10/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.01 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)

# cmd = 'nohup python3 train.py --dataset CIFAR100_base --model resnet50  --datadir /media2/datasets/data/cifar100/ --val_rat 0. --dls_coe 0.99 --swa_lr 0.05 > temp.txt 2>&1 & '
# SearchAndExe(GPUS, cmd, interval=3)