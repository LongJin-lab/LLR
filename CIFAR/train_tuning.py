from comet_ml import Experiment, OfflineExperiment, Optimizer
# import torch.utils.bottleneck as btn

import os
import time
import datetime

import torch
import copy
import re

from models.build import build_model
from data.build import build_dataset, build_train_dataloader, build_val_dataloader, build_test_dataloader
from solver.build import build_optimizer, build_lr_scheduler

from utils.logger import Logger
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.engine import *#train_one_epoch, train_one_epoch_LPF, train_one_epoch_smoothout, evaluate, evaluate_val

import numpy as np
# from turtle import color
import torch
import matplotlib.pyplot as plt
import matplotlib
# from torchcontrib.optim import SWA
import torchcontrib
import numpy as np
from thop import profile
import sys
import os
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import evaluation
import h5py
import scheduler
# import mpi4pytorch as mpi
from plot_surface import *
import random
from LLR import *
import numpy as np
# from turtle import color

import matplotlib.cm as cm

matplotlib.use('Agg')

FAILURE_MODE = True
# use_mpi = False
# if use_mpi:
#     comm = mpi.setup_MPI()
#     rank, nproc = comm.Get_rank(), comm.Get_size()
# else:
comm, rank, nproc = None, 0, 1
   

def plot_grad(grads, num_dims, losses, args, namebase):
    
    plt.style.use(['ieee'])
    # plt.style.use(['seaborn-paper'])
    matplotlib.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
            }
    )
    plt.tight_layout()    
    if grads.device != 'cpu':
        grads.to('cpu')
    steps = list(range(0, grads.shape[0]))
    corr_losses = np.corrcoef(losses, grads.T)
    corr_steps = np.corrcoef(steps, grads.T)
    n_dim = grads.shape[1]
    dim = np.arange(0,n_dim)

    fig, ax = plt.subplots()
    ax.scatter(dim,corr_losses[0,1:],alpha=0.2, label="Gradient-loss correlation", color=(0.15, 0.25, 0.4))
    ax.scatter(dim,np.abs(corr_steps[0,1:]),alpha=0.5, label="Gradient-iteration correlation", marker='^', s=7, color=(0.5,0.1, 0.1))

    torch.save(torch.tensor(dim), namebase+'dim.pt')
    torch.save(torch.tensor(corr_losses), namebase+'corr_losses.pt')
    torch.save(torch.tensor(corr_steps), namebase+'corr_steps.pt')
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Pearson correlation coefficient')
    ax.legend()
    fname = namebase+'dim'+str(num_dims)+'correlation.pdf'
    print('fname',fname)
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    fig, ax = plt.subplots()
    ax.set_xlabel('Loss')
    ax.set_ylabel('Gradient norm')
    for i in range(grads.size(1)):
        # ax.plot(losses, grads[:, i], alpha=0.05, linewidth=1, linestyle='-',color=(0.15, 0.25, 0.4))
        ax.scatter(losses,grads[:, i],alpha=0.5, s=0.2,color=(0.1, 0.25, 0.4))
    torch.save(torch.tensor(losses), namebase+'losses.pt')
    torch.save(torch.tensor(grads), namebase+'grads.pt')
    # plt.show()
    fname = namebase+'dim'+str(num_dims)+'gradient_Loss.pdf'
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient norm')
    for i in range(grads.size(1)):
        # ax.plot(grads[:, i], alpha=0.05, linewidth=1, linestyle='-', color=(0.5,0.1, 0.1))  
        ax.scatter(steps, grads[:, i],alpha=0.5, marker='^',s=0.2,color=(0.5,0.1, 0.1))   
    # plt.show()
    fname = namebase+'dim'+str(num_dims)+'gradient_Iteration.pdf'
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

def plot_grad_norm(grads, num_dims, losses, args, namebase):
    namebase = namebase+'/'
    plt.style.use(['ieee'])
    # plt.style.use(['seaborn-paper'])
    matplotlib.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
            }
    )
    plt.tight_layout()  
    print('grads.device',grads.device)  
    if grads.device != 'cpu':
        grads = grads.to('cpu')
    # print('grads',grads)  
    steps = list(range(0, grads.shape[0]))

    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Gradient norm')
    # for i in range(grads.size(0)):
        # ax.plot(losses, grads[:, i], alpha=0.05, linewidth=1, linestyle='-',color=(0.15, 0.25, 0.4))
    ax.scatter(losses,grads,alpha=0.3, s=0.2,color=(0.1, 0.25, 0.4))
    torch.save(torch.tensor(losses), namebase+'losses.pt')
    torch.save(torch.tensor(grads), namebase+'grads.pt')
    # plt.show()
    fname = namebase+'dim'+str(num_dims)+'gradient_norm_Loss.pdf'
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    # ax.set_yscale('log')    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient norm')
    # for i in range(grads.size(0)):
        # ax.plot(grads[:, i], alpha=0.05, linewidth=1, linestyle='-', color=(0.5,0.1, 0.1))  
    ax.scatter(steps, grads,alpha=0.3, marker='^',s=0.2,color=(0.5,0.1, 0.1))   
    # plt.show()
    fname = namebase+'dim'+str(num_dims)+'gradient_norm_Iteration.pdf'
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')
    

def plot_grad_norm_color(grads, num_dims, losses, args, namebase):
    namebase = namebase+'/'
    plt.style.use(['ieee'])
    # plt.style.use(['seaborn-paper'])
    matplotlib.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
            'font.size': 8,  # Change the font size to 16

            }
    )
    colors = cm.RdBu(np.linspace(0, 1, 10))
    losses = np.array(losses)      # Convert the list to a NumPy array
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    plt.tight_layout()  
    print('grads.device',grads.device)  
    if grads.device != 'cpu':
        grads = grads.to('cpu')
    steps = torch.linspace(0, grads.shape[0], grads.shape[0])
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Loss', fontsize='xx-large')
    ax.set_ylabel('Gradient norm', fontsize='xx-large')    
    # ax.set_xlim(0,0.01)
    # ax.set_ylim(0,2)
    # for i in range(grads.size(0)):
        # ax.plot(losses, grads[:, i], alpha=0.05, linewidth=1, linestyle='-',color=(0.15, 0.25, 0.4))
    # ax.scatter(losses,grads,alpha=0.3, s=0.2,color=(0.1, 0.25, 0.4))
    mask = np.logical_and(losses >= 0, losses <= max(losses))
    
    # mask = np.logical_and(losses >= 0, losses <= 0.01)
    # mask = np.logical_and(losses >= 2.5, losses <= max(losses))

    losses = losses[mask]
    grads = grads[mask]
    steps = steps[mask]
    # scatter = ax.scatter(losses,grads,alpha=0.8, s=0.2,c=steps)
    scatter = ax.scatter(losses,grads,alpha=0.8, s=5,c=steps)
    
    cb = plt.colorbar(scatter)
    cb.set_label('Iteration', fontsize='xx-large')
    plt.tick_params( 'both', labelsize='large')
    fname = namebase+cur_time+'dim'+str(num_dims)+'gradient_norm_Loss.pdf'
    plt.tight_layout()  
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    # ax.set_yscale('log')    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient norm')
    ax.scatter(steps, grads,alpha=0.3, marker='^',s=0.2,color=(0.5,0.1, 0.1))   
    # plt.show()
    fname = namebase+cur_time+'dim'+str(num_dims)+'gradient_norm_Iteration.pdf'
    print('fname',fname)
    # plt.show()
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')
        
def torch_coorcoef(A,B):
    A_tensor = torch.tensor(A).cuda()
    B_tensor = torch.tensor(B).cuda()
    combined_tensor = torch.stack((A_tensor, B_tensor), dim=0)
    corr_matrix = torch.corrcoef(combined_tensor)
    return corr_matrix[0, 1:]
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
def main(args,experiment=None):
    seed_torch(seed=int(args.seed))
    if 'CIFAR10_' in args.dataset:
        args.rho = 0.1
    if 'CIFAR100_' in args.dataset:
        args.rho = 0.2        
    def sam_hyper_param(args):
        args_opt = args.opt.split('-')
        if len(args_opt) == 1:
            return []
        elif len(args_opt) == 2:
            sam_opt, base_opt = args_opt[0], args_opt[1]
        # SAM, SSAMF, SSAMD
        output_name = ['rho{}'.format(args.rho)]
        if sam_opt[:4].upper() == 'SSAM':
            output_name.extend(['s{}u{}'.format(args.sparsity, args.update_freq), 'D{}{}'.format(args.drop_rate, args.drop_strategy), 'R{}'.format(args.growth_strategy), 'fisher-n{}'.format(args.num_samples)])
        return output_name    
    
    if 1:#args.output_name is None:
        args.output_name = '_'.join([
            args.dataset,
            'bsz' + str(args.batch_size),
            'epoch' + str(args.epochs),
            args.model,
            'lr' + str(args.lr),
            str(args.opt),
        ] + sam_hyper_param(args) + ['seed{}'.format(args.seed)])    
        
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # args.output_dir = args.output_dir
    save_dir = args.output_dir+args.output_name+'act_'+str(args.dls_act)+'_set_'+str(args.setting)+'_coe0_'+str(args.dls_coe0)+'coe1_'+str(args.dls_coe1)+'coe2_'+str(args.dls_coe2)+str(args.dls_start)+'_'+str(args.dls_end)+str(args.rep_num)+str(args.plus_base)+str(args.end)+str(args.StartLast)+cur_time
    print('save_dir', save_dir)
    comet_dir = "./logs/CometData/circ"
    if not os.path.exists(comet_dir):
        os.makedirs(comet_dir)
    if experiment is None:
        experiment = Experiment(
        # experiment = OfflineExperiment(
            # offline_directory=comet_dir,            
            api_key="KbJPNIfbsNUoZJGJBSX4BofNZ",
            project_name="Deforming-CIFAR-temp",#"deforming-cifar-sam",#
            workspace="logichen",
            # auto_histogram_weight_logging=True,
        )
    hyper_params = vars(args)
    experiment.log_parameters(hyper_params)
    # init seed
    # setup_seed(args)

    # init dist
    init_distributed_model(args)

    # init log
    logger = Logger(args,save_dir)
    # logger.log(args)

    # build dataset and dataloader
    train_data, val_data, test_data, n_classes = build_dataset(args)
    # print('len(val_data):',len(val_data))
    train_loader = build_train_dataloader(
        train_dataset=train_data,
        args=args
    )
    test_loader = build_test_dataloader(
        test_dataset=test_data,
        args=args
    )
    val_loader = build_val_dataloader(
        val_dataset=val_data,
        args=args
    )    
    args.n_classes = n_classes
    # logger.log(f'Train Data: {len(train_data)}, Test Data: {len(test_data)}.')

    # build model
    model = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    logger.log(f'Model: {args.model}')

    # build loss
    criterion = torch.nn.CrossEntropyLoss()
    if 'Inner_loss' in args.setting:
        criterion = CrossEntropyLoss_LLR()
    elif 'Loss_Act' in args.setting:
        criterion = CrossEntropyLoss_act()
    # build solver
    opt, base_optimizer = build_optimizer(args, model=model_without_ddp)
    swa_rat = 0.75
    if args.swa:
        print('Using SWA')
        # if 'CIFAR10_' in args.dataset:
        #     args.swa_lr = 0.01
        # elif 'CIFAR100_' in args.dataset:
        #     args.swa_lr = 0.05
        # elif 'SVHN_' in args.dataset:
        #     args.swa_lr = 0.01
        print('int(swa_rat*(args.epochs)',int(swa_rat*(args.epochs)))
        # optimizer = torchcontrib.optim.SWA(opt, swa_start=int(swa_rat*(args.epochs)), swa_freq=1, swa_lr=args.swa_lr)
        optimizer = torchcontrib.optim.SWA(opt,swa_freq=1)#,swa_lr=args.swa_lr)
    else:
        optimizer = opt
        print('Using SGD')
    if 'LossSche' in args.setting or 'LossRepLR' in args.setting:
        args.lr_scheduler = 'MultiStepLRscheduler'
        args.gamma = 1
        
    lr_scheduler = build_lr_scheduler(args, optimizer=base_optimizer)
    logger.log(f'Optimizer: {type(optimizer)}')
    logger.log(f'LR Scheduler: {type(lr_scheduler)}')

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr_scheduler.step(args.start_epoch)
        logger.log(f'Resume training from {args.resmue_path}.')


    # start train:
    # logger.log(f'Start training for {args.epochs} Epochs.')
    start_training = time.time()
    max_acc_test = 0.0
    max_acc_val = 0.0
    grads = []
    losses = []

    if args.plot_grad: 
        inputs = torch.rand(1,3,32,32).cuda()
        flops, params = profile(model, inputs=(inputs, ))
        params = int(params)
        print('flops, params',flops, params)
        num_dims = params#500        
        tracked_dims = np.random.choice(params, num_dims, replace=False)
    else:
        tracked_dims = None 
        
    # dls_coe = nn.Parameter(torch.ones(1)*args.dls_coe, requires_grad=True).cuda()
    # dls_coe = args.dls_coe
    tr_te_loss_gapS = []
    tr_te_acc_gapS= []
    tr_va_loss_gapS= []
    tr_va_acc_gapS= []
    tr_losses = []
    te_losses = []
    va_losses = []
    threshold = 10
    last = 3
    train_stats = None
    dls_coe = None
    use_dls = None
    mean = 0
    std = 0
    error = 0
    prev_error = 0#-acceptable_gap
    int_error = 0
    kp = 0.5
    ki = 0.01
    kd = 0.5   
    control_signal = 0
    shift = 0
    ini_loss = None 
    for epoch in range(args.start_epoch, args.epochs):
        tracked_grad = []
        start_epoch = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        # cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # experiment.log_metric('cur_lr', cur_lr, step=epoch)
        # cur_lr2 = optimizer.param_groups[0]['lr']
        # experiment.log_metric('cur_lr2', cur_lr2, step=epoch)
        # if epoch == 1:
        #     # train_stats = evaluate(model, train_loader)
        #     if args.norm_loss:
        #         ini_loss = train_stats['train_loss']# it is training actually

        #         # ini_loss = train_stats['test_loss']# it is training actually
        #         print('ini_loss',ini_loss)        
        if epoch == 0:
            train_stats = evaluate(model, train_loader)
            if args.norm_loss:
                ini_loss = train_stats['test_loss']# it is training actually
                print('ini_loss',ini_loss)
            else:
                ini_loss = None       
                print('no_ini_loss')             
        if train_stats and args.dls_act:
            if args.pid:
                dls_coe = [control_signal]+[args.dls_coe1]
                use_dls = True
            else:
                dls_coe = [args.dls_coe0]+[args.dls_coe1]   
                use_dls = True         

            if args.dls_start:
                # if train_stats['train_loss']/ini_loss<=args.dls_start and train_stats['train_loss']/ini_loss>=args.dls_end:
                    # print('Using DLS, train_loss/ini_loss', train_stats['train_loss']/ini_loss)
                dls_coe = [args.dls_coe0]+[args.dls_coe1]
                use_dls = True
                # else:
                #     dls_coe = None
            
            # if (epoch + 1) >= int(0.9*(args.epochs)):
            #     dls_coe = None
            #     use_dls = None
            
            # if len(tr_losses) > last:
            #     mean = sum(tr_losses[-last:]) / last
            #     std = torch.tensor(tr_losses[-last:]).std()
            #     shift = mean-std
            #     shift = 0
            #     experiment.log_metric('loss_mean', mean, step=epoch)
            #     experiment.log_metric('loss_std', std, step=epoch)
            #     # print('mean, std', mean, std)
            #     if std < threshold:
            #         # dls_coe = [args.dls_coe0]+[args.dls_coe1]
            #         dls_coe = [control_signal]+[args.dls_coe1]

            #     else:
            #         dls_coe = None
        if (epoch + 1) >= int(args.epochs-args.calm_epoch) and "calmdown" in args.setting:
                use_dls = None
        if args.StartLast:
            if (epoch + 1) <= int(args.epochs)-int(args.StartLast):
                use_dls = None
            else:
                use_dls = True
        if  not args.dls_act:  
            use_dls = None                
                # args.dls_coe0 = 1.0
                # args.dls_coe1 = 0.0
                # args.dls_act = 'linear'
                # args.dls_start = -9999.0
                # args.dls_end = -99999.0
        if 'LPF' in args.setting:
            train_stats, grads, losses = train_one_epoch_LPF(
                model=model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                epoch=epoch, 
                logger=logger, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), use_dls=use_dls, args=args, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss,
            )
        elif 'smoothout' in args.setting:
            train_stats, grads, losses = train_one_epoch_smoothout(
                model=model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                epoch=epoch, 
                logger=logger, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), use_dls=use_dls, args=args, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss,
            )
        elif 'Inner_loss' in args.setting or "Loss_Act" in args.setting:
            train_stats, grads, losses, da_dl = train_one_epoch_inner(
                model=model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                epoch=epoch, 
                logger=logger, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), use_dls=use_dls, args=args, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss,
            )
        elif 'simple' in args.setting:
            train_stats, grads, losses, da_dl = train_one_epoch_simple(
                model=model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                epoch=epoch, 
                logger=logger, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), use_dls=use_dls, args=args, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss,
            )

            experiment.log_metric('dTdL'+args.dataset, da_dl, step=epoch)            
        
        else:
            train_stats, grads, losses, da_dl = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                epoch=epoch, 
                logger=logger, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), use_dls=use_dls, args=args, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss,
            )

            experiment.log_metric('dTdL'+args.dataset, da_dl, step=epoch)

        if args.swa and ( (epoch + 1) >= int(swa_rat*(args.epochs)) ):
                # Batchnorm update
                print('Using SWA')
                optimizer.swap_swa_sgd()
                optimizer.bn_update(train_loader, model, device='cuda')
                # swa_res = utils.eval(loaders['test'], model, criterion)
                lr_scheduler.step(epoch)
                test_stats = evaluate(model, test_loader)
                val_stats = evaluate_val(model, val_loader)
                optimizer.swap_swa_sgd()
        # print('grad.shape,loss.shape:', grad.shape,loss.shape)
        else:
            lr_scheduler.step(epoch)
            test_stats = evaluate(model, test_loader)
            # if not int(args.val_rat) == 0:
            val_stats = evaluate_val(model, val_loader)
            # else:
            #     val_stats = test_stats
        if max_acc_test < test_stats["test_acc1"]:
            max_acc_test = test_stats["test_acc1"]
            checkpoint_dir = "/media3/clm/Deforming/"+save_dir
            if is_main_process:
                if not os.path.exists(checkpoint_dir):
                    # os.makedirs(os.path.join(args.output_dir, args.output_name))
                    os.makedirs(checkpoint_dir)                    
                # torch.save({
                #     'model': model_without_ddp.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'lr_scheduler': lr_scheduler.state_dict(),
                #     'epoch': epoch,
                #     'args': args,
                # }, os.path.join(checkpoint_dir, str(epoch)+'checkpoint.pth')) # os.path.join(args.output_dir, args.output_name, 'checkpoint.pth'))
        if max_acc_val < val_stats["val_acc1"]:
            max_acc_val = val_stats["val_acc1"]
        for item in train_stats.items():
            name, value = item[0], item[1]
            # print('train_stats: name, value',name, value)
            if not 'acc5'  in name:
                experiment.log_metric(name+args.dataset, value, step=epoch)
            
        for item in test_stats.items():
            name, value = item[0], item[1]
            # print('test_stats: name, value',name, value)
            if not 'acc5'  in name:
                experiment.log_metric(name+args.dataset, value, step=epoch)
        
        for item in val_stats.items():
            name, value = item[0], item[1]
            # print('val_stats: name, value',name, value)
            if not 'acc5'  in name:
                experiment.log_metric(name+args.dataset, value, step=epoch)
        # tr_te_loss_gap = test_stats['test_loss'] - train_stats['train_loss']
        # tr_te_acc_gap = train_stats['train_acc1'] - test_stats['test_acc1']
        # tr_va_loss_gap = val_stats['val_loss'] - train_stats['train_loss']
        # tr_va_acc_gap = train_stats['train_acc1'] - val_stats['val_acc1']
        # tr_te_loss_gapS.append(tr_te_loss_gap)  
        # tr_te_acc_gapS.append(tr_te_acc_gap)
        # tr_va_loss_gapS.append(tr_va_loss_gap)  
        # tr_va_acc_gapS.append(tr_va_acc_gap)
        # tr_losses.append(train_stats['train_loss'])
        # te_losses.append(test_stats['test_loss'])
        # va_losses.append(val_stats['val_loss'])
        
        # experiment.log_metric('tr_te_loss_gap'+args.dataset, tr_te_loss_gap, step=epoch)
        # experiment.log_metric('tr_te_acc_gap'+args.dataset, tr_te_acc_gap, step=epoch)
        # experiment.log_metric('tr_va_loss_gap'+args.dataset, tr_va_loss_gap, step=epoch)
        # experiment.log_metric('tr_va_acc_gap'+args.dataset, tr_va_acc_gap, step=epoch)
        
        acceptable_gap = 0
        # if 'CIFAR10_' in args.dataset:
        #     acceptable_gap = 2
        # if 'CIFAR100_' in args.dataset:
        #     acceptable_gap = 14
        
        # error = tr_va_acc_gap-acceptable_gap
        # error = tr_va_loss_gap-acceptable_gap
        # d_error = (error - prev_error) #/ 0.01
        # prev_error = error
        # int_error += error #* 0.01
    
        # control_signal = kp * error + ki * int_error + kd * d_error
        # print('control_signal,error,int_error,d_error',control_signal,error,int_error,d_error)
        # # control_signal = max(control_signal,0.0)
        # experiment.log_metric('error'+args.dataset, error, step=epoch)
        # experiment.log_metric('int_error'+args.dataset, int_error, step=epoch)
        # experiment.log_metric('d_error'+args.dataset, d_error, step=epoch)
        # experiment.log_metric('control_signal'+args.dataset, control_signal, step=epoch)

        steps = list(range(0, len(tr_losses)))
        
        # corr_losses_tr_te_loss = torch_coorcoef(tr_losses, tr_te_loss_gapS)
        # corr_losses_tr_va_loss = torch_coorcoef(tr_losses, tr_va_loss_gapS)
        # corr_steps_tr_te_loss = torch_coorcoef(steps, tr_te_loss_gapS)
        # corr_steps_tr_va_loss = torch_coorcoef(steps, tr_va_loss_gapS)
                        
        # corr_losses_tr_te_acc = torch_coorcoef(tr_losses, tr_te_acc_gapS)
        # corr_losses_tr_va_acc = torch_coorcoef(tr_losses, tr_va_acc_gapS)
        # corr_steps_tr_te_acc = torch_coorcoef(steps, tr_te_acc_gapS)
        # corr_steps_tr_va_acc = torch_coorcoef(steps, tr_va_acc_gapS)  
            
        # experiment.log_metric('corr_losses_tr_te_loss'+args.dataset, corr_losses_tr_te_loss, step=epoch)
        # experiment.log_metric('corr_losses_tr_va_loss'+args.dataset, corr_losses_tr_va_loss, step=epoch)
        # experiment.log_metric('corr_steps_tr_te_loss'+args.dataset, corr_steps_tr_te_loss, step=epoch)
        # experiment.log_metric('corr_steps_tr_va_loss'+args.dataset, corr_steps_tr_va_loss, step=epoch)
        
        # experiment.log_metric('corr_losses_tr_te_acc'+args.dataset, corr_losses_tr_te_acc, step=epoch)
        # experiment.log_metric('corr_losses_tr_va_acc'+args.dataset, corr_losses_tr_va_acc, step=epoch)
        # experiment.log_metric('corr_steps_tr_te_acc'+args.dataset, corr_steps_tr_te_acc, step=epoch)
        # experiment.log_metric('corr_steps_tr_va_acc'+args.dataset, corr_steps_tr_va_acc, step=epoch)        
                    
        # logger.wandb_log(epoch=epoch, **train_stats)
        # logger.wandb_log(epoch=epoch, **test_stats)
        if  epoch == 1 or epoch % 200 == 200 - 1 or epoch >= args.epochs - 2:
            msg = ' '.join([
                'Epoch:{epoch}',
                'Train Loss:{train_loss:.4f}',
                'Train Acc1:{train_acc1:.4f}',
                'Train Acc5:{train_acc5:.4f}',
                'Test Loss:{test_loss:.4f}',
                'Test Acc1:{test_acc1:.4f}(Max:{max_acc_test:.4f})',
                'Test Acc5:{test_acc5:.4f}',
                'Val Loss:{val_loss:.4f}',
                'Val Acc1:{val_acc1:.4f}(Max:{max_acc_test:.4f})',
                'Val Acc5:{val_acc5:.4f}',            
                'Time:{epoch_time:.3f}s'])
            logger.log(msg.format(epoch=epoch, **train_stats, **test_stats, **val_stats, max_acc_test=max_acc_test, epoch_time=time.time()-start_epoch))
        if epoch >=10 and max_acc_test <=12:
            print('pass')
            experiment.end()
        # if plot_surf and (epoch == 1 or epoch % 100 == 100 - 1 or epoch >= args.epochs - 2):
        if args.plot_surf and ( epoch >= args.epochs - 1):
            # plot_2d = False
            # proj_file = None
            # show = True
            modelTMP = copy.deepcopy(model)
            
            try:
                args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
                args.ymin, args.ymax, args.ynum = (None, None, None)
                if args.y:
                    args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                    args.ynum =  int(args.ynum)

                    assert args.ymin and args.ymax and args.ynum, \
                    'You specified some arguments for the y axis, but not all'
                    
                args.xnum = int(args.xnum)

            except:
                raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
                
            w = net_plotter.get_weights(modelTMP) # initial parameters
            s = copy.deepcopy(modelTMP.state_dict())
            dir_file = save_dir+'/'+str(epoch)+'_'+str(args.dls_start)+'_'+str(args.dls_end)+'_'+str(args.dls_coe0)+'_'+str(args.dls_coe1)+net_plotter.name_direction_file(args) # name the direction file
           #if rank == 0:
            net_plotter.setup_direction(args, dir_file, modelTMP)

            surf_file = name_surface_file(args, dir_file)
            #if rank == 0:
            setup_surface_file(args, surf_file, dir_file)

            # wait until master has setup the direction file and surface file
            # mpi.barrier(comm)
            # load directions
            d = net_plotter.load_directions(dir_file)
            # calculate the consine similarity of the two directions
            if len(d) == 2 and rank == 0:
                similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
                print('cosine similarity between x-axis and y-axis: %f' % similarity)
            # mpi.barrier(comm)
            crunch(surf_file, modelTMP, w, s, d, train_loader, 'train_loss', 'train_acc', comm, rank, args)
            crunch(surf_file, modelTMP, w, s, d, test_loader, 'test_loss', 'test_acc', comm, rank, args)

            #--------------------------------------------------------------------------
            # Plot figures
            #--------------------------------------------------------------------------
            if 1:
                if args.y and args.proj_file:
                    plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
                elif args.y:
                    # vmin = 0.1
                    # vmax = 10
                    # vlevel = 0.5
                    plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
                else:
                    # xmin, xmax, x_num = -1,1,10
                    # loss_max = 5
                    # log = False
                    plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.logscale, args.show)
                    plot_1D.plot_1d_loss_appro(surf_file, args.xmin, args.xmax, args.loss_max, args.logscale, args.show)

            
    # if args.swa:
    #     optimizer.bn_update(train_loader, model)        
    #     optimizer.swap_swa_sgd()   
    if args.plot_grad: 
        grads = torch.stack(grads)
        # plot_grad_norm(grads, num_dims, losses, args, save_dir)
        plot_grad_norm_color(grads, num_dims, losses, args, save_dir)

    logger.log('Train Finish. Max Test Acc1:{:.4f}'.format(max_acc_test))
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training-start_training))
    logger.log('Training Time:{}'.format(used_training))
    # logger.mv('{}_{:.4f}'.format(logger.logger_path, max_acc_test))
    inputs = torch.rand(1,3,32,32).cuda()
    flops, params = profile(model, inputs=(inputs, ))
    params = int(params)
    print('flops, params',flops, params)
    experiment.log_metric('flops', flops)
    experiment.log_metric('params', params)
    if not args.tuning:
        experiment.end()
    return max_acc_val, max_acc_test

# print('sys.argv[1]',sys.argv[1])

if __name__ == '__main__':
# if 1:
    from configs.defaulf_cfg import default_parser
    cfg_file = default_parser()
    args = cfg_file.get_args()
    if 'singlerun' in args.setting:
        main(args=args)
    else:
    # if 1:
    # with btn.bottleneck_kit():
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:

        # if args.dls_act == 'logexp':
        #     config = {
        #         "algorithm": "bayes",
        #         #random
        #         # Declare your hyperparameters in the Vizier-inspired format:
        #         "parameters": {
        #             "dls_coe0": {"type": "float", "min": 0.0, "max": 1.},
        #             # "dls_coe1": {"type": "float", "min": 0.0, "max": 8.},
        #             #"dataset": {"type": "categorical", "values": [
        #             #"cifar100",
        #             #]},
        #         },
        #         #"float", "min": 0, "max": 2
        #         "spec": {
        #         "maxCombo": 1000,
        #         "metric": "val_acc_SumDatasets",
        #             "objective": "maximize",
        #         },
        #         'trials': 2,
        #         #"experiment_class": "OfflineExperiment",
        #     }
        # elif args.dls_act == 'atan':
        #     config = {
        #         "algorithm": "bayes",                
        #         #random
        #         # Declare your hyperparameters in the Vizier-inspired format:
        #         "parameters": {
        #             "dls_coe0": {"type": "float", "min": 0.0, "max": 10.},
        #             "dls_coe1": {"type": "float", "min": 0.0, "max": 8.},
        #             #"dataset": {"type": "categorical", "values": [
        #             #"cifar100",
        #             #]},
        #         },
        #         #"float", "min": 0, "max": 2
        #         "spec": {
        #         "maxCombo": 1000,
        #         "metric": "val_acc_SumDatasets",
        #             "objective": "maximize",
        #         },
        #         'trials': 2,
        #         #"experiment_class": "OfflineExperiment",
        #     }
        # elif args.dls_act == 'linear':
        #     config = {
        #         "algorithm": "grid",                
        #         #random
        #         # Declare your hyperparameters in the Vizier-inspired format:
        #         "parameters": {
        #             # "dls_start": {"type": "float", "min": 0., "max": 0.5},
        #             # "dls_end": {"type": "float", "min": 0., "max": 0.5},
        #             # "dls_start": {"type": "discrete", "values": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
        #             # "dls_end": {"type": "discrete", "values": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
        #             # "dls_start": {"type": "discrete", "values": [0.9,0.8,0.7]},#,0.4,0.3,0.2,0.1#
        #             # "dls_end": {"type": "discrete", "values": [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
        #             # "dls_start": {"type": "discrete", "values": [0.6,0.5,0.4,0.3,0.2,0.1]},#0.9,0.8,0.7,0.6,0.5]},#,0.4,0.3,0.2,0.1#
        #             # "dls_end": {"type": "discrete", "values": [0.5,0.4,0.3,0.2,0.1]},#0.8,0.7,0.6,0.5,0.4,
        #             "dls_start": {"type": "discrete", "values": [0.3]},#0.9,0.8,0.7,0.6,0.5]},#,0.4,0.3,0.2,0.1#
        #             "dls_end": {"type": "discrete", "values": [0.2]},#0.8,0.7,0.6,0.5,0.4,                    
        #             "dls_coe0": {"type": "discrete", "values": [5,2.]},

        #             # "dls_coe0": {"type": "discrete", "values": [2., 5.]},
        #             # "dls_coe0": {"type": "categorical", "values": [2,5,10]},
        #         },
        #         #"float", "min": 0, "max": 2
        #         "spec": {
        #         # "gridSize": 10,
        #         "maxCombo": 10000,
        #         "metric": "val_acc_SumDatasets",
        #             "objective": "maximize",
        #         },
        #         'trials': 1,
        #         #"experiment_class": "OfflineExperiment",
        #     }
        # opt = Optimizer(config)
        
        # opt = Optimizer(sys.argv[1], experiment_class=OfflineExperiment)
        # print('sys.argv[1]',sys.argv[1])
        opt = Optimizer(sys.argv[1])
        # from configs.defaulf_cfg import default_parser

        #opt.init(experiment_class=OfflineExperiment)
        opt.get_id()
        comet_dir = "./logs/CometData_tuning23"
        # TODO: upload "./logs/CometData_tuning5"
        # args.dls_act = 'linear'
        if not os.path.exists(comet_dir):
            os.makedirs(comet_dir)
        for experiment in opt.get_experiments(
            project_name="deforming-cutout-tuning-plot",
            # offline_directory=comet_dir,
            log_code=False,
            log_graph=False, 
            auto_param_logging=False, 
            auto_metric_logging=False,
            auto_output_logging=False, 
            log_env_details=False,
            log_git_metadata=False, 
            log_git_patch=False, 
            log_env_gpu=False, 
            log_env_host=False, 
            log_env_cpu=False,
            ):
            cfg_file = default_parser()
            args = cfg_file.get_args()
            
            args.val_rat=0.0
            # args.momentum=0.0
          
            # args.swa = None  
            # args.setting = 'Default'#'LossSche'Default#LossRepLR
            # args.lr_scheduler = 'MultiStepLRscheduler'

            args.val_type = 'test'
            # args.epochs = 200
            # args.dls_act = 'sech'#'linear'#'sech'
            args.tuning = True
            args.plot_surf = True #TrueTrue#
            # args.plot_grad = True#False#True
            # args.lr = 0.2
            # args.dls_coe0 = 1.0 
            # args.dls_coe1 = 4.0
            args.calm_base = experiment.get_parameter("calm_base")
            args.epochs = experiment.get_parameter("epochs")
            # args.lr_scheduler = experiment.get_parameter("lr_scheduler")
            
            args.end = experiment.get_parameter("end")
            args.rep_num = experiment.get_parameter("rep_num")
            args.lr = experiment.get_parameter("lr")
            args.plus_base = experiment.get_parameter("plus_base")

            # args.min_shift = experiment.get_parameter("min_shift")


            args.dls_act = experiment.get_parameter("dls_act")
            args.seed = experiment.get_parameter("seed")
            args.model=experiment.get_parameter("model")
            args.setting=experiment.get_parameter("setting")
            args.swa = experiment.get_parameter("swa")
            args.opt = experiment.get_parameter("opt")
            args.M = experiment.get_parameter("M")
            args.dls_coe0=experiment.get_parameter("dls_coe0")
            args.dls_coe1=experiment.get_parameter("dls_coe1")
            args.dls_coe2=experiment.get_parameter("dls_coe2")
            if 'no_loss_norm' in args.setting:
                args.norm_loss = None
            else:
                args.norm_loss = True
            if args.swa == 'None':
                args.swa = None
            if 'sam' in args.opt:
                args.weight_decay = 1e-3
                args.sparsity = 0.5
                args.num_samples = 16
                args.update_freq = 1
                if args.swa:
                    print('pass')
                    experiment.log_metric("val_acc_SumDatasets", None)
                    experiment.end()
                    continue
                
            if args.dls_act == 'atan' or args.dls_act == 'logexp':
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                if args.dls_act == 'atan':
                    args.dls_coe1=experiment.get_parameter("dls_coe1")
                print('args.dls_coe0,args.dls_coe1',args.dls_coe0,args.dls_coe1)
                
            elif args.dls_act == 'linear':
                args.dls_start=experiment.get_parameter("dls_start")
                if args.dls_start>=0.1:
                    args.dls_end=args.dls_start-0.1#experiment.get_parameter("dls_end")
                else:
                    args.dls_end=args.dls_start-0.01
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                print('args.dls_start, args.dls_end, args.dls_coe0',args.dls_start, args.dls_end,args.dls_coe0)
                if args.dls_start <= args.dls_end:
                    print('pass')
                    experiment.log_metric("val_acc_SumDatasets", None)
                    experiment.end()
                    continue
            elif args.dls_act == 'linear_whole':
                args.dls_start=999999
                args.dls_end = -9999.0#0.0
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                args.setting=experiment.get_parameter("setting")
                if args.setting == "scaling LR":
                    # base_lr = args.lr
                    args.lr = args.dls_coe0*args.lr
                    args.dls_coe0 = 1.0
                if  args.setting == "Scaling Inversely":
                    args.lr = args.lr/args.dls_coe0
                print('args.setting, args.dls_start, args.dls_end, args.dls_coe0, args.lr',args.setting, args.dls_start, args.dls_end,args.dls_coe0, args.lr)

                    
            elif args.dls_act == 'sech':
                # args.dls_list=experiment.get_parameter("dls_list")
                # print('dls_list',args.dls_list, type(args.dls_list))
                # dls_values = list(map(float, re.findall(r'\d+\.\d+', args.dls_list)))
                # print('dls_values',dls_values)
                # args.dls_coe0 = dls_values[0]
                # args.dls_coe1 = dls_values[1]
                # args.lr = 1.0
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                args.dls_coe1=experiment.get_parameter("dls_coe1")
            # args.data=experiment.get_parameter("data")
            if args.setting == 'baseline':
                args.dls_coe0 = 1.0
                args.dls_coe1 = 0.0
                args.dls_act = 'linear_whole'
                args.dls_start = -9999.0
                args.dls_end = -99999.0
        # if 1:
        
            # Datasets = ['CIFAR10_base']#,'CIFAR100_base']#'SVHN_base',
            Datasets = ['CIFAR100_cutout','CIFAR10_cutout']
            # Datasets = ['CIFAR100_base']#['CIFAR10_cutout']#'SVHN_base',        
      
            val_acc_c10, val_acc_c100,val_acc_svhn=0,0,0
            for dataset in Datasets:
                args.dataset = dataset
                if 'CIFAR100_' in dataset:
                    args.rho = 0.1
                elif 'CIFAR10_' in dataset:
                    args.rho = 0.2
                max_acc_val, max_acc_test = main(args,experiment)
                
                # print(prof.key_averages().table(sort_by="self_cuda_time_total"))

                if 'CIFAR10_' in dataset:
                    if args.val_rat==0.0:
                        val_acc_c10 = max_acc_test
                    else:
                        val_acc_c10 = max_acc_val
                    experiment.log_metric("val_acc_c10_best", val_acc_c10)
                elif 'CIFAR100_' in dataset:
                    if args.val_rat==0.0:
                        val_acc_c100 = max_acc_test
                    else:
                        val_acc_c100 = max_acc_val                    
                    experiment.log_metric("val_acc_c100_best", val_acc_c100)                
                elif 'SVHN_' in dataset:
                    if args.val_rat==0.0:
                        val_acc_svhn = max_acc_test
                    else:
                        val_acc_svhn = max_acc_val                    
                    experiment.log_metric("val_acc_svhn_best", val_acc_svhn)                
                    
            val_acc_SumDatasets = val_acc_c10 +val_acc_c100+val_acc_svhn
            experiment.log_metric("val_acc_SumDatasets", val_acc_SumDatasets)
            experiment.end()
        # 
            