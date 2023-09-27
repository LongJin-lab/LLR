from comet_ml import Experiment, OfflineExperiment, Optimizer

import os
import time
import datetime

import torch

from models.build import build_model
from data.build import build_dataset, build_train_dataloader, build_val_dataloader, build_test_dataloader
from solver.build import build_optimizer, build_lr_scheduler

from utils.logger import Logger
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.engine import train_one_epoch, train_one_epoch_F, evaluate, evaluate_val
import numpy as np
# from turtle import color
import torch
import matplotlib.pyplot as plt
import matplotlib
# from torchcontrib.optim import SWA
import torchcontrib
import numpy as np
from thop import profile
# import sys

FAILURE_MODE = True

   

def plot_grad(grads, num_dims, losses, args):
    
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
     
    steps = list(range(0, grads.shape[0]))
    # print('losses,steps,grads',losses,steps,grads)
    corr_losses = np.corrcoef(losses, grads.T)
    corr_steps = np.corrcoef(steps, grads.T)
    n_dim = grads.shape[1]
    dim = np.arange(0,n_dim)
    # print('dim.shape',dim.shape)
    # print('corr.shape',corr_losses[0,1:].shape)

    # fig, ax = plt.subplots()

    fig, ax = plt.subplots()
    ax.scatter(dim,corr_losses[0,1:],alpha=0.2, label="Gradient-loss correlation", color=(0.15, 0.25, 0.4))
    ax.scatter(dim,np.abs(corr_steps[0,1:]),alpha=0.5, label="Gradient-iteration correlation", marker='^', s=7, color=(0.5,0.1, 0.1))
    namebase = args.output_dir+args.output_name+'/'

    torch.save(torch.tensor(dim), namebase+'dim.pt')
    torch.save(torch.tensor(corr_losses), namebase+'corr_losses.pt')
    torch.save(torch.tensor(corr_steps), namebase+'corr_steps.pt')
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Pearson correlation coefficient')
    ax.legend()
    fname = namebase+'dim'+str(num_dims)+'correlation.pdf'
    print('fname',fname)
    plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    # fig, ax = plt.subplots()
    # ax.set_xlabel('Loss')
    # ax.set_ylabel('Absolute value of gradient')
    # for i in range(grads.size(1)):
    #     # ax.plot(losses, grads[:, i], alpha=0.05, linewidth=1, linestyle='-',color=(0.15, 0.25, 0.4))
    #     ax.scatter(losses,grads[:, i],alpha=0.05, s=0.2,color=(0.1, 0.25, 0.4))
    # torch.save(torch.tensor(losses), namebase+'losses.pt')
    # torch.save(torch.tensor(grads), namebase+'grads.pt')
    # # plt.show()
    # fname = namebase+'dim'+str(num_dims)+'gradient_Loss.pdf'
    # plt.savefig(fname, format='pdf',bbox_inches = 'tight')

    # fig, ax = plt.subplots()
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Absolute value of gradient')
    # for i in range(grads.size(1)):
    #     # ax.plot(grads[:, i], alpha=0.05, linewidth=1, linestyle='-', color=(0.5,0.1, 0.1))  
    #     ax.scatter(steps, grads[:, i],alpha=0.05, marker='^',s=0.2,color=(0.5,0.1, 0.1))   
    # # plt.show()
    # fname = namebase+'dim'+str(num_dims)+'gradient_Iteration.pdf'
    # plt.savefig(fname, format='pdf',bbox_inches = 'tight')

def torch_coorcoef(A,B):
    A_tensor = torch.tensor(A).cuda()
    B_tensor = torch.tensor(B).cuda()
    combined_tensor = torch.stack((A_tensor, B_tensor), dim=0)
    corr_matrix = torch.corrcoef(combined_tensor)
    return corr_matrix[0, 1:]

def main(args,experiment=None):
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
    
    if args.output_name is None:
        args.output_name = '_'.join([
            args.dataset,
            'bsz' + str(args.batch_size),
            'epoch' + str(args.epochs),
            args.model,
            'lr' + str(args.lr),
            str(args.opt),
        ] + sam_hyper_param(args) + ['seed{}'.format(args.seed)])    
        
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    args.output_dir = args.output_dir+cur_time
    if not os.path.exists("./logs/CometData"):
        os.makedirs("./logs/CometData")
    if experiment is None:
        experiment = Experiment(
        # experiment = OfflineExperiment(
            api_key="KbJPNIfbsNUoZJGJBSX4BofNZ",
            project_name="Deforming-tuning-cutout-where_par-tmp",#"deforming-cifar-sam",#
            workspace="logichen",
            # auto_histogram_weight_logging=True,
            # offline_directory="./logs/CometData",
        )
    hyper_params = vars(args)
    experiment.log_parameters(hyper_params)
    # init seed
    setup_seed(args)

    # init dist
    init_distributed_model(args)

    # init log
    logger = Logger(args)
    logger.log(args)

    # build dataset and dataloader
    train_data, val_data, test_data, n_classes = build_dataset(args)
    print('len(val_data):',len(val_data))
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
    logger.log(f'Train Data: {len(train_data)}, Test Data: {len(test_data)}.')

    # build model
    model = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    logger.log(f'Model: {args.model}')

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # build solver
    opt, base_optimizer = build_optimizer(args, model=model_without_ddp)
    swa_rat = 0.75
    if args.swa:
        print('Using SWA')
        if 'CIFAR10_' in args.dataset:
            args.swa_lr = 0.01
        elif 'CIFAR100_' in args.dataset:
            args.swa_lr = 0.05
        elif 'SVHN_' in args.dataset:
            args.swa_lr = 0.01
        print('int(swa_rat*(args.epochs)',int(swa_rat*(args.epochs)))
        # optimizer = torchcontrib.optim.SWA(opt, swa_start=int(swa_rat*(args.epochs)), swa_freq=1, swa_lr=args.swa_lr)
        optimizer = torchcontrib.optim.SWA(opt,swa_freq=1,swa_lr=args.swa_lr)

    else:
        optimizer = opt
        print('Using SGD')
        
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
    logger.log(f'Start training for {args.epochs} Epochs.')
    start_training = time.time()
    max_acc = 0.0
    max_acc_val = 0.0
    grads = []
    losses = []

    plotgrads = False 
    if plotgrads: 
        inputs = torch.rand(1,3,32,32).cuda()
        flops, params = profile(model, inputs=(inputs, ))
        params = int(params)
        print('flops, params',flops, params)
        num_dims = 500        
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
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        experiment.log_metric('cur_lr', cur_lr, step=epoch)
        # print('dls_coe',dls_coe.grad)
        if train_stats and args.dls_act:
            if args.pid:
                dls_coe = [control_signal]+[args.dls_coe1]
            else:
                dls_coe = [args.dls_coe0]+[args.dls_coe1]            
            if args.norm_loss:
                if epoch == 1:
                    ini_loss = train_stats['train_loss']
            else:
                ini_loss = None
            if args.dls_start:
                if train_stats['train_loss']/ini_loss<=args.dls_start and train_stats['train_loss']/ini_loss>=args.dls_end:
                    print('Using DLS, train_loss/ini_loss', train_stats['train_loss']/ini_loss)
                    dls_coe = [args.dls_coe0]+[args.dls_coe1]
                else:
                    dls_coe = None
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

        if (epoch + 1) >= int(swa_rat*(args.epochs)) and not args.dls_start:
            dls_coe = None

        train_stats, grads, losses = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epoch=epoch, 
            logger=logger, log_freq=args.log_freq, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), dls_coe=dls_coe,dls_act=args.dls_act, grads=grads,losses=losses, tracked_dims=tracked_dims, shift=shift, ini_loss=ini_loss
        )

                
        if args.swa and ( (epoch + 1) >= int(swa_rat*(args.epochs)) or epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1):
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
            val_stats = evaluate_val(model, val_loader)
        
        if max_acc < test_stats["test_acc1"]:
            max_acc = test_stats["test_acc1"]
            if is_main_process:
                if not os.path.exists(args.output_dir+args.output_name):
                    # os.makedirs(os.path.join(args.output_dir, args.output_name))
                    os.makedirs(args.output_dir+args.output_name)                    
                # torch.save({
                #     'model': model_without_ddp.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'lr_scheduler': lr_scheduler.state_dict(),
                #     'epoch': epoch,
                #     'args': args,
                # }, os.path.join(args.output_dir+args.output_name, 'checkpoint.pth')) # os.path.join(args.output_dir, args.output_name, 'checkpoint.pth'))
        if max_acc_val < val_stats["val_acc1"]:
            max_acc_val = val_stats["val_acc1"]
        for item in train_stats.items():
            name, value = item[0], item[1]
            print('train_stats: name, value',name, value)
            experiment.log_metric(name+args.dataset, value, step=epoch)
            
        for item in test_stats.items():
            name, value = item[0], item[1]
            print('test_stats: name, value',name, value)
            experiment.log_metric(name+args.dataset, value, step=epoch)
        
        for item in val_stats.items():
            name, value = item[0], item[1]
            print('val_stats: name, value',name, value)
            experiment.log_metric(name+args.dataset, value, step=epoch)
        tr_te_loss_gap = test_stats['test_loss'] - train_stats['train_loss']
        tr_te_acc_gap = train_stats['train_acc1'] - test_stats['test_acc1']
        tr_va_loss_gap = val_stats['val_loss'] - train_stats['train_loss']
        tr_va_acc_gap = train_stats['train_acc1'] - val_stats['val_acc1']
        tr_te_loss_gapS.append(tr_te_loss_gap)  
        tr_te_acc_gapS.append(tr_te_acc_gap)
        tr_va_loss_gapS.append(tr_va_loss_gap)  
        tr_va_acc_gapS.append(tr_va_acc_gap)
        tr_losses.append(train_stats['train_loss'])
        te_losses.append(test_stats['test_loss'])
        va_losses.append(val_stats['val_loss'])
        
        experiment.log_metric('tr_te_loss_gap'+args.dataset, tr_te_loss_gap, step=epoch)
        experiment.log_metric('tr_te_acc_gap'+args.dataset, tr_te_acc_gap, step=epoch)
        experiment.log_metric('tr_va_loss_gap'+args.dataset, tr_va_loss_gap, step=epoch)
        experiment.log_metric('tr_va_acc_gap'+args.dataset, tr_va_acc_gap, step=epoch)
        
        acceptable_gap = 0
        # if 'CIFAR10_' in args.dataset:
        #     acceptable_gap = 2
        # if 'CIFAR100_' in args.dataset:
        #     acceptable_gap = 14
        
        # error = tr_va_acc_gap-acceptable_gap
        error = tr_va_loss_gap-acceptable_gap
        d_error = (error - prev_error) #/ 0.01
        prev_error = error
        int_error += error #* 0.01
    
        # Compute the control signal
        control_signal = kp * error + ki * int_error + kd * d_error
        print('control_signal,error,int_error,d_error',control_signal,error,int_error,d_error)
        # control_signal = max(control_signal,0.0)
        experiment.log_metric('error'+args.dataset, error, step=epoch)
        experiment.log_metric('int_error'+args.dataset, int_error, step=epoch)
        experiment.log_metric('d_error'+args.dataset, d_error, step=epoch)
        experiment.log_metric('control_signal'+args.dataset, control_signal, step=epoch)

        steps = list(range(0, len(tr_losses)))
        
        corr_losses_tr_te_loss = torch_coorcoef(tr_losses, tr_te_loss_gapS)
        corr_losses_tr_va_loss = torch_coorcoef(tr_losses, tr_va_loss_gapS)
        corr_steps_tr_te_loss = torch_coorcoef(steps, tr_te_loss_gapS)
        corr_steps_tr_va_loss = torch_coorcoef(steps, tr_va_loss_gapS)
                
        
        print('corr_losses_tr_te_loss, corr_losses_tr_va_loss, corr_steps_tr_te_loss, corr_steps_tr_va_loss', corr_losses_tr_te_loss, corr_losses_tr_va_loss, corr_steps_tr_te_loss, corr_steps_tr_va_loss)
        
        corr_losses_tr_te_acc = torch_coorcoef(tr_losses, tr_te_acc_gapS)
        corr_losses_tr_va_acc = torch_coorcoef(tr_losses, tr_va_acc_gapS)
        corr_steps_tr_te_acc = torch_coorcoef(steps, tr_te_acc_gapS)
        corr_steps_tr_va_acc = torch_coorcoef(steps, tr_va_acc_gapS)  
        print('corr_losses_tr_te_acc, corr_losses_tr_va_acc, corr_steps_tr_te_acc, corr_steps_tr_va_acc', corr_losses_tr_te_acc, corr_losses_tr_va_acc, corr_steps_tr_te_acc, corr_steps_tr_va_acc)
            
        experiment.log_metric('corr_losses_tr_te_loss'+args.dataset, corr_losses_tr_te_loss, step=epoch)
        experiment.log_metric('corr_losses_tr_va_loss'+args.dataset, corr_losses_tr_va_loss, step=epoch)
        experiment.log_metric('corr_steps_tr_te_loss'+args.dataset, corr_steps_tr_te_loss, step=epoch)
        experiment.log_metric('corr_steps_tr_va_loss'+args.dataset, corr_steps_tr_va_loss, step=epoch)
        
        experiment.log_metric('corr_losses_tr_te_acc'+args.dataset, corr_losses_tr_te_acc, step=epoch)
        experiment.log_metric('corr_losses_tr_va_acc'+args.dataset, corr_losses_tr_va_acc, step=epoch)
        experiment.log_metric('corr_steps_tr_te_acc'+args.dataset, corr_steps_tr_te_acc, step=epoch)
        experiment.log_metric('corr_steps_tr_va_acc'+args.dataset, corr_steps_tr_va_acc, step=epoch)        
                    
        # logger.wandb_log(epoch=epoch, **train_stats)
        # logger.wandb_log(epoch=epoch, **test_stats)
        msg = ' '.join([
            'Epoch:{epoch}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Test Loss:{test_loss:.4f}',
            'Test Acc1:{test_acc1:.4f}(Max:{max_acc:.4f})',
            'Test Acc5:{test_acc5:.4f}',
            'Val Loss:{val_loss:.4f}',
            'Val Acc1:{val_acc1:.4f}(Max:{max_acc:.4f})',
            'Val Acc5:{val_acc5:.4f}',            
            'Time:{epoch_time:.3f}s'])
        logger.log(msg.format(epoch=epoch, **train_stats, **test_stats, **val_stats, max_acc=max_acc, epoch_time=time.time()-start_epoch))
        
    # if args.swa:
    #     optimizer.bn_update(train_loader, model)        
    #     optimizer.swap_swa_sgd()   
    if plotgrads: 
        grads = torch.stack(grads)
        plot_grad(grads, num_dims, losses, args)

        
    logger.log('Train Finish. Max Test Acc1:{:.4f}'.format(max_acc))
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training-start_training))
    logger.log('Training Time:{}'.format(used_training))
    # logger.mv('{}_{:.4f}'.format(logger.logger_path, max_acc))
    if not args.tuning:
        experiment.end()
    return max_acc_val

if __name__ == '__main__':
    from configs.defaulf_cfg import default_parser
    cfg_file = default_parser()
    args = cfg_file.get_args()
    if not args.tuning:
        main(args=args)
    else:
        if args.dls_act == 'logexp':
            config = {
                "algorithm": "bayes",
                #random
                # Declare your hyperparameters in the Vizier-inspired format:
                "parameters": {
                    "dls_coe0": {"type": "float", "min": 0.0, "max": 1.},
                    # "dls_coe1": {"type": "float", "min": 0.0, "max": 8.},
                    #"dataset": {"type": "categorical", "values": [
                    #"cifar100",
                    #]},
                },
                #"float", "min": 0, "max": 2
                "spec": {
                "maxCombo": 1000,
                "metric": "val_acc_SumDatasets",
                    "objective": "maximize",
                },
                'trials': 2,
                #"experiment_class": "OfflineExperiment",
            }
        elif args.dls_act == 'atan':
            config = {
                "algorithm": "bayes",                
                #random
                # Declare your hyperparameters in the Vizier-inspired format:
                "parameters": {
                    "dls_coe0": {"type": "float", "min": 0.0, "max": 10.},
                    "dls_coe1": {"type": "float", "min": 0.0, "max": 8.},
                    #"dataset": {"type": "categorical", "values": [
                    #"cifar100",
                    #]},
                },
                #"float", "min": 0, "max": 2
                "spec": {
                "maxCombo": 1000,
                "metric": "val_acc_SumDatasets",
                    "objective": "maximize",
                },
                'trials': 2,
                #"experiment_class": "OfflineExperiment",
            }
        elif args.dls_act == 'linear':
            config = {
                "algorithm": "grid",                
                #random
                # Declare your hyperparameters in the Vizier-inspired format:
                "parameters": {
                    # "dls_start": {"type": "float", "min": 0., "max": 0.5},
                    # "dls_end": {"type": "float", "min": 0., "max": 0.5},
                    # "dls_start": {"type": "discrete", "values": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
                    # "dls_end": {"type": "discrete", "values": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
                    # "dls_start": {"type": "discrete", "values": [0.9,0.8,0.7]},#,0.4,0.3,0.2,0.1#
                    # "dls_end": {"type": "discrete", "values": [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]},
                    # "dls_start": {"type": "discrete", "values": [0.6,0.5,0.4,0.3,0.2,0.1]},#0.9,0.8,0.7,0.6,0.5]},#,0.4,0.3,0.2,0.1#
                    # "dls_end": {"type": "discrete", "values": [0.5,0.4,0.3,0.2,0.1]},#0.8,0.7,0.6,0.5,0.4,
                    "dls_start": {"type": "discrete", "values": [0.3]},#0.9,0.8,0.7,0.6,0.5]},#,0.4,0.3,0.2,0.1#
                    "dls_end": {"type": "discrete", "values": [0.2]},#0.8,0.7,0.6,0.5,0.4,                    
                    "dls_coe0": {"type": "discrete", "values": [1.]},

                    # "dls_coe0": {"type": "discrete", "values": [2., 5.]},
                    # "dls_coe0": {"type": "categorical", "values": [2,5,10]},
                },
                #"float", "min": 0, "max": 2
                "spec": {
                # "gridSize": 10,
                "maxCombo": 10000,
                "metric": "val_acc_SumDatasets",
                    "objective": "maximize",
                },
                'trials': 3,
                #"experiment_class": "OfflineExperiment",
            }
        opt = Optimizer(config)
        
        for experiment in opt.get_experiments(
            project_name="Deforming-tuning-cutout-where5-"+args.dls_act):
            cfg_file = default_parser()
            args = cfg_file.get_args()
            if args.dls_act == 'atan' or args.dls_act == 'logexp':
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                if args.dls_act == 'atan':
                    args.dls_coe1=experiment.get_parameter("dls_coe1")
                print('args.dls_coe0,args.dls_coe1',args.dls_coe0,args.dls_coe1)
            elif args.dls_act == 'linear':
                args.dls_start=experiment.get_parameter("dls_start")
                args.dls_end=experiment.get_parameter("dls_end")
                args.dls_coe0=experiment.get_parameter("dls_coe0")
                print('args.dls_start, args.dls_end, args.dls_coe0',args.dls_start, args.dls_end,args.dls_coe0)
                if args.dls_start <= args.dls_end:
                    print('pass')
                    experiment.log_metric("val_acc_SumDatasets", None)
                    experiment.end()
                    continue
            # args.data=experiment.get_parameter("data")        
        # if 1:
        
            # Datasets = ['CIFAR100_base','CIFAR10_base']#'SVHN_base',
            Datasets = ['CIFAR100_cutout','CIFAR10_cutout']#'SVHN_base',        
            val_acc_c10, val_acc_c100,val_acc_svhn=0,0,0
            for dataset in Datasets:
                args.dataset = dataset
                max_acc_val = main(args,experiment)
                if 'CIFAR10_' in dataset:
                    val_acc_c10 = max_acc_val
                    experiment.log_metric("val_acc_c10_best", val_acc_c10)
                elif 'CIFAR100_' in dataset:
                    val_acc_c100 = max_acc_val
                    experiment.log_metric("val_acc_c100_best", val_acc_c100)                
                elif 'SVHN_' in dataset:
                    val_acc_svhn = max_acc_val
                    experiment.log_metric("val_acc_svhn_best", val_acc_svhn)                
                    
            val_acc_SumDatasets = val_acc_c10 +val_acc_c100+val_acc_svhn
            experiment.log_metric("val_acc_SumDatasets", val_acc_SumDatasets)
            experiment.end()