import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
from utils.dist import is_dist_avail_and_initialized
import numpy as np
# torch.autograd.set_detect_anomaly(True)
def LinQuad(loss, dls_coe0):
    return loss + dls_coe0 * loss**2
def train_one_epoch(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()
    c = math.log(1 + math.sqrt(2))
    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    
    if args.min_shift:
        min_ = 0
        min_bias = -args.dls_coe0*args.dls_coe1/(c*np.cosh(c/(args.dls_coe1+0.000001)*min_)+0.000001)
        if args.plus_base:
            min_bias=min_bias+args.plus_base*min_
        
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # print('batch_idx',batch_idx)
        def closure():
            output = model(images)
            loss_ori = criterion(output, targets)
            if 'LossSche' in args.setting and ini_loss:
                Sche = loss_ori.item()/ini_loss
            else:
                Sche = 1.0
            loss_sch = loss_ori*Sche    
            # use_dls = False        
            if use_dls:# and 
                if ini_loss:
                    loss_inp = loss_sch /ini_loss
                else:
                    loss_inp = loss_sch*1.0
                if args.dls_act == 'logexp':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    # dls_coe0 = float(np.tanh(dls_coe0))
                    loss_act = args.dls_coe1*torch.log(torch.exp(loss_inp)-dls_coe0)
                    da_dl = args.dls_coe1*torch.exp(loss_inp) / (-dls_coe0 + torch.exp(loss_inp))
                elif args.dls_act == 'atan':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                elif args.dls_act == 'sech':
                    # dls_coe0 = max(args.dls_coe0,0.0)
                    # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                    dls_coe1 = args.dls_coe1
                    dls_coe0 = args.dls_coe0
                    if "cyclic_dls" in args.setting:
                        interval = args.epochs/args.rep_num
                        dls_coe1 = args.end*((epoch+interval-1)%(interval))/(interval+0.00000001)+args.dls_coe1
                    loss_act_ = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                    da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1  
                                  
                    if args.plus_base:
                        _loss_act = loss_act_+args.plus_base*loss_inp
                        da_dl = da_dl+args.plus_base
                    else:
                        _loss_act = loss_act_*1.0
                    if args.min_shift:
                        loss_act = _loss_act-min_bias
                    else:
                        loss_act = _loss_act*1.0
                        
                elif 'linear' in args.dls_act:
                    if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                        # dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                        loss_act = args.dls_coe0*loss_inp
                        # print('using linear')
                    else:
                        loss_act = loss_inp  *1.0
                if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                    loss_fin = ini_loss*loss_act
                    # print('ini_loss,loss_ori,loss_sch,loss_inp,loss_act,loss_fin',ini_loss,float(loss_ori.item()),float(loss_sch.item()),float(loss_inp.item()),float(loss_act.item()),float(loss_fin.item()))

                else:
                    loss_fin = loss_act*1.0#deepcopy?
                # print('loss2',loss_fin)
            else:
                loss_fin = loss_sch  *args.calm_base  
            loss_fin.backward()

        output = model(images)
        loss_ori = criterion(output, targets)
        # loss_ori = F.cross_entropy(output, targets)

        # pre_loss = loss_.clone().detach()
        
        # if float(torch.abs(pre_loss - loss_).item()) < epsilon and dls_coe:
        if 'LossSche' in args.setting and ini_loss:
            Sche = loss_ori.item()/ini_loss
        else:
            Sche = 1.0
        loss_sch = loss_ori*Sche    
        # use_dls = False  
        da_dl = 1.0      
        if use_dls:# and 
            if ini_loss:
                loss_inp = loss_sch /ini_loss
            else:
                loss_inp = loss_sch*1.0
            if args.dls_act == 'logexp':
                dls_coe0 = max(args.dls_coe0,0.0)
                # dls_coe0 = float(np.tanh(dls_coe0))
                loss_act = args.dls_coe1*torch.log(torch.exp(args.dls_coe2*loss_inp)-dls_coe0)/args.dls_coe2
                da_dl = args.dls_coe1*torch.exp(args.dls_coe2*loss_inp) / (-dls_coe0 + torch.exp(args.dls_coe2*loss_inp))
            elif args.dls_act == 'atan':
                dls_coe0 = max(args.dls_coe0,0.0)
                loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
            elif 'LinQuad' in args.dls_act:
                loss_act = LinQuad(loss_inp, args.dls_coe0)
            elif args.dls_act == 'sech':
                # dls_coe0 = max(args.dls_coe0,0.0)
                # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                dls_coe1 = args.dls_coe1
                dls_coe0 = args.dls_coe0
                if "cyclic_dls" in args.setting:
                    interval = args.epochs/args.rep_num
                    dls_coe1 = args.end*((epoch+interval-1)%(interval))/(interval+0.00000001)+args.dls_coe1
                loss_act_ = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1  
                                
                if args.plus_base:
                    _loss_act = loss_act_+args.plus_base*loss_inp
                    da_dl = da_dl+args.plus_base
                else:
                    _loss_act = loss_act_*1.0
                if args.min_shift:
                    loss_act = _loss_act-min_bias
                else:
                    loss_act = _loss_act*1.0
            elif 'linear' in args.dls_act:
                if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                    # dls_coe0 = max(args.dls_coe0,0.0)
                    # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                    loss_act = args.dls_coe0*loss_inp
                    # print('using linear')
                else:
                    loss_act = loss_inp  *1.0
            else:
                loss_act = loss_inp  *1.0
            if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                loss_fin = ini_loss*loss_act
                # print('ini_loss,loss_ori,loss_sch,loss_inp,loss_act,loss_fin',ini_loss,float(loss_ori.item()),float(loss_sch.item()),float(loss_inp.item()),float(loss_act.item()),float(loss_fin.item()))

            else:
                loss_fin = loss_act*1.0#deepcopy?
            # print('loss2',loss_fin)
        else:
            da_dl = args.calm_base
            loss_fin = loss_sch  *args.calm_base  
        optimizer.zero_grad()
        loss_fin.backward()   
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
            # print('lr',optimizer.param_groups[0]['lr'])
        if use_closure: 
            # print('use_closure')
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr / (loss_ori.item()/ini_loss)
            # print('lr2',optimizer.param_groups[0]['lr'])
            
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss_ori.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        

        # if batch_idx % args.log_freq == 0:
        #     msg = ' '.join([
        #     'Epoch: {epoch}',
        #     '[{batch_id}/{batch_len}]',
        #     'lr:{lr:.6f}',
        #     'Train Loss:{train_loss:.4f}',
        #     'Train Acc1:{train_acc1:.4f}',
        #     'Train Acc5:{train_acc5:.4f}',
        #     'Time:{batch_time:.3f}s'])
        #     # if dls_coe:
        #     #     print('dls_coe0',dls_coe0)
        #     #     print('loss_,loss_fin',loss_,loss_fin)
        #     logger.log(
        #         msg.format(
        #             epoch=epoch, 
        #             batch_id=batch_idx, batch_len = len(train_loader),
        #             lr=optimizer.param_groups[0]["lr"],
        #             train_loss=_memory.meters["train_loss"].global_avg,
        #             train_acc1=_memory.meters["train_acc1"].global_avg,
        #             train_acc5=_memory.meters["train_acc5"].global_avg,
        #             batch_time=time.time() - batch_start,
        #         )
        #     )
        _memory.synchronize_between_processes()
        if tracked_dims is not None:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    # grad.append(torch.abs(p.grad).detach().view(-1))
                    grad.append(p.grad.detach().view(-1))

            grad = torch.cat(grad)
            tracked_grad = torch.norm(grad, p='fro')
            # tracked_grad = torch.max(torch.abs(grad))
            # for dim in tracked_dims:
            #     tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_ori.item())

    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }, grads, losses, da_dl


def train_one_epoch_inner(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()
    c = math.log(1 + math.sqrt(2))
    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    # use_dls = False
    if args.min_shift:
        min_ = 0
        min_bias = -args.dls_coe0*args.dls_coe1/(c*np.cosh(c/(args.dls_coe1+0.000001)*min_)+0.000001)
        if args.plus_base:
            min_bias=min_bias+args.plus_base*min_
        
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # print('batch_idx',batch_idx)
        def closure():
            output = model(images)
            loss_ori = criterion(output, targets,ini_loss, args, use_dls,epoch)
            if 'LossSche' in args.setting and ini_loss:
                Sche = loss_ori.item()/ini_loss
            else:
                Sche = 1.0
            loss_sch = loss_ori*Sche    
            # use_dls = False        
            if use_dls:# and 
                if ini_loss:
                    loss_inp = loss_sch /ini_loss
                else:
                    loss_inp = loss_sch*1.0
                if args.dls_act == 'logexp':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    # dls_coe0 = float(np.tanh(dls_coe0))
                    loss_act = args.dls_coe1*torch.log(torch.exp(loss_inp)-dls_coe0)
                    da_dl = args.dls_coe1*torch.exp(loss_inp) / (-dls_coe0 + torch.exp(loss_inp))
                elif args.dls_act == 'atan':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                elif args.dls_act == 'sech':
                    # dls_coe0 = max(args.dls_coe0,0.0)
                    # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                    dls_coe1 = args.dls_coe1
                    dls_coe0 = args.dls_coe0
                    if "cyclic_dls" in args.setting:
                        interval = args.epochs/args.rep_num
                        dls_coe1 = args.end*((epoch+interval)%(interval))/(interval+0.00000001)+args.dls_coe1
                    loss_act_ = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                    da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1  
                                  
                    if args.plus_base:
                        _loss_act = loss_act_+args.plus_base*loss_inp
                        da_dl = da_dl+args.plus_base
                    else:
                        _loss_act = loss_act_*1.0
                    if args.min_shift:
                        loss_act = _loss_act-min_bias
                    else:
                        loss_act = _loss_act*1.0
                        
                elif 'linear' in args.dls_act:
                    if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                        # dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                        loss_act = args.dls_coe0*loss_inp
                        # print('using linear')
                    else:
                        loss_act = loss_inp  *1.0
                if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                    loss_fin = ini_loss*loss_act
                    # print('ini_loss,loss_ori,loss_sch,loss_inp,loss_act,loss_fin',ini_loss,float(loss_ori.item()),float(loss_sch.item()),float(loss_inp.item()),float(loss_act.item()),float(loss_fin.item()))

                else:
                    loss_fin = loss_act*1.0#deepcopy?
                # print('loss2',loss_fin)
            else:
                loss_fin = loss_sch  *1.0  
            loss_fin.backward()

        output = model(images)
        loss_ori = criterion(output, targets,ini_loss, args, use_dls,epoch)
        # loss_ori = F.cross_entropy(output, targets)

        # pre_loss = loss_.clone().detach()
        
        # if float(torch.abs(pre_loss - loss_).item()) < epsilon and dls_coe:
        if 'LossSche' in args.setting and ini_loss:
            Sche = loss_ori.item()/ini_loss
        else:
            Sche = 1.0
        loss_sch = loss_ori*Sche    
        # use_dls = False  
        da_dl = 1.0      
        if False:#use_dls:# and 
            if ini_loss:
                loss_inp = loss_sch /ini_loss
            else:
                loss_inp = loss_sch*1.0
            if args.dls_act == 'logexp':
                dls_coe0 = max(args.dls_coe0,0.0)
                # dls_coe0 = float(np.tanh(dls_coe0))
                loss_act = args.dls_coe1*torch.log(torch.exp(args.dls_coe2*loss_inp)-dls_coe0)/args.dls_coe2
                da_dl = args.dls_coe1*torch.exp(args.dls_coe2*loss_inp) / (-dls_coe0 + torch.exp(args.dls_coe2*loss_inp))
            elif args.dls_act == 'atan':
                dls_coe0 = max(args.dls_coe0,0.0)
                loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
            elif args.dls_act == 'sech':
                # dls_coe0 = max(args.dls_coe0,0.0)
                # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                dls_coe1 = args.dls_coe1
                dls_coe0 = args.dls_coe0
                if "cyclic_dls" in args.setting:
                    interval = args.epochs/args.rep_num
                    dls_coe1 = args.end*((epoch+interval)%(interval))/(interval+0.00000001)+args.dls_coe1
                loss_act_ = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1  
                                
                if args.plus_base:
                    _loss_act = loss_act_+args.plus_base*loss_inp
                    da_dl = da_dl+args.plus_base
                else:
                    _loss_act = loss_act_*1.0
                if args.min_shift:
                    loss_act = _loss_act-min_bias
                else:
                    loss_act = _loss_act*1.0
            elif 'linear' in args.dls_act:
                if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                    # dls_coe0 = max(args.dls_coe0,0.0)
                    # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                    loss_act = args.dls_coe0*loss_inp
                    # print('using linear')
                else:
                    loss_act = loss_inp  *1.0
            else:
                loss_act = loss_inp  *1.0
            if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                loss_fin = ini_loss*loss_act
                # print('ini_loss,loss_ori,loss_sch,loss_inp,loss_act,loss_fin',ini_loss,float(loss_ori.item()),float(loss_sch.item()),float(loss_inp.item()),float(loss_act.item()),float(loss_fin.item()))

            else:
                loss_fin = loss_act*1.0#deepcopy?
            # print('loss2',loss_fin)
        else:
            loss_fin = loss_sch  *1.0          
        optimizer.zero_grad()
        loss_fin.backward()   
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
            # print('lr',optimizer.param_groups[0]['lr'])
        if use_closure: 
            # print('use_closure')
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr / (loss_ori.item()/ini_loss)
            # print('lr2',optimizer.param_groups[0]['lr'])
            
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss_ori.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        

        # if batch_idx % args.log_freq == 0:
        #     msg = ' '.join([
        #     'Epoch: {epoch}',
        #     '[{batch_id}/{batch_len}]',
        #     'lr:{lr:.6f}',
        #     'Train Loss:{train_loss:.4f}',
        #     'Train Acc1:{train_acc1:.4f}',
        #     'Train Acc5:{train_acc5:.4f}',
        #     'Time:{batch_time:.3f}s'])
        #     # if dls_coe:
        #     #     print('dls_coe0',dls_coe0)
        #     #     print('loss_,loss_fin',loss_,loss_fin)
        #     logger.log(
        #         msg.format(
        #             epoch=epoch, 
        #             batch_id=batch_idx, batch_len = len(train_loader),
        #             lr=optimizer.param_groups[0]["lr"],
        #             train_loss=_memory.meters["train_loss"].global_avg,
        #             train_acc1=_memory.meters["train_acc1"].global_avg,
        #             train_acc5=_memory.meters["train_acc5"].global_avg,
        #             batch_time=time.time() - batch_start,
        #         )
        #     )
        _memory.synchronize_between_processes()
        if tracked_dims is not None:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    # grad.append(torch.abs(p.grad).detach().view(-1))
                    grad.append(p.grad.detach().view(-1))

            grad = torch.cat(grad)
            tracked_grad = torch.norm(grad, p='fro')
            # for dim in tracked_dims:
            #     tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_ori.item())

    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }, grads, losses, da_dl

def train_one_epoch_LPF(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()
    c = math.log(1 + math.sqrt(2))
    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        inputs1 = torch.split(images, int(args.batch_size / args.M), dim=0)
        targets1 = torch.split(targets, int(args.batch_size / args.M), dim=0)

        for images, targets in zip(inputs1, targets1):
            # new technique
            with torch.no_grad():
                noise = []
                for mp in model.parameters():
                    if len(mp.shape) > 1:
                        sh = mp.shape
                        sh_mul = np.prod(sh[1:])
                        temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
                        temp = torch.normal(0, args.std*temp).to(mp.data.device)
                    else:
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, args.std*(mp.view(-1).norm().item() + 1e-16))
                    noise.append(temp)
                    mp.data.add_(noise[-1])
                        

            def closure():
                output = model(images)
                loss_ori = criterion(output, targets)/ args.M
                if 'LossSche' in args.setting and ini_loss:
                    Sche = loss_ori.item()/ini_loss
                else:
                    Sche = 1.0
                loss_sch = loss_ori*Sche    
                if use_dls:# and 
                    if ini_loss:
                        loss_inp = loss_sch /ini_loss
                    else:
                        loss_inp = loss_sch*1.0
                    if args.dls_act == 'logexp':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = float(np.tanh(dls_coe0))
                        loss_act = torch.log(torch.exp(loss_inp)-dls_coe0)
                    elif args.dls_act == 'atan':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                    elif args.dls_act == 'sech':
                        dls_coe1 = args.dls_coe1
                        dls_coe0 = args.dls_coe0
                        if "cyclic_dls" in args.setting:
                            interval = args.epochs/args.rep_num
                            dls_coe1 = ((epoch+args.dls_coe1*interval)%(interval))/(interval+0.000001)
                        loss_act = -dls_coe0*dls_coe1/(c*torch.cosh(c/dls_coe1*loss_inp))
                    elif 'linear' in args.dls_act:
                        if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:

                            loss_act = args.dls_coe0*loss_inp
                            # print('using linear')
                        else:
                            loss_act = loss_inp  *1.0
                    if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                        loss_fin = ini_loss*loss_act


                    else:
                        loss_fin = loss_act*1.0#deepcopy?
                    # print('loss2',loss_fin)
                else:
                    loss_fin = loss_sch  *1.0 
                loss_fin.backward()

            with torch.set_grad_enabled(True):

                output = model(images)
                loss_ori = criterion(output, targets)/ args.M

                if 'LossSche' in args.setting and ini_loss:
                    Sche = loss_ori.item()/ini_loss
                else:
                    Sche = 1.0
                loss_sch = loss_ori*Sche 
                da_dl = 1.0      
   
                if use_dls:# and 
                    if ini_loss:
                        loss_inp = loss_sch /ini_loss
                    else:
                        loss_inp = loss_sch*1.0
                    if args.dls_act == 'logexp':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = float(np.tanh(dls_coe0))
                        loss_act = torch.log(torch.exp(loss_inp)-dls_coe0)
                    elif args.dls_act == 'atan':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                    elif args.dls_act == 'sech':
                        dls_coe1 = args.dls_coe1
                        dls_coe0 = args.dls_coe0
                        if "cyclic_dls" in args.setting:
                            interval = args.epochs/args.rep_num
                            dls_coe1 = args.end*((epoch+interval)%(interval))/(interval+0.00000001)+args.dls_coe1
                        loss_act = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                        da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1
                        if 'plus0p1' in args.setting:
                            loss_act = loss_act+0.1*loss_inp
                    elif 'linear' in args.dls_act:
                        if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                            loss_act = args.dls_coe0*loss_inp
                        else:
                            loss_act = loss_inp  *1.0
                    if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                        loss_fin = ini_loss*loss_act

                    else:
                        loss_fin = loss_act*1.0#deepcopy?
                else:
                    loss_fin = loss_sch  *1.0          
                optimizer.zero_grad()
                loss_fin.backward() 
                

            # going back to without theta
            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)
                    
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
        if use_closure: 
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
            
        optimizer.zero_grad()
        
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr / (loss_ori.item()/ini_loss)
            
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss_ori.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        
        _memory.synchronize_between_processes()
        if tracked_dims:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    grad.append(torch.abs(p.grad).detach().view(-1))
            # print('type(grad),type(loss_fin):', type(grad),type(loss_fin))

            # print('len(grad):', len(grad))
            # print('grad[-1].shape:', grad[-1].shape)
            grad = torch.cat(grad)
            for dim in tracked_dims:
                tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_ori.item())
        # print('len(grads),len(losses):', len(grads),len(losses))        

    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }, grads, losses, da_dl


def train_one_epoch_smoothout(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()
    c = math.log(1 + math.sqrt(2))
    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            noise = []
            for mp in model.parameters():
                temp = torch.empty_like(mp, device=mp.data.device)
                temp.uniform_(-args.smooth_out_a, args.smooth_out_a) * torch.norm(mp.view(-1))
                noise.append(temp)
                mp.data.add_(noise[-1])
                
        with torch.set_grad_enabled(True):

            def closure():
                output = model(images)
                loss_ori = criterion(output, targets)#/ args.M
                if 'LossSche' in args.setting and ini_loss:
                    Sche = loss_ori.item()/ini_loss
                else:
                    Sche = 1.0
                loss_sch = loss_ori*Sche    
                # use_dls = False        
                if use_dls:# and 
                    if ini_loss:
                        loss_inp = loss_sch /ini_loss
                    else:
                        loss_inp = loss_sch*1.0
                    if args.dls_act == 'logexp':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = float(np.tanh(dls_coe0))
                        loss_act = torch.log(torch.exp(loss_inp)-dls_coe0)
                    elif args.dls_act == 'atan':
                        dls_coe0 = max(args.dls_coe0,0.0)
                        loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                    elif args.dls_act == 'sech':
                        # dls_coe0 = max(args.dls_coe0,0.0)
                        # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                        dls_coe1 = args.dls_coe1
                        dls_coe0 = args.dls_coe0
                        if "cyclic_dls" in args.setting:
                            interval = args.epochs/args.rep_num
                            dls_coe1 = ((epoch+args.dls_coe1*interval)%(interval))/(interval+0.000001)
                        loss_act = -dls_coe0*dls_coe1/(c*torch.cosh(c/dls_coe1*loss_inp))
                    elif 'linear' in args.dls_act:
                        if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                            # dls_coe0 = max(args.dls_coe0,0.0)
                            # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                            loss_act = args.dls_coe0*loss_inp
                            # print('using linear')
                        else:
                            loss_act = loss_inp  *1.0
                    if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                        loss_fin = ini_loss*loss_act

                    else:
                        loss_fin = loss_act*1.0#deepcopy?
                    # print('loss2',loss_fin)
                else:
                    loss_fin = loss_sch  *1.0 
                loss_fin.backward()

            output = model(images)
            loss_ori = criterion(output, targets) #/ args.M #TODO: maybe should devide it after operations to the loss  
            # loss_ori = F.cross_entropy(output, targets)

            # pre_loss = loss_.clone().detach()
            
            # if float(torch.abs(pre_loss - loss_).item()) < epsilon and dls_coe:
            if 'LossSche' in args.setting and ini_loss:
                Sche = loss_ori.item()/ini_loss
            else:
                Sche = 1.0
            loss_sch = loss_ori*Sche    
            # use_dls = False      
            da_dl = 1.0   
            if use_dls:# and 
                if ini_loss:
                    loss_inp = loss_sch /ini_loss
                else:
                    loss_inp = loss_sch*1.0
                if args.dls_act == 'logexp':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    # dls_coe0 = float(np.tanh(dls_coe0))
                    loss_act = torch.log(torch.exp(loss_inp)-dls_coe0)
                elif args.dls_act == 'atan':
                    dls_coe0 = max(args.dls_coe0,0.0)
                    loss_act = (args.dls_coe0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
                elif args.dls_act == 'sech':
                    # dls_coe0 = max(args.dls_coe0,0.0)
                    # loss_act = -args.dls_coe0*1/torch.cosh(args.dls_coe1*loss_inp)**2   #dls_coe0=1, dls_coe1=4
                    dls_coe1 = args.dls_coe1
                    dls_coe0 = args.dls_coe0
                    if "cyclic_dls" in args.setting:
                        interval = args.epochs/args.rep_num
                        # dls_coe1 = ((epoch+args.dls_coe1*interval)%(interval))/(interval+0.000001)
                        dls_coe1 = args.end*((epoch+interval)%(interval))/(interval+0.00000001)+args.dls_coe1
                    loss_act = -dls_coe0*dls_coe1/(c*torch.cosh(c/(dls_coe1+0.000001)*loss_inp)+0.000001)
                    da_dl = 2*dls_coe0/torch.cosh(c/dls_coe1*loss_inp)*torch.tanh(c/dls_coe1*loss_inp)#+1

                    if 'plus0p1' in args.setting:
                        loss_act = loss_act+0.1*loss_inp
                elif 'linear' in args.dls_act:
                    if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:
                        # dls_coe0 = max(args.dls_coe0,0.0)
                        # dls_coe0 = 1+max(0.5*dls_coe0,-0.9)
                        loss_act = args.dls_coe0*loss_inp
                        # print('using linear')
                    else:
                        loss_act = loss_inp  *1.0
                if ini_loss and not 'no_unnorm' in args.setting:# and not args.dls_act == 'sech':
                    loss_fin = ini_loss*loss_act


                else:
                    loss_fin = loss_act*1.0#deepcopy?
                # print('loss2',loss_fin)
            else:
                loss_fin = loss_sch  *1.0    
            optimizer.zero_grad()
            loss_fin.backward()   
        
        with torch.no_grad():
            for mp, n in zip(model.parameters(), noise):
                mp.data.sub_(n)
                    
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
            # print('lr',optimizer.param_groups[0]['lr'])
        if use_closure: 
            # print('use_closure')
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
            optimizer.zero_grad()

            
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr / (loss_ori.item()/ini_loss)
            # print('lr2',optimizer.param_groups[0]['lr'])
            
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss_ori.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        

        _memory.synchronize_between_processes()
        if tracked_dims:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    grad.append(torch.abs(p.grad).detach().view(-1))
            # print('type(grad),type(loss_fin):', type(grad),type(loss_fin))

            # print('len(grad):', len(grad))
            # print('grad[-1].shape:', grad[-1].shape)
            grad = torch.cat(grad)
            for dim in tracked_dims:
                tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_ori.item())
        # print('len(grads),len(losses):', len(grads),len(losses))        

    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }, grads, losses, da_dl


def LinExp(loss, dls_coe0, beta=1):
    alpha = 1 - beta * dls_coe0
    return alpha * loss + beta * (torch.exp(dls_coe0 * loss) - 1)

def train_one_epoch_simple(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()
    c = math.log(1 + math.sqrt(2))
    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    da_dl = 0

        
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # print('batch_idx',batch_idx)

        output = model(images)
        loss = criterion(output, targets)
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss.item(), n=batch_num)
        # if 'baseline' in args.setting:
        #     loss = loss  
        if 'logexp' in args.dls_act:
            loss = torch.log(torch.exp(loss)-args.dls_coe0)
        elif 'LinQuad' in args.dls_act:
            loss = LinQuad(loss, args.dls_coe0)
            
        optimizer.zero_grad()
        loss.backward()   
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
            # print('lr',optimizer.param_groups[0]['lr'])
        if use_closure: 
            # print('use_closure')
            optimizer.step(
                closure, 
                epoch=epoch,
                step=epoch*len(train_loader)+batch_idx,
                batch_idx=batch_idx,
                model=model, 
                train_data=train_loader.dataset, 
                logger=logger,
            ) 
        else: 
            optimizer.step()
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr / (loss_ori.item()/ini_loss)
            # print('lr2',optimizer.param_groups[0]['lr'])
            
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        

        _memory.synchronize_between_processes()
        if tracked_dims is not None:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    # grad.append(torch.abs(p.grad).detach().view(-1))
                    grad.append(p.grad.detach().view(-1))

            grad = torch.cat(grad)
            tracked_grad = torch.norm(grad, p='fro')
            # for dim in tracked_dims:
            #     tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_ori.item())

    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }, grads, losses, da_dl


# def train_one_epoch_F(
#     model: torch.nn.Module,
#     train_loader : Iterable,
#     criterion, optimizer, epoch, logger, log_freq, use_closure, dls_coe, dls_act, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
# ):
#     model.train()

#     _memory = MetricLogger()
#     _memory.add_meter('train_loss', Metric())
#     _memory.add_meter('train_acc1', Metric())
#     _memory.add_meter('train_acc5', Metric())
#     epsilon = 0.1
#     for batch_idx, (images, targets) in enumerate(train_loader):
#         batch_start = time.time()
#         images = images.cuda(non_blocking=True)
#         targets = targets.cuda(non_blocking=True)

#         def closure():
#             output = model(images)
#             loss_ = criterion(output, targets)
#             if dls_coe:
#                 print('loss_fin',loss_)
#                 if dls_act == 'logexp':
#                     loss_fin = torch.log(torch.exp(loss_-shift)-dls_coe0)
#                 elif dls_act == 'atan':
#                     loss_fin = dls_coe[0]*torch.atan(dls_coe[1]*loss_)**2 + loss_
#                 # print('loss2',loss_fin)
#             else:
#                 loss_fin = loss_
#             loss_fin.backward()

#         output = model(images)
#         loss_ = F.cross_entropy(output, targets)
#         # pre_loss = loss_.clone().detach()
        
#         # if float(torch.abs(pre_loss - loss_).item()) < epsilon and dls_coe:
#         if dls_coe:# and 
#             if ini_loss:
#                 loss_inp = loss_ /ini_loss
#             else:
#                 loss_inp = loss_
#             if dls_act == 'logexp':
#                 dls_coe0 = max(dls_coe[0],0.0)
#                 # dls_coe0 = float(np.tanh(dls_coe0))
#                 loss_act = torch.log(torch.exp(loss_inp)-dls_coe0)
#             elif dls_act == 'atan':
#                 dls_coe0 = max(dls_coe[0],0.0)
#                 loss_act = (dls_coe0*torch.atan(dls_coe[1]*loss_inp)**2 + loss_inp)
#             elif dls_act == 'linear':
#                 dls_coe0 = torch.tensor(max(dls_coe[0],0.0))
#                 # dls_coe0 = 1+max(0.5*dls_coe[0],-0.9)
#                 loss_act = dls_coe0*loss_inp
#             if ini_loss:
#                 loss_fin = ini_loss*loss_act
#                 # print('ini_loss,loss_,loss_inp,loss_act,loss_fin',ini_loss,loss_,loss_inp,loss_act,loss_fin)
#             else:
#                 loss_fin = loss_act
#             # print('loss2',loss_fin)
#         else:
#             loss_fin = loss_            
#         optimizer.zero_grad()
#         loss_fin.backward()   
#         if use_closure: 
#             optimizer.step(
#                 closure, 
#                 epoch=epoch,
#                 step=epoch*len(train_loader)+batch_idx,
#                 batch_idx=batch_idx,
#                 model=model, 
#                 train_data=train_loader.dataset, 
#                 logger=logger,
#             ) 
#         else: 
#             optimizer.step()

#         acc1, acc5 = accuracy(output, targets, topk=(1, 5))
#         batch_num = images.shape[0]
#         _memory.update_meter('train_loss', loss_.item(), n=batch_num)
#         _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
#         _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        

#         # if batch_idx % log_freq == 0:
#         #     msg = ' '.join([
#         #     'Epoch: {epoch}',
#         #     '[{batch_id}/{batch_len}]',
#         #     'lr:{lr:.6f}',
#         #     'Train Loss:{train_loss:.4f}',
#         #     'Train Acc1:{train_acc1:.4f}',
#         #     'Train Acc5:{train_acc5:.4f}',
#         #     'Time:{batch_time:.3f}s'])
#         #     if dls_coe:
#         #         print('dls_coe0',dls_coe0)
#         #         print('loss_,loss_fin',loss_,loss_fin)
#         #     logger.log(
#         #         msg.format(
#         #             epoch=epoch, 
#         #             batch_id=batch_idx, batch_len = len(train_loader),
#         #             lr=optimizer.param_groups[0]["lr"],
#         #             train_loss=_memory.meters["train_loss"].global_avg,
#         #             train_acc1=_memory.meters["train_acc1"].global_avg,
#         #             train_acc5=_memory.meters["train_acc5"].global_avg,
#         #             batch_time=time.time() - batch_start,
#         #         )
#         #     )
#         _memory.synchronize_between_processes()
#         grad = []
#         tracked_grad = []
#         for p in model.parameters():
#             if p.grad is not None:
#                 grad.append(torch.abs(p.grad).detach().view(-1))
#         grad = torch.cat(grad)
#         print('grad.max()',grad.max())
                    
#         if tracked_dims:
#             # TODO: should be in the outside indent
#             grad = []
#             tracked_grad = []
#             for p in model.parameters():
#                 if p.grad is not None:
#                     grad.append(torch.abs(p.grad).detach().view(-1))
#             # print('type(grad),type(loss_fin):', type(grad),type(loss_fin))

#             # print('len(grad):', len(grad))
#             # print('grad[-1].shape:', grad[-1].shape)
#             grad = torch.cat(grad)
#             for dim in tracked_dims:
#                 tracked_grad.append(grad[dim])  
#             grads.append(torch.tensor(tracked_grad))
#             losses.append(loss_.item())
#         # print('len(grads),len(losses):', len(grads),len(losses))        

#     return {
#         name: meter.global_avg for name, meter in _memory.meters.items()
#     }, grads, losses



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    _memory = MetricLogger()
    _memory.add_meter('test_loss', Metric())
    _memory.add_meter('test_acc1', Metric())
    _memory.add_meter('test_acc5', Metric())

    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss_fin = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('test_loss', loss_fin.item(), n=batch_num)
        _memory.update_meter('test_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('test_acc5', acc5.item(), n=batch_num)
    _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }
    
@torch.no_grad()
def evaluate_val(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    _memory = MetricLogger()
    _memory.add_meter('val_loss', Metric())
    _memory.add_meter('val_acc1', Metric())
    _memory.add_meter('val_acc5', Metric())

    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss_fin = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('val_loss', loss_fin.item(), n=batch_num)
        _memory.update_meter('val_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('val_acc5', acc5.item(), n=batch_num)
    _memory.synchronize_between_processes()
    return {
        name: meter.global_avg for name, meter in _memory.meters.items()
    }    

def accuracy(output, targets, topk=(1,)):
    # output: [b, n]
    # targets: [b]
    batch_size, n_classes = output.size()
    maxk = min(max(topk), n_classes)
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t() # pred: [b, maxk] -> [maxk, b]
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) # targets: [b] -> [1, b] -> [maxk, b]; correct(bool): [maxk, b]
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Metric:
    def __init__(self) -> None:
        self.value = 0
        self.num = 0
    
    def update(self, value, n=1):
        self.num += n
        self.value += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.num, self.value], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.num = int(t[0])
        self.value = t[1]
    
    @property
    def global_avg(self):
        if not self.num == 0:
            return self.value / self.num
        else:
            return 0

class MetricLogger:
    def __init__(self) -> None:
        self.meters = defaultdict(Metric)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def update_meter(self, name, value, n):
        self.meters[name].update(value, n)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()