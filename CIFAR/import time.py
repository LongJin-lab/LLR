import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
from utils.dist import is_dist_avail_and_initialized
import numpy as np
torch.autograd.set_detect_anomaly(True)
c = math.log(1 + math.sqrt(2))
def train_one_epoch(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, use_closure, use_dls, args, grads=[],losses=[], tracked_dims=None, shift=0, ini_loss=None
):
    model.train()

    _memory = MetricLogger()
    _memory.add_meter('train_loss', Metric())
    _memory.add_meter('train_acc1', Metric())
    _memory.add_meter('train_acc5', Metric())
    epsilon = 0.1
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)


        output = model(images)
        loss_ori = criterion(output, targets)

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
                dls_coe_0 = max(args.dls_coe0,0.0)
                # dls_coe_0 = float(np.tanh(dls_coe_0))
                loss_act = torch.log(torch.exp(loss_inp)-dls_coe_0)
            elif args.dls_act == 'atan':
                dls_coe_0 = max(args.dls_coe0,0.0)
                loss_act = (args.dls_coe_0*torch.atan(args.dls_coe1*loss_inp)**2 + loss_inp)
            elif args.dls_act == 'sech':

                loss_act = -args.dls_coe0*args.dls_coe1/(c*torch.cosh(c/args.dls_coe1*loss_inp))
            elif 'linear' in args.dls_act:
                if float(loss_inp.item())<=args.dls_start and float(loss_inp.item())>=args.dls_end:

                    loss_act = args.dls_coe0*loss_inp
                else:
                    loss_act = loss_inp  *1.0
            if ini_loss and not args.dls_act == 'sech':
                loss_fin = ini_loss*loss_act


            else:
                loss_fin = loss_act*1.0#deepcopy?
        else:
            loss_fin = loss_sch  *1.0          
        optimizer.zero_grad()
        loss_fin.backward()   
        if 'LossRepLR' in args.setting and ini_loss:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = current_lr * loss_ori.item()/ini_loss
        if use_closure: 
            print('use_closure')

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
    }, grads, losses



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