import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.distributed as dist
from utils.dist import is_dist_avail_and_initialized

def train_one_epoch(
    model: torch.nn.Module,
    train_loader : Iterable,
    criterion, optimizer, epoch, logger, log_freq, use_closure, dls_coe, dls_act, grads=[],losses=[], tracked_dims=None, shift=0
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

        def closure():
            output = model(images)
            loss_ = criterion(output, targets)
            if dls_coe:
                # print('loss',loss_)
                if dls_act == 'logexp':
                    loss = torch.log(torch.exp(loss_-shift)-dls_coe[0])
                elif dls_act == 'atan':
                    loss = dls_coe[0]*torch.atan(dls_coe[1]*loss_)**2 + loss_
                # print('loss2',loss)
            else:
                loss = loss_
            loss.backward()

        output = model(images)
        loss_ = criterion(output, targets)
        # pre_loss = loss_.clone().detach()
        
        # if float(torch.abs(pre_loss - loss_).item()) < epsilon and dls_coe:
        if dls_coe:
                # print('loss',loss_)
                if dls_act == 'logexp':
                    loss = torch.log(torch.exp(loss_-shift)-dls_coe[0])
                elif dls_act == 'atan':
                    loss = dls_coe[0]*torch.atan(dls_coe[1]*loss_)**2 + loss_
                # print('loss2',loss)
        else:
            loss = loss_            
        optimizer.zero_grad()
        loss.backward()   
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

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        _memory.update_meter('train_loss', loss_.item(), n=batch_num)
        _memory.update_meter('train_acc1', acc1.item(), n=batch_num)
        _memory.update_meter('train_acc5', acc5.item(), n=batch_num)
        
        msg = ' '.join([
            'Epoch: {epoch}',
            '[{batch_id}/{batch_len}]',
            'lr:{lr:.6f}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Time:{batch_time:.3f}s'])
        if batch_idx % log_freq == 0:
            logger.log(
                msg.format(
                    epoch=epoch, 
                    batch_id=batch_idx, batch_len = len(train_loader),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=_memory.meters["train_loss"].global_avg,
                    train_acc1=_memory.meters["train_acc1"].global_avg,
                    train_acc5=_memory.meters["train_acc5"].global_avg,
                    batch_time=time.time() - batch_start,
                )
            )
        _memory.synchronize_between_processes()
        if tracked_dims:
            # TODO: should be in the outside indent
            grad = []
            tracked_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    grad.append(torch.abs(p.grad).detach().view(-1))
            # print('type(grad),type(loss):', type(grad),type(loss))

            # print('len(grad):', len(grad))
            # print('grad[-1].shape:', grad[-1].shape)
            grad = torch.cat(grad)
            for dim in tracked_dims:
                tracked_grad.append(grad[dim])  
            grads.append(torch.tensor(tracked_grad))
            losses.append(loss_.item())
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
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('test_loss', loss.item(), n=batch_num)
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
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        _memory.update_meter('val_loss', loss.item(), n=batch_num)
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
