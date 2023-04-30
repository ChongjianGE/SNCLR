import argparse
import math
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import snclr.builder_snclr_factor
import snclr.loader
import snclr.optimizer

import vits
from log import setup_logger
import ipdb


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small'] + torchvision_model_names

parser = argparse.ArgumentParser(description='SNCLR ImageNet Pre-Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int, metavar='N', help='mini-batch size (default: 4096), this is the total ')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,  metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-6)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=10, type=int,  metavar='N', help='save frequency (default: 10)')
parser.add_argument('--workdir', default='./train_output/snclr/', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,  help='node rank for distributed training')
parser.add_argument('--dist-url', default='None', type=str,  help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch ')

# for the training process
parser.add_argument('--snclr-dim', default=256, type=int,  help='feature dimension (default: 256)')
parser.add_argument('--hidden-dim', default=384, type=int, help='feature dimension (default: 384)')
parser.add_argument('--snclr-mlp-dim', default=4096, type=int,help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--snclr-m', default=0.99, type=float, help='snclr momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--snclr-m-cos', action='store_true',help='gradually increase snclr momentum to 1 with a half-cycle cosine schedule')
parser.add_argument('--snclr-t', default=1.0, type=float, help='softmax temperature (default: 1.0)')

# for the mdoel architecture
parser.add_argument('--stop-grad-conv1', action='store_true', help='stop-grad after first conv, or patch embedding')
parser.add_argument('--num-prototype', default=4, type=int, help='prototype for clustering')
parser.add_argument('--num-topk', default=1, type=int, help='prototype for clustering')
parser.add_argument('--loss-type', default='nce', type=str, choices=['nce', 'single_nce', 'single_nce_second', 'wnce', 'allnce', 'allposnce', 'partposnce','nce_var', 'pair_pos_neg_nce', 'pair_pos_nce', 'pair_pos_nce_v2'],help='loss used (default: nce)')

# other configurations
parser.add_argument('--optimizer', default='lars', type=str,choices=['lars', 'adamw'],help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,help='minimum scale for random cropping (default: 0.08)')

parser.add_argument('--ssl-type', default='snclr_factor', type=str, choices=['snclr_factor'],help='ssl used (default: snclr_factor)')
parser.add_argument('--bank-size', default=128000, type=int, metavar='N', help='size of the bank')
parser.add_argument('--bank-epoch', default=0, type=int, metavar='N', help='epoch of the bank')

parser.add_argument('--threshold', action='store_true', help='control the threshold')
parser.add_argument('--topk', default=5, type=int, metavar='N', help='topk num')

# saving files
parser.add_argument('--script', default='start.sh', type=str, help='saving scripts')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print(f'for this exp, we have rank {args.rank}')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(f'for this exp, we have ranks like {args.rank} - the gpu is {args.gpu}')

    if args.multiprocessing_distributed:
        logger = setup_logger(args.workdir, gpu=args.gpu, rank=args.rank, filename="train_log.txt", mode="a")

    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        pass

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    if args.gpu is not None:
        logger.info("=> creating model '{}'".format(args.arch))
        logger.info('we use the ssl type as {}'.format(args.ssl_type))
        logger.info('we use the num topk as {}'.format(args.topk))
        logger.info('we use the epoch as {} to warm up bank'.format(args.bank_epoch))
        logger.info("we use bank size: {} for training".format(args.bank_size))
        logger.info("we will save the script called {}".format(args.script))

    if args.arch.startswith('vit'):
        if args.ssl_type == 'snclr_factor':
            model = snclr.builder_snclr_factor.SNCLR_ViT(
                partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1, ),
                args.snclr_dim, args.snclr_mlp_dim, args.snclr_t, bank_size=args.bank_size, model='vit',
                world_size=args.world_size, threshold=args.threshold, topk=args.topk,
                batch=int(args.batch_size / args.world_size)
            )


    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        if args.gpu is not None:
            logger.info("=> creating model '{}'".format(args.arch))
    elif args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    if args.gpu is not None:
        logger.info('here is the model, but currently we do not print it')

    if args.optimizer == 'lars':
        optimizer = snclr.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
        if not os.path.exists(args.workdir):
            os.makedirs(args.workdir, exist_ok=True)
        files = os.listdir('./snclr/')
        if not os.path.exists(os.path.join(args.workdir, 'snclr')):
            os.makedirs(os.path.join(args.workdir, 'snclr'), exist_ok=True)
        for file_ in files:
            if '.py' in file_:
                shutil.copy(os.path.join('./snclr/', file_), os.path.join(args.workdir, 'snclr/'))

        if not os.path.exists(os.path.join(args.workdir, 'launch_scripts')):
            os.makedirs(os.path.join(args.workdir, 'launch_scripts'), exist_ok=True)
        shutil.copy(os.path.join('./', args.script), os.path.join(args.workdir, 'launch_scripts/'))

    summary_writer = SummaryWriter(os.path.join(args.workdir, 'runs')) if args.rank == 0 else None

    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is not None:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            if args.gpu is not None:
                logger.info("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
        else:
            if args.gpu is not None:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([snclr.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([snclr.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([snclr.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        snclr.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                      transforms.Compose(augmentation2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, logger)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            if epoch % args.save_freq == 0 or epoch == 299:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=True, filename=os.path.join(args.workdir, 'checkpoint_%04d.pth.tar' % epoch), pre_fix=args.workdir)

    if args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    losses_con = AverageMeter('Loss_CON', ':.4e')
    losses_sim = AverageMeter('Loss_SIM', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, losses_con, losses_sim],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs),
        logger=logger,
        args=args)

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    snclr_m = args.snclr_m
    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.snclr_m_cos:
            snclr_m = adjust_snclr_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss, loss_con, loss_sim = model(images[0], images[1], snclr_m, epoch)
        losses.update(loss.item(), images[0].size(0))
        losses_con.update(loss_con.item(), images[0].size(0))
        losses_sim.update(loss_sim.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("loss_con", loss_con.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("loss_sim", loss_sim.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', pre_fix=None):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(pre_fix, 'model_best.pth.tar'))


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None, args=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger
        self.args=args

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.args.gpu is not None:
            self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_snclr_momentum(epoch, args):
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.snclr_m)
    return m


if __name__ == '__main__':
    main()
