import os
import json
import random
import argparse
import numpy as np
import torchprofile

import torch
import torch.nn as nn
from utils import raw2cfg
from models import VGG_CIFAR, ResNet_CIFAR, ResNet_ImageNet, MobileNet, MobileNetV2, SearchRunConfig
from run_manager import RunManager

parser = argparse.ArgumentParser()

""" basic config """
parser.add_argument('--path', type=str, default='Exp/debug')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=100)

""" optimizer config """
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--warm_epoch', type=int, default=5)
parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default='bn#bias', choices=[None, 'bn', 'bn#bias'])

""" dataset config """
parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/userhome/data/cifar10')
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=250)

""" search config """
parser.add_argument('--model', type=str, default="resnet18",\
                               choices=['vgg','resnet56', 'resnet110', 'resnet18', 'resnet34', 'resnet50', 'mobilenet', 'mobilenetv2'])
parser.add_argument('--div', type=int, default=8)
parser.add_argument('--search_epoch', type=int, default=10)
parser.add_argument('--target_flops', type=float, default=100e6)
parser.add_argument('--n_remove', type=int, default=2)
parser.add_argument('--n_best', type=int, default=3)
parser.add_argument('--n_populations', type=int, default=8)
parser.add_argument('--n_generations', type=int, default=25)  # 25 generations seems enough

""" runtime config """
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument("--local_rank", default=0, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.makedirs(args.path, exist_ok=True)

    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = SearchRunConfig(
        **args.__dict__
    )
    if args.local_rank == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    base_flops = 0
    if args.model == 'resnet18':
        assert args.dataset=='imagenet', 'resnet18 only supports imagenet dataset'
        net = ResNet_ImageNet(num_classes=1000, cfg=None, depth=18)
        weight_path = 'Exp_base/resnet18_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(ResNet_ImageNet(depth=18, num_classes=1000))
    elif args.model == 'resnet34':
        assert args.dataset=='imagenet', 'resnet34 only supports imagenet dataset'
        net = ResNet_ImageNet(num_classes=1000, cfg=None, depth=34)
        weight_path = 'Exp_base/resnet34_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(ResNet_ImageNet(depth=34, num_classes=1000))
    elif args.model == 'resnet50':
        assert args.dataset=='imagenet', 'resnet50 only supports imagenet dataset'
        net = ResNet_ImageNet(num_classes=1000, cfg=None, depth=50)
        weight_path = 'Exp_base/resnet50_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(ResNet_ImageNet(depth=50, num_classes=1000))
    elif args.model == 'mobilenet':
        assert args.dataset=='imagenet', 'mobilenet only supports imagenet dataset'
        net = MobileNet(num_classes=1000, cfg=None)
        weight_path = 'Exp_base/mobilenet_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(MobileNet(num_classes=1000))
    elif args.model == 'mobilenetv2':
        assert args.dataset=='imagenet', 'mobilenetv2 only supports imagenet dataset'
        net = MobileNetV2(num_classes=1000, cfg=None)
        weight_path = 'Exp_base/mobilenetv2_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(MobileNetV2(num_classes=1000))
    elif args.model == 'vgg':
        assert args.dataset=='cifar10', 'vgg only supports cifar10 dataset'
        net = VGG_CIFAR(cfg=None, cutout=False)
        weight_path = 'Exp_base/vgg_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(VGG_CIFAR(cutout=False))
    elif args.model=="resnet56":
        assert args.dataset=='cifar10', 'resnet56 only supports cifar10 dataset'
        net = ResNet_CIFAR(cfg=None, depth=56, num_classes=10, cutout=False)
        weight_path = 'Exp_base/resnet56_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(ResNet_CIFAR(depth=56, num_classes=10, cutout=False))
    elif args.model=="resnet110":
        assert args.dataset=='cifar10', 'resnet110 only supports cifar10 dataset'
        net = ResNet_CIFAR(cfg=None, depth=110, num_classes=10, cutout=False)
        weight_path = 'Exp_base/resnet110_base/checkpoint/model_best.pth.tar'
        net_origin = nn.DataParallel(ResNet_CIFAR(depth=110, num_classes=10, cutout=False))
    net_origin.load_state_dict(torch.load(weight_path)['state_dict'])
    net_origin = net_origin.module
    base_flops = net.cfg2flops(net.config['cfg_base'])

    # build run manager
    run_manager = RunManager(args.path, net, run_config)
    if args.local_rank == 0:
        run_manager.save_config(print_info=True)

    # cfg to fitness
    cfg2fit_dict = {}

    def cfg2fitness(cfg):
        if args.local_rank == 0:
            print(str(cfg))
        if str(cfg) in cfg2fit_dict.keys():
            return cfg2fit_dict[str(cfg)]
        elif cfg==run_manager.net.module.config['cfg_base']:
            return 0.
        else:
            run_manager.run_config.n_epochs = run_manager.run_config.search_epoch
            if args.model == 'resnet18':
                run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=18), net_origin.cpu())
            elif args.model == 'resnet34':
                run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=34), net_origin.cpu())
            elif args.model == 'resnet50':
                run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=50), net_origin.cpu())
            elif args.model == 'mobilenet':
                run_manager.reset_model(MobileNet(num_classes=1000, cfg=cfg), net_origin.cpu())
            elif args.model == 'mobilenetv2':
                run_manager.reset_model(MobileNetV2(num_classes=1000, cfg=cfg), net_origin.cpu())
            elif args.model == 'vgg':
                run_manager.reset_model(VGG_CIFAR(cfg=cfg, cutout=False, num_classes=10), net_origin.cpu())
            elif args.model == 'resnet56':
                run_manager.reset_model(ResNet_CIFAR(cfg=cfg, depth=56, num_classes=10, cutout=False), net_origin.cpu())
            elif args.model== 'resnet110':
                run_manager.reset_model(ResNet_CIFAR(cfg=cfg, depth=110, num_classes=10, cutout=False), net_origin.cpu())
            run_manager.start_epoch = 0
            run_manager.train()
            _, acc1, _ = run_manager.validate(is_test=False, return_top5=True)
            cfg2fit_dict[str(cfg)] = acc1.item()
            return acc1.item()

    # params to be pruned
    run_manager.net.train()
    target_flops = run_manager.run_config.target_flops
    n_population = run_manager.run_config.n_populations
    n_generations = run_manager.run_config.n_generations

    # initialization
    populations = []
    cfgs = []
    fitness = []
    best_acc = 0
    best_cfg = None
    best_cfg_raw = None
    if args.local_rank == 0:
        print('=' * 40 + 'initialization' + '=' * 40)
    cfg_len = len(net.config['cfg_base'])
    for p in range(n_population):
        rand_sample = np.random.rand(cfg_len)
        rand_cfg = raw2cfg(net, np.array(rand_sample), target_flops, div=args.div)
        populations.append(np.array(rand_sample))
        cfgs.append(rand_cfg)
        fitness.append(cfg2fitness(rand_cfg))
    if args.local_rank == 0:
        run_manager.write_log('cfgs: ' + str(cfgs), 'valid')
        run_manager.write_log('fitness: ' + str(fitness), 'valid')
        run_manager.write_log('avg fitness: ' + str(np.mean(fitness)), 'prune')
        run_manager.write_log('best fitness: ' + str(np.max(fitness)), 'prune')
    current_best = np.array(fitness).max()
    if current_best > best_acc:
        best_acc = current_best
        best_cfg = cfgs[np.array(fitness).argsort()[-1]]
        best_cfg_raw = populations[np.array(fitness).argsort()[-1]]
    if args.local_rank == 0:
        run_manager.write_log('best cfg: ' + str(best_cfg), 'prune')
        run_manager.write_log('best val acc: ' + str(best_acc), 'prune')

    # evolutionary search
    if args.local_rank == 0:
        print('=' * 40 + 'evolving' + '=' * 40)
    for g in range(n_generations):
        if args.local_rank == 0:
            run_manager.write_log("=" * 20 + "generation %d" % g + "=" * 20, 'prune')
        idx_remove = np.array(fitness).argsort()[:args.n_remove]
        idx_best = np.array(fitness).argsort()[n_population-args.n_best:]
        cfgs = []
        fitness = []
        idx_shuffle = [i for i in range(n_population) if i not in idx_remove and i not in idx_best]
        old_populations = np.array(populations)
        for idx in range(len(populations)):
            if idx in idx_remove:
                populations[idx] = np.random.rand(cfg_len)
            elif idx in idx_shuffle:
                idxs = random.sample(range(n_population), 2)
                i = idxs[0]
                j = idxs[1]
                populations[idx] = (old_populations[i] + old_populations[j]) / 2
            else:
                populations[idx] = old_populations[idx]
            if args.local_rank == 0:
                print(populations[idx])
            cfg = raw2cfg(net, populations[idx], target_flops, div=args.div)
            cfgs.append(cfg)
            fitness.append(cfg2fitness(cfg))
        if args.local_rank == 0:
            run_manager.write_log('cfgs: ' + str(cfgs), 'valid')
            run_manager.write_log('fitness: ' + str(fitness), 'valid')
            run_manager.write_log('avg fitness: ' + str(np.mean(fitness)), 'prune')
            run_manager.write_log('best fitness: ' + str(np.max(fitness)), 'prune')
        current_best = np.array(fitness).max()
        if current_best > best_acc:
            best_acc = current_best
            best_cfg = cfgs[np.array(fitness).argsort()[-1]]
            best_cfg_raw = populations[np.array(fitness).argsort()[-1]]
        if args.local_rank == 0:
            run_manager.write_log('best cfg: ' + str(best_cfg), 'prune')
            run_manager.write_log('best val acc: ' + str(best_acc), 'prune')
    fitness = best_acc
    cfg = list(best_cfg)

    # final fine-tuning
    if args.model == 'resnet18':
        run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=18), net_origin.cpu())
    elif args.model == 'resnet34':
        run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=34), net_origin.cpu())
    elif args.model == 'resnet50':
        run_manager.reset_model(ResNet_ImageNet(num_classes=1000, cfg=cfg, depth=50), net_origin.cpu())
    elif args.model == 'mobilenet':
        run_manager.reset_model(MobileNet(num_classes=1000, cfg=cfg), net_origin.cpu())
    elif args.model == 'mobilenetv2':
        run_manager.reset_model(MobileNetV2(num_classes=1000, cfg=cfg), net_origin.cpu())
    elif args.model == 'vgg':
        run_manager.reset_model(VGG_CIFAR(cfg=cfg, cutout=True), net_origin.cpu())
    elif args.model == 'resnet56':
        run_manager.reset_model(ResNet_CIFAR(cfg=cfg, depth=56, num_classes=10, cutout=True), net_origin.cpu())
    elif args.model=="resnet110":
        run_manager.reset_model(ResNet_CIFAR(cfg=cfg, depth=110, num_classes=10, cutout=True), net_origin.cpu())

    # train
    run_manager.run_config.n_epochs = args.n_epochs
    if args.train:
        if args.local_rank == 0:
            print('Start training: %d' % args.local_rank)
        run_manager.train(print_top5=True)
        if args.local_rank == 0:
            run_manager.save_model()

    output_dict = {}

    # test
    if args.local_rank == 0:
        print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    if args.local_rank == 0:
        run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)
