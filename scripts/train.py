import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./scripts')

from lib.net import *
from lib.trainer import *
from lib.dataset import VoxelDataset
import lib.DualVoxel as DV

import lib.logger as logger

import warnings
warnings.filterwarnings('ignore')

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    
    # log config
    parser.add_argument('--voxel', type=str, default='data/thingi32_voxel/256/441708.npz', help='Base Voxel file')
    parser.add_argument('--exp_name', type=str, default='./logs/441708', help='Experiment name')    

    # phase1
    parser.add_argument('--p1_reso', type=int, default=32)
    parser.add_argument('--p1_epoch', type=int, default=250)
    parser.add_argument('--p1_nas_e2', type=int, default=40)
    parser.add_argument('--p1_nas_e3', type=int, default=100)

    # phase2
    parser.add_argument('--p2_epoch', type=int, default=1000)
    parser.add_argument('--p2_nas_e2', type=int, default=60)
    parser.add_argument('--p2_nas_e3', type=int, default=200)

    # general
    parser.add_argument('--dataset_batch_size', type=int, default=4096)
    
    # embedder
    parser.add_argument('--enable_pe', type=bool, default=True)
    parser.add_argument('--pe_level', type=int, default=6, help='How many levels positional embedding use')
    parser.add_argument('--pe_ii', type=bool, default=True, help='Whether include input')
    parser.add_argument('--enable_triplane', type=bool, default=True)
    parser.add_argument('--triplane_reso', type=int, default=32, help='Resolution of Planar Embedding')
    parser.add_argument('--triplane_channel', type=int, default=2, help='Channels of each entry')
    
    return parser.parse_args()



def run_phase1(args):
    # <<<<<<<<<<<<<<<<<<<<< Load Model and Build Dataset <<<<<<<<<<<<<<<<<<<<<<<
    voxel_path = args.voxel
    log_prefix = logger.create_directory(os.path.join(args.exp_name, 'phase1'))

    voxel = DV.load_voxel(voxel_path)
    resolution = voxel.shape
    # downsample voxel from high resolution to low resolution
    target_reso = args.p1_reso
    voxel_downsample = int(resolution[0] / target_reso)
    print(voxel_downsample)
    voxel = DV.downsample(voxel, voxel_downsample)

    # dataset
    train_set = VoxelDataset(voxel, sample_type='sample_base')
    dataset_info = train_set.resample()
    logger.log_metrics(os.path.join(log_prefix, 'dataset_info.txt'), dataset_info)

    # <<<<<<<<<<<<<<<<<<<<< Build or Search Network <<<<<<<<<<<<<<<<<<<<<<<<<
        
    # network config
    arch = {
        'input': 3,
        'output': (1, 'sigmoid')
    }
    
    # embedder config
    if args.enable_pe:
        arch['pe'] = {'level': args.pe_level, 'include_input': args.pe_ii}
    if args.enable_triplane:
        arch['triplane'] = {'reso': args.triplane_reso, 'channel': args.triplane_channel}

    search_space = search_space_dict[target_reso]
    mlpnas = MLPNAS(arch, 
                    train_set, 
                    search_space_nodes=search_space, 
                    mlp_training_epoch=args.p1_nas_e2, 
                    mlp_training_epoch2=args.p1_nas_e3)
    
    best_sequence = mlpnas.train()
    best_sequence = mlpnas.filter()
    print(best_sequence)

    architecture = mlpnas.search_space.decode_sequence(best_sequence)[:-1]
    arch['main'] = architecture
        
    net = StandardNet(arch)
    print('[Training Network]{}, size={}'.format(arch['main'], net.size()))

    trainer = VoxelTrainer(net, train_set, workspace=log_prefix)
    net, metrics = trainer.train(args.p1_epoch)
    metrics['architecture'] = architecture
    metrics['size'] = network_parameters(net)
    logger.log_metrics(os.path.join(log_prefix, '[VoxelTrainer]metrics.txt'), metrics)

def run_phase2(args):
    # <<<<<<<<<<<<<<<<<<<<<<< Load Model and Build Dataset <<<<<<<<<<<<<<
    voxel_path = args.voxel
    log_prefix = logger.create_directory(os.path.join(args.exp_name, 'phase2'))
    log_prefix_phase1 = os.path.join(args.exp_name, 'phase1')

    voxel = DV.load_voxel(voxel_path)
    resolution = voxel.shape
    rb = args.p1_reso
    resolution_base = [rb, rb, rb]
    rate = resolution[0] // rb

    # load low resolution model
    net1 = torch.load(os.path.join(log_prefix_phase1, '[VoxelTrainer]net.pth'))
    with torch.no_grad():
        value_base = batchify_run_network_cuda(net1, DV.voxel_centers(resolution_base))
        value_base = (value_base > 0.5).reshape(resolution_base).cpu().int()

    edge = DV.upsample(value_base.bool(), rate).bool()

    train_set = VoxelDataset(voxel, 
                             mask=edge, 
                             batch_size=args.dataset_batch_size,
                             sample_type='sample_in_mask_with_neighbour')
    
    dataset_info = train_set.resample()
    logger.log_metrics(os.path.join(log_prefix, 'dataset_info.txt'), dataset_info)

    # <<<<<<<<<<<<<<<<<<<<< Build or Search Network <<<<<<<<<<<<<<<<<<<<<<<<<

    # network config
    arch = {'input': 3, 'output': (1, 'sigmoid')}

    # embedder config
    if args.enable_pe:
        arch['pe'] = {'level': args.pe_level, 'include_input': args.pe_ii}
    if args.enable_triplane:
        arch['triplane'] = {'reso': args.triplane_reso, 'channel': args.triplane_channel}
    
    search_space = search_space_dict[resolution[0]]

    mlpnas = MLPNAS(arch, 
                    train_set, 
                    search_space_nodes=search_space, 
                    mlp_training_epoch=args.p2_nas_e2, 
                    mlp_training_epoch2=args.p2_nas_e3)
    
    best_sequence = mlpnas.train()
    best_sequence = mlpnas.filter()
    print(best_sequence)

    architecture = mlpnas.search_space.decode_sequence(best_sequence)[:-1]
    arch['main'] = architecture
    net = StandardNet(arch)
    print('[Training Network]{}, size={}'.format(arch['main'], net.size()))

    trainer = VoxelTrainer(net, train_set, workspace=log_prefix, i_scheduler=1)
    net, metrics = trainer.train(args.p2_epoch)
    metrics['architecture'] = architecture
    metrics['size'] = network_parameters(net)
    logger.log_metrics(os.path.join(log_prefix, '[VoxelTrainer]metrics.txt'), metrics)

def run(args):
    log_prefix = logger.create_directory(args.exp_name)
    logger.log_args(os.path.join(log_prefix, 'args.txt'), args)

    # run_phase1(args)
    run_phase2(args)

if __name__ == '__main__':
    args = config_parser()
    run(args)
