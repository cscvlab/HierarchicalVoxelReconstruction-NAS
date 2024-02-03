import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import open3d as o3d

sys.path.append(os.path.dirname(__file__))

import lib.metrics as M
import lib.logger as logger

from lib.net import *
from lib.dataset import VoxelDataset
import lib.DualVoxel as DV

#--------------------------------------------------------
# metrics
#--------------------------------------------------------

def compare_layer(pred: torch.Tensor, gt: torch.Tensor, i: int):
    """
    Args:
        pred: predict value
        gt: ground truth value
        i: layer index
    """
    resolution = pred.shape
    label1, label2 = pred > 0.5, gt > 0.5
    
    x = DV.voxel_area_layerI([resolution[0] for i in range(3)], i).cuda()
    x = DV.voxel_centers([resolution[0] for i in range(3)], x).cuda()

    tmp = label1 & label2
    tp = torch.sum(tmp).item()
    x_tp = x[tmp.reshape(-1)]
    tmp = ~label1 & label2
    fp = torch.sum(tmp).item()
    x_fp = x[tmp.reshape(-1)]
    tmp = label1 & ~label2
    fn = torch.sum(tmp).item()
    x_fn = x[tmp.reshape(-1)]
    tn = torch.sum(~label1 & ~label2).item()


    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}, x_tp, x_fp, x_fn

def compare_voxel(pred: torch.Tensor, gt: torch.Tensor, mode='edge'):
    """
    Args:
        pred: pred voxel
        gt: ground truth
        mode: Define how to calculate chamfer distance. choices: ['edge', 'fill'].
    """
    resolution = pred.shape
    value = pred > 0.5
    voxel = gt > 0.5
    
    metrics = M.compute_binary_metrics(value, voxel)
    x = DV.voxel_centers(resolution)
    if mode == 'edge':
        x1, x2 = x[value.reshape(-1)], x[voxel.reshape(-1)]
    elif mode == 'fill':
        value = DV.edge_detact(value)
        voxel = DV.edge_detact(voxel)
        x1, x2 = x[value.reshape(-1)], x[voxel.reshape(-1)]
    cd = M.compute_chamfer(x1.cpu().numpy(), x2.cpu().numpy())
    metrics['cd'] = cd
    
    return metrics

#--------------------------------------------------------
# log difference
#--------------------------------------------------------
def log_difference(file: str, value: torch.Tensor, voxel: torch.Tensor, type='ply'):
    resolution = voxel.shape
    if resolution[0] < 512:
        x = DV.voxel_centers(value.shape)

        pts_tp = ((voxel > 0.5) & (value > 0.5)).reshape(-1).cpu().numpy() # object point as well predict correctly
        pts_tp = x[pts_tp]
        pts_fp = ((voxel > 0.5) & (value < 0.5)).reshape(-1).cpu().numpy() # object point but predict false
        pts_fp = x[pts_fp]
        pts_fn = ((voxel < 0.5) & (value > 0.5)).reshape(-1).cpu().numpy() # non-object point but predict false
        pts_fn = x[pts_fn]
        lenTP, lenFP, lenFN = pts_tp.shape[0], pts_fp.shape[0], pts_fn.shape[0]

        pts = np.concatenate([pts_tp, pts_fp, pts_fn], axis=0)
    
    else:
        x_tp, x_fp, x_fn = [], [], []
        for i in range(resolution[0]):
            x = DV.voxel_area_layerI(resolution, i - resolution[0]//2)
            layer_value, layer_voxel = value[i], voxel[i]
            tp = (layer_value > 0.5) & (layer_voxel > 0.5).reshape(-1).cpu()
            fp = (layer_value < 0.5) & (layer_voxel > 0.5).reshape(-1).cpu()
            fn = (layer_value < 0.5) & (layer_voxel > 0.5).reshape(-1).cpu()
            x_tp.append(x[tp])
            x_fp.append(x[fp])
            x_fn.append(x[fn])

        x_tp = torch.cat(x_tp, dim=0)
        x_fp = torch.cat(x_fp, dim=0)
        x_fn = torch.cat(x_fn, dim=0)
        lenTP, lenFP, lenFN = x_tp.shape[0], x_fp.shape[0], x_fn.shape[0]
        pts = torch.cat([x_tp, x_fp, x_fn], dim=0)
        pts = DV.voxel_centers(resolution, pts)

    if type == 'ply':
        # BGR type
        color_tp = np.zeros((lenTP, 3))
        color_tp[:, 2] = 1.0
        color_fp = np.zeros((lenFP,3))
        color_fp[:, 1] = 1.0
        color_fn = np.zeros((lenFN, 3))
        color_fn[:, 0] = 1.0

        color = np.concatenate([color_tp, color_fp, color_fn], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(file, pcd, compressed=True)
    elif type == 'txt':
        prefix = file.split('.')[0]
        np.savetxt(prefix + '_tp.txt', pts_tp, fmt=['%.6f', '%.6f', '%.6f'])
        np.savetxt(prefix + '_fp.txt', pts_fp, fmt=['%.6f', '%.6f', '%.6f'])
        np.savetxt(prefix + '_fn.txt', pts_fn, fmt=['%.6f', '%.6f', '%.6f'])

#-------------------------------------------------------
# Validate
#-------------------------------------------------------
@torch.no_grad()
def validate(net: nn.Module, voxel: torch.Tensor, mask: torch.Tensor = None):
    """
    Args:
        net(nn.Module): network
        voxel(torch.Tensor): ground truth
    """
    # assert next(net.parameters()).device == 'cuda:0'
    resolution = voxel.shape
    size = resolution[0] * resolution[1] * resolution[2]
    
    if mask == None:
        value = batchify_run_network(net, DV.voxel_centers(resolution).cuda()).cpu()
        loss = F.binary_cross_entropy(value, voxel.float().view(-1, 1)).item()
    else:
        value = torch.zeros(size, 1)
        value[mask.view(-1)] = batchify_run_network(net, DV.voxel_centers(resolution)[mask.view(-1)].cuda()).cpu()
        loss = F.binary_cross_entropy(value[mask.view(-1, 1)], voxel.view(-1, 1)[mask.view(-1, 1)].float())
    
    value = value.reshape(resolution)
    metrics = compare_voxel(value, voxel)
    metrics['loss'] = loss
    metrics['size'] = net.size()
    
    return metrics, value
    
#--------------------------------------------------------
# Trainer
#--------------------------------------------------------
class BaseTrainer:
    def __init__(self, 
                 model : StandardNet,
                 train_set,
                 **kwargs
                 ):
        # parameters
        self.optimizer_type = kwargs.get('optimizer_type', 'adam')
        self.optimizer_lr_init = kwargs.get('optimizer_lr_init', 1e-3)
        self.optimizer_decay = kwargs.get('optimizer_decay', 1e-5)
        self.optimizer_momentum = kwargs.get('optimizer_momentum', 0)

        self.scheduler_type = kwargs.get('scheduler_type', 'plateau')
        self.scheduler_verbose = kwargs.get('scheduler_verbose', True)
        self.scheduler_expLR_gamma = kwargs.get('scheduler_expLR_gamma', 0.65)
        self.scheduler_plateau_factor = kwargs.get('scheduler_plateau_factor', 0.7)
        self.scheduler_plateau_mode = kwargs.get('scheduler_plateau_mode', 'min')
        
        # network
        self.net = model
        self.optimizer, self.scheduler = self._create_optimizer_scheduler()
        
        # Dataset
        self.train_set = train_set
        
        # Record
        self.history = pd.DataFrame()
        
        # output configs
        self.workspace = kwargs.get('workspace', '')
        self.tensorboard_output = kwargs.get('tensorboard_output', '')
        self.filename_prefix = kwargs.get('filename_prefix', '[{}]'.format(self.__class__.__name__))

        # frequency of logging
        self.i_resample = kwargs.get('i_resample', -1)
        self.i_recon = kwargs.get('i_recon', 10)
        self.i_figure = kwargs.get('i_figure', 10)
        self.i_scheduler = kwargs.get('i_scheduler', -1)
        self.i_checkpoint = kwargs.get('i_checkpoint', 10)

        # items which need to log
        self.tensorboard_items = kwargs.get('tensorboard_items', [])
        self.figure_items = kwargs.get('figure_items', [])

        # other
        self.enable_early_end = kwargs.get('enable_early_end', True)

        self.writer = None if self.tensorboard_output == '' else SummaryWriter(self.tensorboard_output)
        
    def _create_optimizer_scheduler(self):
        # set optimizer
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.net.parameters(), 
                lr=self.optimizer_lr_init, 
                momentum=self.optimizer_momentum, 
                weight_decay=self.optimizer_decay)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.optimizer_lr_init, 
                weight_decay=self.optimizer_decay)
        else:
            raise ValueError('[MLP Generator]Invalid optimizer')
        
        # Set scheduler
        if self.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min', 
                factor=self.scheduler_plateau_factor, 
                min_lr=1e-7, 
                verbose=self.scheduler_verbose)
        elif self.scheduler_type == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, 
                gamma=self.scheduler_expLR_gamma, 
                verbose=self.scheduler_verbose)
        else:
            raise ValueError('Invalid scheduler setting: {}'.format(self.scheduler))
        
        return optimizer, scheduler
    
    def _compute_loss(self, pred, y):
        """
        Should be implemented

        Args:
            pred (_type_): predict value
            y (_type_): ground truth value

        Returns:
            Tuple: {'loss': torch.Tensor, ...}
            
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('Compute Loss haven\'t been implemented. ')
    
    def _training_step(self, data):
        x, y = data
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        
        pred = self.net(x)
        loss = self._compute_loss(pred, y)['loss']
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _training(self):
        self.net = self.net.train()
        for data in self.train_set:
            loss = self._training_step(data)
        return loss

    def _update_scheduler(self, i, value):
        if self.i_scheduler <= 0:
            return
        if i % self.i_scheduler != 0:
            return
        if i == 0:
            return
        self.scheduler.step(value)

    def _validate(self, i=-1):
        pass

    def _print_to_tensorboard(self, i: int, metrics: dict, key=None):
        if not self.writer is None:
            if key == None:
                for key, value in metrics.items():
                    if key in self.tensorboard_items:
                        self.writer.add_scalar(key, value, i)
            else:
                self.writer.add_scalar(key, metrics[key], i)

    def _print_to_figure(self, i=-1):
        if self.workspace == '':
            return
        
        if i != -1 and i % self.i_figure != 0:
            return
        
        for key in self.figure_items:
            plt.plot(range(len(self.history)), self.history[key])
            plt.title(key)
            plt.savefig(os.path.join(self.workspace, self.filename_prefix+'{}.jpg'.format(key)))
            plt.cla()

    def _save_checkpoint(self, i=-1):
        if self.workspace == '':
            return
        
        if i != -1 and i % self.i_checkpoint != 0:
            return
        
        torch.save(self.net, os.path.join(self.workspace, self.filename_prefix+'net.pth'))

    def train(self, epoch, **kwargs):
        """
        Returns:
            Trained Net
            Metrics of Final Net
        """
        raise NotImplementedError('Method \"train\" is not implemented.')
    
    def fast_train(self, epoch, **kwargs):
        raise NotImplementedError('Method \"fasttrain\" is not implemented.')


class VoxelTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_set: VoxelDataset, **kwargs):
        super().__init__(model, train_set, **kwargs)
        self.resolution = self.train_set.resolution
        
        self.figure_items = ['loss', 'cd', 'iou']
        self.tensorboard_items = ['loss', 'cd', 'iou', 'tp', 'fp', 'fn', 'accuracy', 'precision', 'recall']
    
        if self.workspace != '':
            with open(os.path.join(self.workspace, self.filename_prefix + 'NetArchitecture.txt'), 'w+') as f:
                f.write(str(self.net))
                f.write('size={}'.format(self.net.size()))
    
    def _compute_loss(self, pred, y):
        return {'loss': F.binary_cross_entropy(pred, y)}
    
    def _training_step(self, data):
        x, y = data
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        
        pred = self.net(x)
        loss = self._compute_loss(pred, y)['loss']
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validate(self, i):
        metrics, value = validate(self.net, self.train_set.voxel, self.train_set.mask)
        
        if self.workspace == '':
            return metrics
        
        if i != -1 and i % self.i_recon != 0:
            return metrics
        
        logger.log_metrics(os.path.join(self.workspace, self.filename_prefix+'metrics.txt'), metrics)
        log_difference(os.path.join(self.workspace, self.filename_prefix+'pts_color.ply'), value, self.train_set.voxel)
        np.savez_compressed(os.path.join(self.workspace, self.filename_prefix+'pred.npz'), value.numpy())
        
        return metrics

    def train(self, epoch, **kwargs):
        # start training
        self.net = self.net.cuda()
        task = tqdm(range(epoch), colour='green')
        for i in task:
            # training step
            self._training()   
            # validate step
            metrics = self._validate(i)
            # update learning rate
            self._update_scheduler(i, metrics['loss'])
            
            # log
            self.history = self.history.append(metrics, ignore_index=True)
            self._print_to_tensorboard(i, metrics)
            self._print_to_figure(i)
            self._save_checkpoint(i)
            
            task.set_description('loss: {:.6f}, acc: {:.6f}, iou: {:.6f}, cd*1000: {:.6f}'.format(
                metrics['loss'], metrics['accuracy'], metrics['iou'], metrics['cd']*1000))
            
            if self.enable_early_end:
                if metrics['fp'] + metrics['fn'] == 0:
                    break
            
        # end training
        metrics = self._validate(-1)

        self.net = self.net.cpu()
        self._print_to_figure(-1)
        self._save_checkpoint(-1)
            
        if self.workspace != '':
            self.history.to_csv(os.path.join(self.workspace, self.filename_prefix+'history.csv'))
        
        return self.net, metrics

    def fast_train(self, epoch):
        self.net = self.net.cuda()
        task = tqdm(range(epoch), colour='green')
        for i in task:
            loss = self._training()
            self._update_scheduler(i, loss)

        metrics = self._validate(-1)
        print('loss: {:.6f}, acc: {:.6f}, iou: {:.6f}, cd*1000: {:.6f}'.format(
                metrics['loss'], metrics['accuracy'], metrics['iou'], metrics['cd']*1000))
        self.net = self.net.cpu()
        return self.net, metrics

search_space_dict = {
    32: [8, 16, 24, 32, 40, 48],
    64: [8, 16, 32, 40, 48, 64],
    128: [16, 32, 40, 48, 64],
    256: [16, 32, 48, 64, 128],
    512: [32, 48, 64, 96, 128]
}

smallSearchSpace = [8, 16, 32, 40, 48, 64]
bigSearchSpace = [16, 32, 48, 64, 96, 128]

class MLPSearchSpace:
    def __init__(self, 
                 nodes = smallSearchSpace, 
                 acts = ['swish', 'relu', 'elu']):
        self.nodes = nodes
        self.activation = acts
        self.dict = [(node, act) for node in self.nodes for act in self.activation]
        self.dict.append((-1, ''))  # end signal
        
    def encode_sequence(self, architecture_list):
        sequence = []
        for arch in architecture_list:
            sequence.append(self.dict.index(arch))
        return sequence
            
    def decode_sequence(self, sequence): # input: 0, ..., num_ops
        architecture_list = []
        for idx in sequence:    # 0, ..., len-1
            architecture_list.append(self.dict[idx])  
        return architecture_list

    def _num_ops(self):
        return len(self.dict)

class MLPNAS(BaseTrainer):    
    def __init__(self,
                 net_config,
                 train_set,
                 trainer_class = VoxelTrainer,
                 **kwargs):
        # basic configs
        self.sampling_epoch    : int = kwargs.get('nas_sampling_epoch', 5)
        self.samples_per_epoch : int = kwargs.get('nas_samples_per_epoch', 6)
        self.mlp_training_epoch: int = kwargs.get('mlp_training_epoch', 40)
        self.mlp_training_epoch2: int = kwargs.get('mlp_training_epoch2', 100)

        # searched mlp configs
        self.max_num_layers    : int = kwargs.get('max_num_layers', 6)
        self.search_space_nodes: list = kwargs.get('search_space_nodes', smallSearchSpace)
        self.search_space_acts : list = kwargs.get('search_space_acts', ['swish', 'relu', 'elu'])

        # loss configs
        self.baseline_acc      : float = kwargs.get('baseline_acc', 0.98)
        self.baseline_size     : int = kwargs.get('baseline_size', 7553)
        self.max_size          : int = kwargs.get('max_size', 21121)

        # select configs
        self.threshold_acc     : float = kwargs.get('threshold_acc', 5e-5)
        self.top_n             : int = kwargs.get('top_n', 10)

        # supervise configs
        self.supervise_items   : list = kwargs.get('supervise_items', ['size', 'reward', 'accuracy', 'iou', 'cd'])

        self.search_space = MLPSearchSpace(self.search_space_nodes, self.search_space_acts)
        controller = Controller(self.search_space._num_ops(), self.max_num_layers)
        super().__init__(controller, train_set, **kwargs)
        
        # other net component
        self.net_config = net_config

        # tranier
        self.trainer_class = trainer_class
        
        # production
        self.search_history = pd.DataFrame()
        self.filter_history = pd.DataFrame()

        # override
        self.i_checkpoint = 1
    
    def train_sequence(self, sequence, epoch, fast_train=True, **kwargs):
        """
        Returns:
            dict{net: nn.Module, 
                 net_config: dict,
                 size: int,
                 history: pd.DataFrame(),
                 metrics: dict
                 }
        """
        architecture = self.search_space.decode_sequence(sequence)
        config = self.net_config
        config['main'] = architecture[:-1]
        print(config)
        
        net = StandardNet(config)
        print('[Training Network]{}, size={}'.format(config['main'], net.size()))

        mlp_trainer = self.trainer_class(net, self.train_set, **kwargs)
            
        # Don't need validate
        if fast_train:
            net, metrics = mlp_trainer.fast_train(epoch)
        else:
            net, metrics = mlp_trainer.train(epoch)

        return {
            'net': net,
            'net_config': config,
            'sequence': sequence,
            'size': net.size(),
            'history': mlp_trainer.history,
            'metrics': metrics
        }

    def _training(self):
        # loss 
        log_probs, rewards = [], []
        for i in range(self.samples_per_epoch):
            sequence, entropy, log_prob = self.net.net_sample() # controller loss
            output = self.train_sequence(sequence, self.mlp_training_epoch)
            # log
            performance = output['metrics']
            performance['sequence'] = sequence
            performance['architecture'] = output['net_config']['main']
            performance['size'] = output['size']

            acc = performance['accuracy']
            size = performance['size']
            # calculate controller reward & loss
            reward = (acc - self.baseline_acc) + (self.baseline_size - size) / self.max_size
            performance['reward'] = reward
            
            self.search_history = self.search_history.append(performance, ignore_index=True)
            self._print_search_history()
            print('acc={:.6f}, iou={:.6f}, cd={:.6f}'.format(performance['accuracy'], performance['iou'], performance['cd']))
            log_probs.append(log_prob)
            rewards.append(reward)

        # calculate loss
        log_probs = torch.stack(log_probs)  # [samples_per_epoch]
        rewards = torch.Tensor(rewards).cuda()    # [samples_per_epoch]
        loss = torch.sum(log_probs * rewards) / self.samples_per_epoch

        # update controller
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item()

    def train(self):
        self.net = self.net.cuda()
        # train controller
        # first sample
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<First Round<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        for epoch in range(self.sampling_epoch):
            print('Searching epochs: {}/{}'.format(epoch+1, self.sampling_epoch))
            loss = self._training()
            self.history = self.history.append(
                {'loss' : loss}, 
                ignore_index=True)
            self._print_history()
            self._print_search_history()
            self._save_checkpoint(epoch)
        # select best sequence 
        best_sequence, select = self.best_sequence()

        self.net = self.net.cpu()
        return best_sequence
    
    def filter(self):
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Filter<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # second sample
        if self.top_n > 0:
            top_n_sequence = self.top_n_sequence(self.top_n)
            for sequence in top_n_sequence:
                output = self.train_sequence(sequence, self.mlp_training_epoch2)
                # log
                performance = output['metrics']
                performance['architecture'] = output['net_config']['main']
                performance['size'] = output['size']
                performance['sequence'] = output['sequence']
                self.filter_history = self.filter_history.append(performance, ignore_index = True)
            
            best_sequence, select = self.best_sequence(self.filter_history)
            self._print_filter_history()

        return best_sequence
          
    def _print_search_history(self, i=0):
        if self.workspace == '':
            return  
        
        white_list = self.supervise_items
        for item in white_list:
            plt.plot(range(len(self.search_history)), self.search_history[item])
            plt.title(item)
            plt.savefig(os.path.join(self.workspace, self.filename_prefix+'{}.jpg'.format(item)))
            plt.cla()
        self.search_history.to_csv(os.path.join(self.workspace, self.filename_prefix+'search_history.csv'))
    
    def _print_history(self, i=0):
        if self.workspace == '':
            return
        
        # plot loss
        plt.plot(range(len(self.history)), self.history['loss'])
        plt.title('loss')
        plt.savefig(os.path.join(self.workspace, self.filename_prefix+'loss.jpg'))
        plt.cla()

        self.history.to_csv(os.path.join(self.workspace, self.filename_prefix+'history.csv'))

    def _print_filter_history(self, i=0):
        if self.workspace != '':
            self.filter_history.to_csv(os.path.join(self.workspace, self.filename_prefix+'filter_history.csv'))
    
    def best_sequence(self, search_history = None):
        """
        select best mlp from sampled architectures
        """
        search_history = self.search_history if search_history is None else search_history
        sorted_idx = np.argsort(-np.asarray(search_history['accuracy']))
        if self.threshold_acc > 0:  # select a appropriate network
            select = sorted_idx[0]
            min_size = search_history['size'][sorted_idx[0]]
            best_acc = search_history['accuracy'][sorted_idx[0]]
            for idx in sorted_idx:
                if best_acc - search_history['accuracy'][idx] > self.threshold_acc:
                    break
                if search_history['size'][idx] < min_size:
                    min_size = search_history['size'][idx]
                    select = idx
            sequence = search_history['sequence'][select]
        else:   # select the network with highest accuracy
            sequence = search_history['sequence'][sorted_idx[0]]
        return sequence, select    
    
    def top_n_sequence(self, n=10):
        search_history = self.search_history.sort_values(['accuracy', 'size'], ascending=[False, True])
        return search_history['sequence'][:10].to_list()
            

   
   
 