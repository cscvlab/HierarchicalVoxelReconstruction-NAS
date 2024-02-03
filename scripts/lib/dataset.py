import os
import sys
import numpy as np
import math
from typing import Union, List
import torch

sys.path.append('./scripts')
import lib.DualVoxel as DV
import lib.logger as logger

class BaseDataset:
    def __init__(self, batch_size = 4096, shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = 0

        self.X = torch.zeros(0, 3)
        self.y = torch.zeros(0, 1)
    
    def _shuffle(self):
        if self.X.shape[0] <= 0:
            return
        if self.shuffle:
            p = np.random.permutation(self.X.shape[0])
            p = torch.LongTensor(p)
            self.X = self.X[p]
            self.y = self.y[p]

    def __len__(self):
        return int(math.ceil(self.X.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        i = index * self.batch_size
        X = self.X[i: i+self.batch_size] if i + self.batch_size < len(self.X) else self.X[i:]
        y = self.y[i: i+self.batch_size] if i + self.batch_size < len(self.y) else self.y[i:]
        return X, y

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.X.shape[0]:
            i = self.count
            X = self.X[i: i+self.batch_size] if i + self.batch_size < len(self.X) else self.X[i:]
            y = self.y[i: i+self.batch_size] if i + self.batch_size < len(self.y) else self.y[i:]
            self.count += self.batch_size
            return X, y
        raise StopIteration()

    def resample(self):
        raise NotImplementedError()

class VoxelDataset(BaseDataset):
    voxel: Union[torch.Tensor, None] = None
    mask: Union[torch.Tensor, None] = None
    
    def __init__(self, voxel, mask = None, 
                 batch_size=4096, shuffle=True, **kwargs):
        super().__init__(batch_size, shuffle)
        
        # origin data
        self.voxel = voxel
        self.mask = mask
        
        # voxel infomation
        self.resolution = self.voxel.shape
        self.voxel_size = 2.0 / self.resolution[0]
        self.voxel_num = self.resolution[0] * self.resolution[1] * self.resolution[2]
        self.coord_min = [-res//2 for res in self.resolution]
        self.coord_max = [res//2-1 for res in self.resolution]
        
        # sample config
        self.downsample = kwargs.get('downsample', 1)
        self.sample_type = kwargs.get('sample_type', 'sample_base')
        
        # sample first time
        self.resample()
        
    def from_file(path, target_reso=-1, **kwargs):
        voxel = DV.load_voxel(path)
        print('Loaded Voxel: {}'.format(path))
        print('Resolution: {}, Target Resolution: {}'.format(voxel.shape[0], target_reso))

        if target_reso != -1:
            voxel = DV.downsample(voxel, int(voxel.shape[0] / target_reso))
        return VoxelDataset(voxel, **kwargs)

    def print_info(self):
        print('[VoxelDataset]Resolution: ', self.resolution)
        print('[VoxelDataset]Size of Voxel', self.voxel_size)
        print('[VoxelDataset]Number of Object: [{}/{}] = {}'.format(
            torch.sum(self.voxel), self.voxel_num, torch.sum(self.voxel)/self.voxel_num
        ))
        print('[VoxelDataset]Mask enabled: ', self.mask != None)
        
    def resample(self):
        if self.sample_type == 'sample_base':
            return self._sample_base()
        elif self.sample_type == 'sample_in_mask':
            return self._sample_in_mask()
        elif self.sample_type == 'sample_in_mask_with_neighbour':
            return self._sample_in_mask_with_neighbour()
        else:
            raise ValueError("Invalid Sample Type: ", self.sample_type)
        
    def _sample_base(self):
        """
        Sample accroding to support voxel and unsupport voxel
        """
        args = {}
        # record
        args['downsample'] = self.downsample
        args['type'] = 'sample_base'
        args['resolution'] = self.resolution
        args['voxel_size'] = self.voxel_size
        args['voxel_num'] = self.voxel_num
        args['object_num'] = torch.sum(self.voxel)
        args['voxel_rate'] = torch.sum(self.voxel)/self.voxel_num
        
        label = DV.classify_voxels(self.voxel).reshape(-1)
        
        value = self.voxel.reshape(-1)
        X = DV.voxel_centers(self.resolution)

        # extract support voxels
        support = (label == 2) | (label == -1)
        X_support, y_support = X[support], value[support]

        #record
        args['support_num'] = X_support.shape[0]
        args['support_posi'] = torch.sum(y_support>0.5)
        args['support_nega'] = torch.sum(y_support<0.5)

        # extract unsupport voxels
        unsupport = (label == 0) | (label == 1)
        X_unsupport, y_unsupport = X[unsupport], value[unsupport]

        # downsample
        if self.downsample <= 1:
            N_downsample = int(X_unsupport.shape[0] * self.downsample)
        else:
            N_downsample = int(self.downsample)

        p = torch.LongTensor(np.random.choice(np.arange(X_unsupport.shape[0]), N_downsample, replace=False))
        X_unsupport, y_unsupport = X_unsupport[p], y_unsupport[p]

        args['unsupport_num'] = N_downsample
        args['unsupport_posi'] = torch.sum(y_unsupport>0.5)
        args['unsupport_nega'] = torch.sum(y_unsupport<0.5)
        
        # keep the same magnitude
        if X_support.shape[0] > 0:
            scale = int(math.ceil(X_unsupport.shape[0] / X_support.shape[0]))
            X_support, y_support = X_support.repeat(scale, 1), y_support.repeat(scale)
            args['scale'] = scale
        
        self.X = torch.cat([X_support, X_unsupport], dim=0)
        self.y = torch.cat([y_support, y_unsupport], dim=0).float().reshape(-1, 1)
        args['support_num_scaled'] = X_support.shape[0]

        # shuffle
        self._shuffle()
        args['shuffle'] = self.shuffle

        return args
    
    def _sample_in_mask(self):
        args = {}
        args['type'] = 'sample_in_mask'
        args['resolution'] = self.resolution
        args['voxel_size'] = self.voxel_size
        args['voxel_num'] = self.voxel_num
        args['object_num'] = torch.sum(self.voxel)
        args['voxel_rate'] = torch.sum(self.voxel)/self.voxel_num
        
        mask = self.mask.reshape(-1)
        value = self.voxel.reshape(-1)
        X = DV.voxel_centers(self.resolution)
        
        self.X, self.y = X[mask], value[mask].float().reshape(-1, 1)
        args['samples_num'] = self.X.shape[0]
        args['samples_posi'] = torch.sum(self.y)
        args['samples_nega'] = torch.sum(1-self.y)
        
        self._shuffle()
        args['shuffle'] = self.shuffle
        return args
        
    def _sample_in_mask_with_neighbour(self):
        args = {}
        args['type'] = 'sample_in_mask_with_neighbour'
        args['resolution'] = self.resolution
        args['voxel_size'] = self.voxel_size
        args['voxel_num'] = self.voxel_num
        args['object_num'] = torch.sum(self.voxel)
        args['voxel_rate'] = torch.sum(self.voxel)/self.voxel_num
        
        value = self.voxel.view(-1)
        X = DV.voxel_centers(self.resolution)
        
        kernel = self.mask & (self.voxel>0.5)
        support = DV.classify_voxels(kernel)
        support = (support == 2) | (support == -1) | kernel
        unsupport = self.mask & ~support
        support, unsupport = support.reshape(-1), unsupport.reshape(-1)
        
        args['kernel_num'] = torch.sum(kernel).item()
        args['support_sum'] = torch.sum(support).item()
        args['unsupport_sum'] = torch.sum(unsupport).item()
        
        X_support, y_support = X[support], value[support]
        X_unsupport, y_unsupport = X[unsupport], value[unsupport]
    
        # keep the same magnitude
        scale = int(math.ceil(X_unsupport.shape[0] / X_support.shape[0]))
        X_support = X_support.repeat(scale, 1)
        y_support = y_support.repeat(scale)
        args['scale'] = scale
        args['support_num_scaled'] = X_support.shape[0]
        
        self.X = torch.cat([X_support, X_unsupport], dim=0)
        self.y = torch.cat([y_support, y_unsupport], dim=0).float().reshape(-1, 1)
        
        # shuffle
        self._shuffle()
        args['shuffle'] = self.shuffle
        return args
    

