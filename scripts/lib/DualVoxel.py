import torch
import torch.nn as nn
import torch.nn.functional as F

kernel1 = [
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
]

kernel2 = [
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
]

kernel3 = [
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
]

class VoxelEdgeFilter(nn.Module):
    def __init__(self, type=3):
        # 0 < inner_conv < 27 -> edge voxel
        # edge act & c == 1 -> inner edge voxel
        # edge act & c == 0 -> outer edge voxel
        super(VoxelEdgeFilter, self).__init__()
        self._type = type

        if type == 1:
            self.kernel = torch.Tensor(kernel1)
        elif type == 2:
            self.kernel = torch.Tensor(kernel2)
        else:
            self.kernel = torch.Tensor(kernel3)

        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        
    def forward(self, grid: torch.Tensor):
        device = grid.device
        self.kernel = self.kernel.to(device)
        
        grid = grid.unsqueeze(0).unsqueeze(0)
        output = F.conv3d(grid, self.kernel, padding=1) # {0, 1, ..., 27}
        grid = grid.squeeze(0).squeeze(0)
        output = output.squeeze(0).squeeze(0)
        
        grid = grid.int()   # {0, 1}
        output = output.int()

        if self._type == 1:
            output[output == 6] = 0
            output[output > 0] = 1
        elif self._type == 2:
            output[output == 18] = 0
            output[output > 0] = 1
        else:
            output[output == 27] = 0   # {0, 1, ..., 26}
            output[output > 0] = 1   # {0, 1}

        ch0 = output * grid # act inner ->  1
        ch1 = -(output * (1 - grid)) # act outer -> -1
        ch = ch0 + ch1
        return ch

def voxel_size(resolution: list):
    return torch.FloatTensor([
        2.0 / resolution[0], 
        2.0 / resolution[1], 
        2.0 / resolution[2]
    ])
    
def voxel_all(resolution):
    xc = torch.arange(-resolution[0]/2, resolution[0]/2).int()
    yc = torch.arange(-resolution[1]/2, resolution[1]/2).int()
    zc = torch.arange(-resolution[2]/2, resolution[2]/2).int()
    xc, yc, zc = torch.meshgrid(xc, yc, zc)
    x = torch.cat([xc.reshape(-1,1), yc.reshape(-1,1), zc.reshape(-1,1)], dim=1)
    return x

def voxel_area_layerI(resolution, idx):
    xc = torch.ones(resolution[1] * resolution[2]).int() * idx
    yc = torch.arange(-resolution[1]/2, resolution[1]/2).int()
    zc = torch.arange(-resolution[2]/2, resolution[2]/2).int()
    yc, zc = torch.meshgrid(yc, zc)
    x = torch.cat([xc.reshape(-1,1), yc.reshape(-1,1), zc.reshape(-1,1)], dim=1)
    return x

def voxel_area_patch(resolution: list, patch: list, patch_size=[64, 64, 64]):
    """
    Args:
        resolution
    """
    xc = torch.arange(0, patch_size[0]).int()
    yc = torch.arange(0, patch_size[1]).int()
    zc = torch.arange(0, patch_size[2]).int()
    xc, yc, zc = torch.meshgrid(xc, yc, zc)
    x = torch.cat([xc.reshape(-1,1), yc.reshape(-1,1), zc.reshape(-1,1)], dim=1)
    x -= torch.Tensor([[resolution[0]//2, resolution[1]//2, resolution[2]//2]]).repeat(x.shape[0], 1).int()
    x += torch.Tensor([patch[0]*patch_size[0], patch[1]*patch_size[1], patch[2]*patch_size[2]]).repeat(x.shape[0], 1).int()
    return x

def patches(resolution: list, patch_size=[64, 64, 64]):
    return [resolution[0]// patch_size[0], resolution[1]//patch_size[1], resolution[2]//patch_size[2]]

def patch_slice(patch, patch_size=[64, 64, 64]):
    x, y, z = patch[0], patch[1], patch[2]
    xdim, ydim, zdim = patch_size[0], patch_size[1], patch_size[2]
    return slice(x*xdim, (x+1)*xdim), slice(y*ydim, (y+1)*ydim), slice(z*zdim, (z+1)*zdim)

def valueOf(grid, x: torch.IntTensor, padding=False):
    resolution = torch.IntTensor([grid.shape[i] for i in range(3)]).to(x.device)
    idx_coordinate = x + torch.Tensor(resolution // 2).expand_as(x)
    if padding:
        idx_coordinate %= resolution.expand_as(x)
    value = grid[idx_coordinate[:, 0].long(), 
                 idx_coordinate[:, 1].long(),
                 idx_coordinate[:, 2].long()]
    return value

def voxel_centers(resolution: list, x: torch.IntTensor = None):
    """
    Given Voxel Integer Coordinate, Return its Center Coordinate
    Args:
        resolution (list): [int, int, int]\n
        x (torch.IntTensor, optional): voxel integer coordinate. Shape: [N, 3]. Defaults to None.
    Returns:
        torch.FloatTensor: center coordinate. Shape: [N, 3]
    """
    x = voxel_all(resolution) if x == None else x
    vs = voxel_size(resolution).to(x.device)
    ret = x + 0.5
    ret *= vs
    return ret

def voxel_corners(resolution, x:torch.IntTensor = None):
    """
    Given Voxel Integer Coordinate, Return its Corner Coordinate
    Args:
        resolution (list): [int, int, int]\n
        x (torch.IntTensor, optional): voxel integer coordinate. Shape: [N, 3]. Defaults to None.
    Returns:
        torch.FloatTensor: center coordinate. Shape: [N, 8, 3]
    """
    x = voxel_all(resolution) if x == None else x
    
    x = x.unsqueeze(1)  # [batch_size, 1, 3]
    vs = voxel_size(resolution).to(x.device)
    base = x * vs
    ret = []
    
    xc = torch.Tensor([0.0, vs[0]])
    yc = torch.Tensor([0.0, vs[1]])
    zc = torch.Tensor([0.0, vs[2]])
    xc, yc, zc = torch.meshgrid(xc, yc, zc)
    xc, yc, zc = xc.reshape(-1, 1), yc.reshape(-1, 1), zc.reshape(-1, 1)
    offset = torch.cat([xc, yc, zc], dim=-1)
    for o in offset:
        tmp = base + o.expand_as(base).to(x.device)
        ret.append(tmp)
        
    ret = torch.cat(ret, dim=1)
    
    return ret

def neighbour_voxel(x: torch.IntTensor):
    ret = []
    
    neighbour = torch.IntTensor([
        [-1, -1, -1],[-1, -1,  0],[-1, -1,  1],
        [-1,  0, -1],[-1,  0,  0],[-1,  0,  1],
        [-1,  1, -1],[-1,  1,  0],[-1,  1,  1],
        
        [ 0, -1, -1],[ 0, -1,  0],[ 0, -1,  1],
        [ 0,  0, -1],[ 0,  0,  0],[ 0,  0,  1],
        [ 0,  1, -1],[ 0,  1,  0],[ 0,  1,  1],
        
        [ 1, -1, -1],[ 1, -1,  0],[ 1, -1,  1],
        [ 1,  0, -1],[ 1,  0,  0],[ 1,  0,  1],
        [ 1,  1, -1],[ 1,  1,  0],[ 1,  1,  1],
    ])
    
    for offset in neighbour:
        ret.append(x + offset.expand_as(x))
        
    ret = torch.cat(ret, dim=0)
    return ret
   
def pts_in_voxel(resolution, x:torch.FloatTensor):
    """
    Given Points Coordinate in [-1, 1], Return its Voxel Coordinate   \n
    Args:
        resolution (list): [int, int, int]  \n
        x (torch.FloatTensor, optional): points float coordinate. Shape: [N, 3].   \n
    Returns:
        torch.IntTensor: voxel coordinate. Shape: [N, 3]
    """
    vs = voxel_size(resolution).expand_as(x).to(x.device)
    ret = x / vs
    ret = torch.floor(ret).int()
    return ret

def classify_voxels(grid: torch.IntTensor):
    m_grid = grid.float()   # [N, N, N] 0 -> outer, 1 -> inner
    filter = VoxelEdgeFilter()
    output = filter(m_grid) # [N, N, N] 0 -> other space, 1 -> inner edge, -1 -> outer edge
    
    label = grid + output   # -1 -> outer edge, 0 -> outer, 1 -> inner, 2 -> inner edge
    return label

def edge_detact(grid: torch.IntTensor, type=1):
    m_grid = grid.float()
    filter = VoxelEdgeFilter(type)
    output = filter(m_grid)
    return output

def extract_coordinate(label: torch.IntTensor, value: int):
    """
    value:
        -1 -> outer edge    \n
         0 -> outer         \n
         1 -> inner         \n
         2 -> innder edge   \n
    """
    resolution = label.shape
    c = torch.argwhere(label == value)
    c -= torch.tensor(resolution).expand_as(c).to(c.device) // 2
    return c

def load_voxel(path):
    import numpy as np
    return torch.from_numpy(np.load(path)['arr_0'])

def downsample(grid: torch.IntTensor, srate: int):
    """
    grid: high resolution voxel grid\n
    srate: shrink rate
    """
    ret = grid.unsqueeze(0).unsqueeze(0).float()
    ret = F.max_pool3d(ret, kernel_size=srate)
    ret = ret.squeeze(0).squeeze(0).int()
    return ret

def upsample(grid: torch.IntTensor, srate: int):
    """
    grid: low resolution voxel grid\n
    srate: enlarge rate
    """
    
    ret = grid.unsqueeze(0).unsqueeze(0).float()   # [1, 1, R, R, R]
    ret: torch.Tensor = torch.nn.functional.interpolate(ret, scale_factor=srate, mode='nearest')
    ret = ret.squeeze(0).squeeze(0).int()     # [R, R, R]
    return ret
                
def export(path: str, voxel):
    resolution = voxel.shape
    pts = voxel_centers(resolution)
    pts = pts[voxel.reshape(-1).bool()]
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(path, pcd, compressed=True)
