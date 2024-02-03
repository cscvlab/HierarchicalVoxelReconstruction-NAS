import os
import time
import sys
import shutil
import numpy as np
import torch
from tqdm import tqdm

import DualVoxel
import trimesh

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--mesh_dir', type=str, help='Directory of origin mesh ')
    parser.add_argument('--mesh_out', type=str, help='Directory of output normalized mesh')
    parser.add_argument('--voxel_out', type=str, help='Directory of output voxel')
    parser.add_argument('--name', type=str, help='Specify mesh without suffix of .obj')
    parser.add_argument('--suffix', type=str, default='.obj')

    parser.add_argument('--resolution', type=int, choices=[32, 64, 128, 256, 512], help='Target Resolution of voxel')
    
    return parser.parse_args()

def normalize(V, F):
    # translation
    V_max = np.max(V, axis=0)
    V_min = np.min(V, axis=0)
    V_center = (V_max + V_min) / 2
    V = V - V_center
    
    # scale
    max_dist = np.sqrt(np.max(np.sum(V**2, axis=-1)))
    V_scale = 1. / max_dist
    V *= V_scale
    
    return V, F, {'origin_max': V_max, 'origin_min': V_min, 'origin_center': V_center, 'origin_max dist': max_dist}

def normalize_mesh(mesh: trimesh.Trimesh, rotate=[0, 1, 2]):

    if 'geometry' in mesh.__dict__.keys(): 
        V = []
        F = []
        offset = 0
        for key, value in mesh.geometry.items():
            V.append(value.vertices)
            F.append(value.faces + offset)
            offset += value.vertices.shape[0]
        
        V = np.concatenate(V, axis=0)
        F = np.concatenate(F, axis=0)
        
    else:
        V = mesh.vertices
        F = mesh.faces
        
    V, F, info = normalize(V, F)
    
    out = trimesh.Trimesh(V[:, rotate], F)
    return out


if __name__ == '__main__':

    args = config_parser()

    if args.mesh_out is not None:
        if not os.path.exists(args.mesh_out):
            os.makedirs(args.mesh_out)

    if not os.path.exists(args.voxel_out):
        os.makedirs(args.voxel_out)

    if args.name is not None:
        input_file = os.path.join(args.mesh_dir, args.name + args.suffix)
        mesh = trimesh.load(input_file)
        # normalize
        mesh = normalize_mesh(mesh)

        if args.mesh_out is not None:
            mesh.export(os.path.join(args.mesh_out, args.name + '.obj'))
        
        # voxelize
        triangles = mesh.triangles
        r = args.resolution
        grid = DualVoxel.voxelizeTriangle(triangles, [r, r, r])
        grid = np.asarray(grid, dtype=np.int8).reshape(r, r, r)

        np.savez_compressed(os.path.join(args.voxel_out, args.name + '.npz'), grid)
    else:
        task = tqdm(os.listdir(args.mesh_dir))
        for file_name in task:
            name = file_name.split('.')[0]
            task.set_description('Processing: {}'.format(name))

            mesh = trimesh.load(os.path.join(args.mesh_dir, file_name))

            mesh = normalize_mesh(mesh)

            if args.mesh_out is not None:
                mesh.export(os.path.join(args.mesh_out, name + '.obj'))

            triangles = mesh.triangles
            r = args.resolution
            grid = DualVoxel.voxelizeTriangle(triangles, [r, r, r])
            grid = np.asarray(grid, dtype=np.int8).reshape(r, r, r)

            np.savez_compressed(os.path.join(args.voxel_out, name + '.npz'), grid)

            
    