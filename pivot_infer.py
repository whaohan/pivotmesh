import shutil
from einops import rearrange, repeat
from model.meshAE import MeshAutoencoder
from model.pivotmesh import MeshTransformer
import torch
import os
from data.data_utils import write_mesh
from data.dataset import Objaverse, ShapeNetCore
from meshgpt_pytorch.data import custom_collate
from torch.utils.data import DataLoader
from functools import partial
import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--AE_path', type=str)
parser.add_argument('--dataset_name', type=str, default='objaverse')
parser.add_argument('--condition', type=str, default='no')
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--pivot_rate', type=float, default=0.1)
args = parser.parse_args()

if args.dataset_name == 'shapenet':
    max_seq_len = 4900
elif args.dataset_name in ['objaverse', 'objaverse-xl']:
    max_seq_len = 3100


autoencoder = MeshAutoencoder.init_and_load(args.AE_path)
transformer = MeshTransformer(
    autoencoder,
    dim = 1024,
    attn_depth = 24,
    max_seq_len = max_seq_len,
    mode = 'vertices',
)


def write_coords(coords, masks, name='output'):
    for i in range(len(coords)):
        vertices = coords[i][masks[i]].view(-1, 3).cpu()
        n = vertices.shape[0]
        faces = torch.arange(1, n + 1).view(-1, 3)
        try:
            write_mesh(vertices.numpy(), faces.numpy().tolist(), f'{args.output_path}/{name}_{i}.obj')
            print('The number of face:', n//3)
        except:
            pass


if __name__ == '__main__':
    device = torch.device('cuda:0')
    state_dict = torch.load(args.model_path)['model']
    transformer.load_state_dict(state_dict)
    transformer = transformer.to(device)
    transformer = transformer.eval()
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.dataset_name == 'objaverse':
        TRAIN_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/train-500"
        VAL_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/val-500"
        train_dataset = Objaverse(TRAIN_PATH, pivot_rate=args.pivot_rate, quant_bit=7, augment=False, return_model_path=True)
        val_dataset = Objaverse(VAL_PATH, pivot_rate=args.pivot_rate, quant_bit=7, augment=False, return_model_path=True)  

    elif args.dataset_name == 'shapenet':
        TRAIN_PATH = f"{os.environ['HOME']}/data/shapenet-800/train"
        VAL_PATH = f"{os.environ['HOME']}/data/shapenet-800/val"
        train_dataset = ShapeNetCore(TRAIN_PATH, version=2, return_pivot=True, pivot_rate=args.pivot_rate,
                                    synsets=['02828884', '03001627', '03636649', '04379243'],
                                    augment=False, quant_bit=7, return_model_path=True)
        val_dataset = ShapeNetCore(VAL_PATH, version=2, return_pivot=True, pivot_rate=args.pivot_rate,
                                    synsets=['02828884', '03001627', '03636649', '04379243'], 
                                    augment=False, quant_bit=7, return_model_path=True)
        
    elif args.dataset_name == 'objaversexl':
        TRAIN_PATH = f"./data/train/objaverse-objaversexl-500-train.txt"
        VAL_PATH = f"./data/val/objaversexl-500-val.txt"
        # VAL_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/val-500"
        train_dataset = Objaverse(TRAIN_PATH, load_sketch=False, load_pc=False, augment=False,
                                return_pivot=True, pivot_rate=args.pivot_rate)
        val_dataset = Objaverse(VAL_PATH, load_sketch=False, load_pc=False, augment=False,
                                return_pivot=True, pivot_rate=args.pivot_rate)  
        
    dataloader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = True,
        drop_last = True,
        collate_fn = partial(custom_collate, pad_id=-1)
    )

    with torch.no_grad():
        if args.condition == 'no':
            for i in range(args.sample_num):
                coords, masks = transformer.generate(
                    batch_size = args.batch_size,
                    temperature = args.temperature,
                ) # [b, n, 3, 3], [b, n]
                write_coords(coords, masks, name=f'output_{i}')
            
        elif args.condition == 'pivot':
            for i, data in enumerate(dataloader):
                if i == args.sample_num:
                    break
                
                codes = transformer.autoencoder.tokenize(
                    vertices = data['vertices'].to(device),
                    faces = data['faces'].to(device),
                    pivot_mask = data['pivot_mask'].to(device),
                )
                codes = repeat(codes, 'b ... -> (b r) (...)', r = args.batch_size)
                pivot_len = (data['pivot_mask'] != -1).sum(dim=1).to(device)
                
                coords, masks = transformer.generate(
                    prompt = codes[:, :pivot_len * 2 + 2],
                    temperature = args.temperature,
                ) # [b, n, 3, 3], [b, n]
                write_coords(coords, masks, name=f'output_{i}')
                
                # save ground truth
                write_mesh(data['vertices'].squeeze().numpy(), data['faces'].squeeze().numpy().tolist(),
                           f'{args.output_path}/gt_{i}.obj')
                
                pivot_ind = repeat(data['pivot_mask'], 'b n -> b n r', r=3)
                pivots = data['vertices'].gather(1, pivot_ind)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pivots.squeeze(0).numpy())
                o3d.io.write_point_cloud(f'{args.output_path}/pivot_{i}.ply', pcd)
                
    print(f'save at {args.output_path}')

