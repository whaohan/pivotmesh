from trainer import MeshTransformerTrainer, trackers
from model.meshAE import MeshAutoencoder
from model.pivotmesh import MeshTransformer
from data.dataset import Objaverse, ShapeNetCore
from accelerate.utils import DistributedDataParallelKwargs
import os
import yaml

with open("configs/PivotMesh.yaml","r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

autoencoder = MeshAutoencoder.init_and_load(config['model_path'])
transformer = MeshTransformer(
    autoencoder,
    dim = config['dim'],
    attn_depth = config['depth'],
    max_seq_len = config['max_seq_len'],
    dropout = config['dropout'],
    mode = config['mode'],
)

if config['resume']:
    transformer.load(config['resume'])

if config['dataset_name'] == 'shapenet':
    TRAIN_PATH = f"{os.environ['HOME']}/data/shapenet-800/train"
    VAL_PATH = f"{os.environ['HOME']}/data/shapenet-800/val"
    train_dataset = ShapeNetCore(TRAIN_PATH, version=2, return_pivot=True, pivot_rate=config['pivot_rate'],
                                synsets=['02828884', '03001627', '03636649', '04379243'], 
                                augment=True, augment_dict=config['augment_dict'], quant_bit=config['quant_bit'])
    val_dataset = ShapeNetCore(VAL_PATH, version=2, return_pivot=True, pivot_rate=config['pivot_rate'],
                            synsets=['02828884', '03001627', '03636649', '04379243'], 
                            augment=True, augment_dict=config['augment_dict'], quant_bit=config['quant_bit'])
    
elif config['dataset_name'] in ['objaverse', 'objaversexl']:
    TRAIN_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/train-500"
    VAL_PATH = f"{os.environ['HOME']}/data/objaverse-lp-500/val-500"
    train_dataset = Objaverse(TRAIN_PATH, augment=True, augment_dict=config['augment_dict'],
                              return_pivot=True, pivot_rate=config['pivot_rate'], pivot_path=config['pivot_path'],
                              codes_path = os.path.join(config['codes_path'], 'train') if config['codes_path'] else None)
    val_dataset = Objaverse(VAL_PATH, augment=True, augment_dict=config['augment_dict'],
                            return_pivot=True, pivot_rate=config['pivot_rate'],
                            codes_path = os.path.join(config['codes_path'], 'val') if config['codes_path'] else None)  


trainer = MeshTransformerTrainer(
    model = transformer,
    dataset = train_dataset,
    val_dataset = val_dataset,
    val_every = config['val_every'],
    val_num_batches = 1,
    num_train_steps = int(config['num_train_steps']),
    batch_size = config['batch_size'],
    learning_rate = config['learning_rate'],
    grad_accum_every = config['grad_accum_every'],
    warmup_steps = config['warmup_steps'], 
    weight_decay = config['weight_decay'],
    use_wandb_tracking = True,
    checkpoint_every = config['checkpoint_every'],
    checkpoint_folder = f'./checkpoints/{config["exp_name"]}',
    ema_kwargs = {
        "allow_different_devices": True,
    },
    accelerator_kwargs = {
        'kwargs_handlers': [
            DistributedDataParallelKwargs(find_unused_parameters=False)
        ]
    }
)


with trackers(trainer, project_name='PivotMesh', run_name=config["exp_name"], hps=config):
    trainer()
