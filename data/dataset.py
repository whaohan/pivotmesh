import random
import torch
from PIL import Image
import json
import os
import warnings
from os import path
from pathlib import Path
from typing import Dict
from .data_utils import load_process_mesh, to_mesh
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
SYNSET_DICT_DIR = Path(__file__).resolve().parent

class ShapeNetCore(torch.utils.data.Dataset):  # pragma: no cover
    def __init__(
        self,
        data_dir,
        synsets=None,
        version: int = 2,
        load_textures: bool = False,
        texture_resolution: int = 4,
        augment: bool = False,
        augment_dict: dict = None,
        return_model_path: bool = False,
        return_pivot: bool = True,
        pivot_rate: float = 0.1,
        patch_size: int = 1,
        quant_bit: int = 8,
    ) -> None:
        super().__init__()
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = data_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.augment = augment
        self.patch_size = patch_size
        self.quant_bit = quant_bit
        self.augment_dict = augment_dict
        self.return_model_path = return_model_path
        self.return_pivot = return_pivot
        self.pivot_rate = pivot_rate


        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"

        # Synset dictionary mapping synset offsets to corresponding labels.
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(path.join(SYNSET_DICT_DIR, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # If categories are specified, check if each category is in the form of either
        # synset offset or synset label, and if the category exists in the given directory.
        if synsets is not None:
            # Set of categories to load in the form of synset offsets.
            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    path.isdir(path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    (path.isdir(path.join(data_dir, self.synset_inv[synset])))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)
        # If no category is given, load every category in the given directory.
        # Ignore synset folders not included in the official mapping.
        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset]) for synset in synset_not_present]

        # Extract model_id of each object from directory names.
        # Each grandchildren directory of data_dir contains an object, and the name
        # of the directory is the object's model_id.
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)
                    warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count

    def __len__(self) -> int:
        return len(self.model_ids)

    def _get_item_ids(self, idx) -> Dict:
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def load_mesh(self, model_path):
        # load and process mesh
        mesh = load_process_mesh(model_path, quantization_bits=self.quant_bit, 
                                 augment=self.augment, augment_dict=self.augment_dict)
        
        verts, faces = mesh['vertices'], mesh['faces']
        verts = torch.tensor(verts)
        faces = torch.tensor(faces)
        return verts, faces

    def __getitem__(self, idx: int) -> Dict:
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        try:
            verts, faces = self.load_mesh(model_path)
        except Exception as e:
            print('Load mesh fail at:', model_path, e)
            model = self._get_item_ids(random.randint(0, len(self.model_ids)-1))
            model_path = path.join(
                self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
            )
            verts, faces = self.load_mesh(model_path)
        
        data = {}
        data["vertices"] = verts
        data["faces"] = faces

        if self.return_pivot:
            mesh = to_mesh(vertices=verts, faces=faces)
            num_verts = len(verts)
            degree = torch.tensor(mesh.vertex_degree)
            assert num_verts == len(degree)
            
            # pivot points selection (0.1 will be fine)
            vals, indexes = torch.topk(degree, int(num_verts * self.pivot_rate) + 1, sorted=False)
            pivot_len = 50 * int(10 * self.pivot_rate)
            indexes = torch.nonzero(degree >= vals.min()).squeeze(dim=-1)[:pivot_len]
            
            data['pivot_mask'] = indexes
            
        if self.return_model_path:
            data['model_path'] = model_path
        
        
        return data


class Objaverse(torch.utils.data.Dataset):  # pragma: no cover
    def __init__(
        self,
        data_dir,
        patch_size: int = 1,
        quant_bit: int = 8,
        augment: bool = False,
        augment_dict: dict = None,
        codes_path: str = None,
        return_model_path: bool = False,
        return_pivot: bool = True,
        pivot_path: str = None,
        pivot_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.quant_bit = quant_bit
        self.augment = augment
        self.augment_dict = augment_dict
        self.codes_path = codes_path
        self.return_model_path = return_model_path
        self.return_pivot = return_pivot
        self.pivot_path = pivot_path
        self.pivot_rate = pivot_rate
        
        if self.pivot_path is not None:
            self.data_path = os.listdir(pivot_path)
        elif path.isdir(self.data_dir):
            self.data_path = os.listdir(data_dir)
        else:
            with open(self.data_dir, 'r') as f:
                self.data_path = [line.strip() for line in f]
            self.data_dir = '/'

    def __len__(self) -> int:
        return len(self.data_path)


    def load_mesh(self, model_path):
        # load and process mesh
        mesh = load_process_mesh(model_path, quantization_bits=self.quant_bit, 
                                 augment=self.augment, augment_dict=self.augment_dict)
        
        verts, faces = mesh['vertices'], mesh['faces']
        verts = torch.tensor(verts)
        faces = torch.tensor(faces)
        return verts, faces
    

    def __getitem__(self, idx: int) -> Dict:
        model_path = path.join(self.data_dir, self.data_path[idx])
        model_id = path.basename(self.data_path[idx]).split('.')[0]
        data = {}
        
        if self.pivot_path is not None:
            pivot_indexes = os.path.join(self.pivot_path, f'{model_id}.pt')
            data = torch.load(pivot_indexes)
            return data
        
        if self.codes_path:
            code_path = os.path.join(self.codes_path, model_id + '.pt')
            data['codes'] = torch.load(code_path)
        else:
            try:
                verts, faces = self.load_mesh(model_path)
            except Exception as e:
                print('Load mesh fail at:', model_path, e)
                model_path = path.join(self.data_dir, random.choice(self.data_path))
                verts, faces = self.load_mesh(model_path)
            # verts, faces = self.load_mesh(model_path)

            data["vertices"] = verts
            data["faces"] = faces
            
        if self.return_pivot:
            mesh = to_mesh(vertices=verts, faces=faces)
            num_verts = len(verts)
            degree = torch.tensor(mesh.vertex_degree)
            assert num_verts == len(degree)
            
            # pivot sample with random dropout
            extend_rate = 1.5
            vals, indexes = torch.topk(degree, int(num_verts * self.pivot_rate * extend_rate) + 1)
            indexes = torch.nonzero(degree >= vals.min()).squeeze(dim=-1)
            
            drop_mask = torch.rand(len(indexes)) <= (1 / extend_rate)
            pivot_len = 50 * int(10 * self.pivot_rate)
            indexes = indexes[drop_mask][:pivot_len]

            data['pivot_mask'] = indexes
            
        if self.return_model_path:
            data['model_path'] = model_path
            
        return data
