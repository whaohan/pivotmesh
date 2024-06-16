# PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance

### [Project Page](https://whaohan.github.io/pivotmesh)  | [Paper](https://arxiv.org/abs/2405.16890) | [Weight](https://huggingface.co/whaohan/pivotmesh/tree/main)

### Preparation

Install the packages in `requirements.txt`. The code is tested under CUDA version 12.1.

```bash
conda create -n pivotmesh python=3.9
conda activate pivotmesh
pip install -r requirements.txt
```

### Training

```bash
# train AE
accelerate launch --mixed_precision=fp16 train_AE.py 

# train PivotMesh
accelerate launch --mixed_precision=fp16 train_pivotmesh.py
```

### Inference

```bash
python pivot_infer.py \
    --model_path checkpoints/PivotMesh-objaversexl/mesh-transformer.ckpt.ft.50.pt \
    --AE_path checkpoints/AE-objaversexl/mesh-autoencoder.ckpt.72.pt \
    --output_path output/PivotMesh \
    --dataset_name objaverse \
    --batch_size 16 \
    --sample_num 1 \
    --temperature 0.5 \
    --pivot_rate 0.1 \
    --condition no   # 'no' for unconditional, 'pivot' for conditional 

```

### Evaluation

```bash
# evaluate the performance on PivotMesh
python evaluate.py
```

### Acknowledgement

- [MeshGPT](https://github.com/lucidrains/meshgpt-pytorch)
- [HyperDiffusion](https://github.com/Rgtemze/HyperDiffusion/)

### Citation

```tex
@misc{weng2024pivotmesh,
    title={PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance}, 
    author={Haohan Weng and Yikai Wang and Tong Zhang and C. L. Philip Chen and Jun Zhu},
    year={2024},
    eprint={2405.16890},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
