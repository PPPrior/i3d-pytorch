# I3D-PyTorch
This is a simple and crude implementation of Inflated 3D ConvNet Models (I3D) in PyTorch. Different from models reported in "[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew Zisserman, this implementation uses [ResNet](https://arxiv.org/pdf/1512.03385.pdf) as backbone.

<div align="center">
  <img src="QuoVadis.png" width="600px"/>
</div>

This implementation is based on OpenMMLab's [MMAction2](https://github.com/open-mmlab/mmaction2). 

## Data Preparation

For optical flow extraction and video list generation, please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks#code--data-preparation) for details.

## Training

To train a new model, use the `main.py` script.

For example, command to train models with RGB modality on UCF101 can be

```bash
python main.py ucf101 RGB <root_path> \
    <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
    --arch i3d_resnet50 --clip_length 64 \
    --lr 0.001 --lr_steps 30 60 --epochs 80 \
    -b 32 -j 8 --dropout 0.8 \
    --snapshot_pref ucf101_i3d_resnet50
```

For flow models:

```bash
python main.py ucf101 Flow <root_path> \
    <ucf101_flow_train_list> <ucf101_flow_val_list> \
    --arch i3d_resnet50 --clip_length 64 \
    --lr 0.001 --lr_steps 15 30 --epochs 40 \
    -b 64 -j 8 --dropout 0.8 \
    --snapshot_pref ucf101_i3d_resnet50
```

Please refer to [main.py](main.py) for more details.

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_i3d_resnet50_rgb_model_best.pth.tar`.

Use the following command to test its performance:

```bash
python test_models.py ucf101 RGB <root_path> \
    <ucf101_rgb_val_list> ucf101_i3d_resnet50_rgb_model_best.pth.tar \
    --arch i3d_resnet50 --save_scores <score_file_name>
```

Or for flow models:

```bash
python test_models.py ucf101 Flow <root_path> \
    <ucf101_flow_val_list> ucf101_i3d_resnet50_flow_model_best.pth.tar \
    --arch i3d_resnet50 --save_scores <score_file_name>
```

Please refer to [test_models.py](test_models.py) for more details.