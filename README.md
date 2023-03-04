# Diffusion Model based Semi-supervised Learning on Brain Hemorrhage Images for Efficient Midline Shift Quantification

Implementation for IPMI 2023 paper [Diffusion Model based Semi-supervised Learning on Brain Hemorrhage Images for Efficient Midline Shift Quantification](https://arxiv.org/abs/2301.00409).
by Shizhan Gong, [Cheng Chen](https://cchen-cc.github.io/), Yuqi Gong, Nga Yan Chan, Wenao Ma, Calvin Hoi-Kwan Mak, Jill Abrigo, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

# Sample Results
![Alt text](results.png?raw=true "Title")

# Setup
We recommend using Miniconda to set up an environment:
```
conda create -n diffusion_mls python=3.8.13
conda activate diffusion_mls
pip install numpy
pip install mpi4py
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai
pip install scikit-image
pip install -e .
```
We managed to test our code on Ubuntu 18.04 with Python 3.8 and CUDA 11.3. Our implementation is based on single GPU setting.

# Dataset
To test our method on your own data, first preprosess the CT image so that the pixel value represents the HU value. 
Save the image as `.npy` format, with each case reshaped into a 3D volume with shape `~30 * 256 * 256`.
The metadata of the images and labels are saves as `.pkl`, here is an example:
```
{
    "patient_id": ["path/to/image.npy",
    1, #1 for MLS cases and 0 for non-MLS cases
    {  11: array([[127.4165 ,  41.228  ],
              [133.25   , 213.89725],
              [117.447  , 100.617  ],
              [129.4555 , 100.253  ]]),
       12: array([[128.    ,  41.865 ],
              [133.929 , 214.7936],
              [115.4835, 109.3895],
              [130.2675, 108.782 ]])}, 
       # 11 and 12 is the number of labeled slice.
       # the array is the 2D coordiantes of four labeled landmark.
       # [anterior falx, posterior falx, shifted landmark, hypothetically normal position of the landmark].
       # can have one or multiple labeled slices.
    14.79647634573854 # case-level MLS measurement.
    ],
    "patient_id": ["path/to/image.npy", 0, {}, 0],
    ......
}
```
For the purpose of training diffusion models, the slice-level label and volume-wise label can be skipped. 
We recommend to upsample the MLS cases when training unconditional diffusion model with extremely imbalanced dataset. i.e. one case corresponds to multiple repeated records in the `.pkl` metadata file.

# Training
The pipeline of our model entails pre-train two diffusion models and then utilize the pre-trained diffusion model to train a deformation model.

## Training Diffusion Models

Type the command below to train the conditional and unconditional diffusion models:
```sh
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
python scripts/image_train.py --data_dir datasets/datafile.pkl $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
As our model requires two pre-trained diffusion models (with only non-MLS images and all images respectively), we need to train these two models separately, by replacing `--data_dir` to corresponding files.

## Training Deformation Network

Type the command below to train the deformation network:
```sh
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
python scripts/deformation_train.py --data_dir datasets/data_train.pkl --val_data_dir datasets/data_eval.pkl --model_con_path models/model_con.pt --model_uncon_path models/model_uncon.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
`--model_con_path` and `--model_uncon_path` denote the path of pre-trained conditional diffusion model and unconditional diffusion model.

# Evaluation
Type the command below to evaluate the performance of deformation network:
```sh
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
python scripts/deformation_evaluate.py --val_data_dir datasets/data_eval.pkl --model_path path/to/deformation/checkpoint.pt --model_con_path models/model_con.pt --model_uncon_path models/model_uncon.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Bibtex
If you find this work helpful, you can cite our paper as follows:
```
@article{gong2023diffusion,
  title={Diffusion Model based Semi-supervised Learning on Brain Hemorrhage Images for Efficient Midline Shift Quantification},
  author={Gong, Shizhan and Chen, Cheng and Gong, Yuqi and Chan, Nga Yan and Ma, Wenao and Mak, Calvin Hoi-Kwan and Abrigo, Jill and Dou, Qi},
  journal={arXiv preprint arXiv:2301.00409},
  year={2023}
}
```

## Acknowledgement
Our code is based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [voxelmorph](https://github.com/voxelmorph/voxelmorph).

## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>