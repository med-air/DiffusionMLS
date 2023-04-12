import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    RandFlip,
    RandZoom,
    ScaleIntensity,
ScaleIntensityRange,
RandFlip,
RandRotate,
ResizeWithPadOrCrop
)

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not ava:qilable and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    mls = [d[i][3] for i in list(d.keys())]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [d[i][1] for i in list(d.keys())]

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        mls = mls,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_data_slice(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    landmark = False,
    mask = True,
    return_loader = False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [d[i][1] for i in list(d.keys())]
    landmarks = None
    if landmark:
        landmarks = [d[i][2] for i in list(d.keys())]
    mls = [d[i][3] for i in list(d.keys())]

    dataset = SliceDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        landmarks = landmarks,
        mls = mls,
        mask = mask
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )

    while True:
        yield from loader


def load_data_slice_epoch(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    landmark = False,
    mask = True,
    return_loader = False,
    semi_prop = 1.
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [d[i][1] for i in list(d.keys())]
    landmarks = None
    if landmark:
        landmarks = [d[i][2] for i in list(d.keys())]
    mls = [d[i][3] for i in list(d.keys())]

    dataset = SliceDataset_epoch(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        landmarks = landmarks,
        mls = mls,
        mask = mask,
        semi_prop = semi_prop
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    if return_loader:
        return loader

def load_data_slice_epoch_lm(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    landmark = False,
    mask = True,
    return_loader = False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [d[i][1] for i in list(d.keys())]
    landmarks = None
    if landmark:
        landmarks = [d[i][2] for i in list(d.keys())]
    mls = [d[i][3] for i in list(d.keys())]

    dataset = SliceDataset_epoch_lm(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        landmarks = landmarks,
        mls = mls,
        mask = mask
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    if return_loader:
        return loader

def load_data_slice_lm(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    landmark = False,
    return_loader = False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [d[i][1] for i in list(d.keys())]
    if landmark:
        landmarks = [d[i][2] for i in list(d.keys())]
    else:
        landmarks = None
    mls = [d[i][3] for i in list(d.keys())]
    dataset = SliceDataset_lm(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        landmarks=landmarks,
        mls = mls
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    if return_loader:
        return loader

def load_data_volume(
        *,
        data_dir,
        batch_size,
        deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)
    all_files = [d[i][0] for i in list(d.keys())]
    slice_meta = [d[i][3] for i in list(d.keys())]

    dataset = VolumeDataset(
        all_files,
        slice_meta
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader


class VolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,
            mls_info,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.mls_info = mls_info

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        path = self.img_dict[idx]
        train_imtrans = Compose(
            [LoadImage(image_only=True, ensure_channel_first=True),
             ScaleIntensityRange(0, 80, -1, 1, clip=True),
             ])
        arr = train_imtrans(path)
        mls = self.mls_info[idx]

        return arr[0], mls

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
from skimage.measure import label
import torch as th
class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        mls = None
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.positive = np.array(self.local_images)[np.array(self.local_classes) == 1]
        self.random_flip = random_flip
        self.mls = mls

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        out_dict = {}
        path = self.local_images[idx]
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.random_crop:
            train_imtrans = Compose(
                [LoadImage(image_only=True, ensure_channel_first=True),
                 ScaleIntensityRange(0, 80, -1, 1, clip=True),
                 RandFlip(prob=0.5, spatial_axis=2),
                 RandRotate(range_x=np.pi / 6, prob=0.5),
                 ResizeWithPadOrCrop([32, 256, 256], mode="constant", constant_values=-1)]
            )
        else:
            train_imtrans = Compose(
                [LoadImage(image_only=True, ensure_channel_first=True),
                 ScaleIntensityRange(0, 80, -1, 1, clip=True),
                 ResizeWithPadOrCrop([32, 256, 256], mode="constant", constant_values=-1)]
            )

        arr = train_imtrans(path)
        arr = arr.astype(np.float32)
        out_dict["mls"] = self.mls[idx]
        return arr[0], out_dict


class SliceDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        landmarks = None,
        mls = None,
        mask = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_landmarks = None if landmarks is None else landmarks[shard:][::num_shards]
        self.mask = mask
        self.mls = mls

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        if self.random_flip:
            train_imtrans = Compose(
            [LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensityRange(0, 80, -1, 1, clip=True),
             RandFlip(prob=0.5, spatial_axis=2),
             RandRotate(range_x=np.pi / 12, prob=0.5)
            ]
            )
        else:
            train_imtrans = Compose(
                [LoadImage(image_only=True, ensure_channel_first=True),
                 ScaleIntensityRange(0, 80, -1, 1, clip=True),
                 ]
            )
        arr = train_imtrans(path)
        d = arr.shape[1]
        d_sample = np.random.randint(8, d-5)
        arr_slice = arr[:, d_sample].astype(np.float32)
        return arr_slice, {}

class SliceDataset_epoch(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        landmarks = None,
        mls = None,
        mask = True,
        semi_prop = 1.
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_landmarks = landmarks[shard:][::num_shards]
        self.mask = mask
        self.mls = mls
        idx = 0
        img_dict = {}
        for img_address, point, mls in zip(self.local_images, self.local_landmarks, self.mls):
            p = np.random.rand()
            img = np.load(img_address)
            for j in range(8, len(img)-5):
                tmp = [img_address, j, mls]
                if j in point:
                    tmp.append(point[j])
                    img_dict[idx] = tmp
                    idx += 1
                else:
                    if p < semi_prop:
                        img_dict[idx] = tmp
                        idx += 1
                    else:
                        continue
        self.img_dict = img_dict

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        tmp = self.img_dict[idx]
        if len(tmp) == 3:
            path, d_sample, mls = tmp
            landmarks = None
        else:
            path, d_sample, mls, landmarks = tmp

        train_imtrans = Compose(
            [LoadImage(image_only=True, ensure_channel_first=True),
             ScaleIntensityRange(0, 80, -1, 1, clip=True),
             ]
        )

        arr = train_imtrans(path)
        arr_slice = arr[:, d_sample].astype(np.float32)
        val_img = arr_slice[0]
        out_dict = {}
        try:
            mask = (val_img != -1)
            labels = label(mask)
            largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
            largestCC = 1 - largestCC
            labels = label(largestCC)
            largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
            mask = 1 - largestCC
            out_dict["mask"] = th.tensor(mask).unsqueeze(0)
        except:
            out_dict["mask"] = th.ones([1, 256, 256]).long()
        if landmarks is not None:
            out_dict["landmarks"] = th.tensor(landmarks)
        else:
            out_dict["landmarks"] = th.ones([4, 2])
        out_dict["mls"] = mls
        return arr_slice, out_dict


class SliceDataset_epoch_lm(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        landmarks = None,
        mls = None,
        mask = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_landmarks = landmarks[shard:][::num_shards]
        self.mask = mask
        self.mls = mls
        idx = 0
        img_dict = {}
        for img_address, point, mls in zip(self.local_images, self.local_landmarks, self.mls):
            img = np.load(img_address)
            for j in range(8, len(img)-5):
                tmp = [img_address, j, mls]
                if j in point:
                    tmp.append(point[j])
                    img_dict[idx] = tmp
                    idx += 1
                else:
                    continue
        self.img_dict = img_dict

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        tmp = self.img_dict[idx]
        if len(tmp) == 3:
            path, d_sample, mls = tmp
            landmarks = None
        else:
            path, d_sample, mls, landmarks = tmp

        train_imtrans = Compose(
            [LoadImage(image_only=True, ensure_channel_first=True),
             ScaleIntensityRange(0, 80, -1, 1, clip=True),
             ]
        )

        arr = train_imtrans(path)
        arr_slice = arr[:, d_sample].astype(np.float32)
        val_img = arr_slice[0]
        out_dict = {}
        try:
            mask = (val_img != -1)
            labels = label(mask)
            largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
            largestCC = 1 - largestCC
            labels = label(largestCC)
            largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
            mask = 1 - largestCC
            out_dict["mask"] = th.tensor(mask).unsqueeze(0)
        except:
            out_dict["mask"] = th.ones([1, 256, 256]).long()
        if landmarks is not None:
            out_dict["landmarks"] = th.tensor(landmarks)
        else:
            out_dict["landmarks"] = th.ones([4, 2])
        out_dict["mls"] = mls
        return arr_slice, out_dict

class SliceDataset_lm(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=True,
            random_flip=True,
            landmarks=None,
            mls = None
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_landmarks = None if landmarks is None else landmarks[shard:][::num_shards]
        self.mls = mls

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        train_imtrans = Compose(
            [LoadImage(image_only=True, ensure_channel_first=True),
             ScaleIntensityRange(0, 80, -1, 1, clip=True),
             ]
        )
        arr = train_imtrans(path)
        d = arr.shape[1]
        i = 0

        i += 1
        d_sample = np.random.randint(0, len(self.local_landmarks[idx].keys()))
        d_sample = list(self.local_landmarks[idx].keys())[d_sample]
        arr_slice = arr[:, d_sample].astype(np.float32)
        val_img = arr_slice[0]
        mask = (val_img != -1)
        labels = label(mask)
        largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
        largestCC = 1 - largestCC
        labels = label(largestCC)
        largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(int)
        mask = 1 - largestCC
        out_dict = {}
        out_dict["mask"] = th.tensor(mask).unsqueeze(0)
        if self.local_landmarks is not None:
            out_dict["landmarks"] = th.tensor(np.array(self.local_landmarks[idx][d_sample]))
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        out_dict["mls"] = self.mls[idx]
        return arr_slice, out_dict

