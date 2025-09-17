import os
import psutil, shutil

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from monai.data import DataLoader  # , Dataset
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                              Orientationd, Spacingd, ScaleIntensityRanged,
                              EnsureTyped)

import numpy as np
import pathlib
from monai.data import MetaTensor
import SimpleITK as sitk
from monai.transforms import ScaleIntensityRange
import monai

# ---------------- resource thresholds -----------------
MIN_FREE_RAM_GB = 1.0
MIN_FREE_DISK_GB = 1.0
CHECK_EVERY_N_SLICES = 50
TARGET_SPACING = (0.8, 0.8, 3.0)


# ------------------------------------------------------


def _enough_resources():
    avail_ram_gb = psutil.virtual_memory().available / 2 ** 30
    cache_root = os.path.expanduser("~/.cache")
    avail_disk_gb = shutil.disk_usage(cache_root).free / 2 ** 30
    return (avail_ram_gb > MIN_FREE_RAM_GB
            and avail_disk_gb > MIN_FREE_DISK_GB), avail_ram_gb, avail_disk_gb


def center_pad_crop_2d(t, size=256):
    # t: (1,H,W)
    _, H, W = t.shape
    # pad (bottom/right) if smaller than target
    pad_h = max(0, size - H)
    pad_w = max(0, size - W)
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    # center crop to exactly size×size
    H2, W2 = t.shape[1:]
    start_h = (H2 - size) // 2
    start_w = (W2 - size) // 2
    t = t[:, start_h:start_h + size, start_w:start_w + size]
    return t


class NRRDDataset(Dataset):

    def __init__(self,
                 img_dir=None,
                 seg_dir=None,
                 split="train",
                 img_size=256,
                 segmentation_guided=True):
        super().__init__()
        assert split in ("train", "val", "test")
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.split = split
        self.img_size = img_size
        self.segmentation_guided = segmentation_guided
        self.samples = []

        # discover segmentation types
        seg_types = os.listdir(seg_dir) if segmentation_guided else []
        seg_keys = [f"seg_{t}" for t in seg_types]

        # list volumes
        if img_dir is not None:
            vol_paths = [
                os.path.join(img_dir, split, f)
                for f in os.listdir(os.path.join(img_dir, split))
                if f.endswith(".nrrd")
            ]
        else:
            vol_paths = [
                os.path.join(seg_dir, seg_types[0], split, f)
                for f in os.listdir(os.path.join(seg_dir, seg_types[0], split))
                if f.endswith(".nrrd")
            ]

        # keys list: image + all segmentations
        all_keys = (["image"] if img_dir is not None else []) + seg_keys

        # transform once per volume
        vol_transform = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(keys=all_keys,
                     pixdim=TARGET_SPACING,
                     mode=("bilinear",) +
                          ("nearest",) * len(seg_keys) if img_dir is not None else
                     ("nearest",) * len(seg_keys)),
            # scale only the image
            ScaleIntensityRanged(keys="image",
                                 a_min=-1000,
                                 a_max=1000,
                                 b_min=-1.0,
                                 b_max=1.0,
                                 clip=True) if img_dir is not None else
            (lambda x: x),
            EnsureTyped(keys=all_keys),
        ])

        # --- preload volumes and slice ---
        slice_counter = 0
        for vol_path in vol_paths:
            # assemble dictionary for this volume
            data_dict = {}
            if img_dir is not None:
                data_dict["image"] = vol_path
            for t in seg_types:
                data_dict[f"seg_{t}"] = os.path.join(
                    seg_dir, t, split, os.path.basename(vol_path))

            data = vol_transform(data_dict)  # after this: tensors (1,H,W,D)

            # depth along last axis
            ref_key = "image" if img_dir is not None else seg_keys[0]
            depth = data[ref_key].shape[-1]

            stem = os.path.splitext(os.path.basename(vol_path))[0]
            for slice_idx in range(depth):
                record = {}
                if img_dir is not None:
                    img_slice = data["image"][..., slice_idx]
                    img_slice = center_pad_crop_2d(img_slice, size=img_size)
                    record["images"] = img_slice
                if segmentation_guided:
                    for sk in seg_keys:
                        m_slice = data[sk][..., slice_idx]
                        m_slice = center_pad_crop_2d(m_slice, size=img_size)
                        record[sk] = m_slice
                record["image_filenames"] = f"{stem}_axial_{slice_idx:04d}"
                self.samples.append(record)

                slice_counter += 1
                if slice_counter % CHECK_EVERY_N_SLICES == 0:
                    ok, ram, disk = _enough_resources()
                    if not ok:
                        print(
                            f"[NRRDDataset] stopping preload: only {ram:.1f} GB RAM / {disk:.1f} GB disk free"
                        )
                        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ----------------------------------------------------------------------


def center_pad_crop_k(t, size):
    k, h, w = t.shape
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h))
        h, w = t.shape[1:]
    sh, sw = (h - size) // 2, (w - size) // 2
    return t[:, sh:sh + size, sw:sw + size].contiguous()


class MultiSliceDataset(Dataset):

    def __init__(
            self,
            img_dir=None,
            seg_dir=None,
            split="train",
            img_size=256,
            segmentation_guided=True,
            use_multislice=True,
            num_slice=2,
            stride=1,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert stride >= 1 and num_slice >= 1

        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.split = split
        self.img_size = img_size
        self.segmentation_guided = segmentation_guided
        self.k = num_slice if use_multislice else 1
        self.stride = stride

        # discover segmentation types
        self.seg_types = os.listdir(seg_dir) if (segmentation_guided
                                                 and seg_dir) else []
        self.seg_keys = [f"seg_{t}" for t in self.seg_types]

        # list volumes
        if img_dir:
            vol_paths = [
                os.path.join(img_dir, split, f)
                for f in os.listdir(os.path.join(img_dir, split))
                if f.lower().endswith(".nrrd")
            ]
        else:
            assert seg_dir and self.seg_types, "Need seg_dir with seg types when img_dir is None"
            base = os.path.join(seg_dir, self.seg_types[0], split)
            vol_paths = [
                os.path.join(base, f) for f in os.listdir(base)
                if f.lower().endswith(".nrrd")
            ]

        # keys list: image + all segmentations
        all_keys = (["image"] if img_dir else []) + self.seg_keys

        self.vol_transform = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(
                keys=all_keys,
                pixdim=TARGET_SPACING,
                mode=("bilinear",) +
                     ("nearest",) * len(self.seg_keys) if img_dir is not None else
                ("nearest",) * len(self.seg_keys)),
            # scale only the image
            ScaleIntensityRanged(keys="image",
                                 a_min=-1000,
                                 a_max=1000,
                                 b_min=-1.0,
                                 b_max=1.0,
                                 clip=True) if img_dir is not None else
            (lambda x: x),
            EnsureTyped(keys=all_keys),
        ])

        self._vol_cache = {}
        self.index = []  # list of {vol_id , start, image_filename}

        for vol_id, vol_path in enumerate(vol_paths):
            paths = {"image": vol_path} if img_dir else {}
            for t in self.seg_types:
                paths[f"seg_{t}"] = os.path.join(seg_dir, t, split,
                                                 os.path.basename(vol_path))

            vol = self.vol_transform(paths)  # tensors (1,H,W,D)
            self._vol_cache[vol_id] = vol

            # depth along last axis
            depth = vol["image" if img_dir else self.seg_keys[0]].shape[-1]

            max_start = depth - self.k
            if max_start < 0:
                continue
            stem = os.path.splitext(os.path.basename(vol_path))[0]
            for start_idx in range(0, max_start + 1, self.stride):
                self.index.append({
                    "vol_id":
                        vol_id,
                    "start_slice":
                        start_idx,
                    "image_filenames":
                        f"{stem}_axial_{start_idx:04d}of{depth - 1:04d}",
                })

        print(
            f"[MultiSliceDataset] {split}: {len(self.index):,} windows from {len(self._vol_cache)} volumes"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record = self.index[idx]
        vol_id, start_idx = record["vol_id"], record["start_slice"]
        vol = self._vol_cache[vol_id]

        sample = {}
        if self.img_dir:
            img_block = vol["image"][...,
            start_idx:start_idx + self.k][0].permute(
                2, 0, 1)
            sample["images"] = center_pad_crop_k(img_block, self.img_size)

        if self.segmentation_guided:
            for sk in self.seg_keys:
                seg_block = vol[sk][...,
                start_idx:start_idx + self.k][0].permute(
                    2, 0, 1)
                sample[sk] = center_pad_crop_k(seg_block, self.img_size)

        sample["image_filenames"] = record["image_filenames"]
        return sample


class NrrdDataset(Dataset):

    def __init__(self,
                 image_dir,
                 label_dir,
                 img_size=256,
                 seq_len=8,
                 mode="train"):
        image_dir = pathlib.Path(image_dir)
        label_dir = pathlib.Path(label_dir)
        self.image_paths = sorted(
            [str(f) for f in list(image_dir.rglob("*.nrrd"))])
        self.label_paths = sorted(
            [str(f) for f in list(label_dir.rglob("*.nrrd"))])
        print(f"{len(self.image_paths)} volumes found in {image_dir}")
        self.seq_len = seq_len
        self.mode = mode

        # Preprocess
        self.image_lists = [sitk.ReadImage(path) for path in self.image_paths]
        self.label_lists = [sitk.ReadImage(path) for path in self.label_paths]
        self.resampled_images = []
        self.resampled_labels = []
        for image, label in zip(self.image_lists, self.label_lists):
            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            image_x, image_y, image_z = image_array.shape
            label_array = sitk.GetArrayFromImage(label).astype(np.int64)
            transform = ScaleIntensityRange(a_min=-1000,
                                            a_max=1000,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True)
            if image_x != img_size or image_y != img_size or image_z != img_size:
                resampled_array = self.resample(image,
                                                target_shape=(img_size,
                                                              img_size,
                                                              img_size))
                resampled_array = transform(resampled_array)
                self.resampled_images.append(resampled_array)
                resampled_label = self.resample(label,
                                                target_shape=(img_size,
                                                              img_size,
                                                              img_size))
                self.resampled_labels.append(resampled_label)
            else:
                image_array = transform(image_array.unsqueeze(0))
                self.resampled_images.append(torch.tensor(image_array))
                self.resampled_labels.append(
                    torch.tensor(label_array).unsqueeze(0))
        print(f"Preprocessed {len(self.resampled_images)} volumes.")

        # get slice
        self.img_slices = torch.cat(self.resampled_images,
                                    dim=1)  
        self.img_slices = self.img_slices.squeeze(0)  
        self.label_slices = torch.cat(self.resampled_labels,
                                      dim=1)  
        self.label_slices = self.label_slices.squeeze(0) 
        self.slice_count = self.img_slices.shape[0] // self.seq_len

    def __len__(self):
        if self.mode == "train":
            return self.slice_count
        else:
            return len(self.label_lists)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.img_slices[idx * self.seq_len:(idx + 1) *
                                                     self.seq_len]  
            label = self.label_slices[idx * self.seq_len:(idx + 1) *
                                                         self.seq_len]  
            return img, label  # Return preprocessed volume
        else:
            # index = torch.randint(0, len(self.label_lists) - 1, (1, )).item()
            index = idx
            img = self.resampled_images[index].squeeze()  
            label = self.resampled_labels[index].squeeze()  
            return img, label

    def resample(self, image, target_shape=(256, 256, 256)):
        origin = image.GetOrigin()
        old_spacing = image.GetSpacing()
        image_array = sitk.GetArrayFromImage(image)
        original_spacing = list(old_spacing)
        old_shape = image_array.shape
        new_spacing = [
            original_spacing[i] * old_shape[i] / target_shape[i]
            for i in range(3)
        ]

        affine_matrix = self.create_affine(original_spacing, origin)
        image_tensor = torch.tensor(image_array).unsqueeze(0)
        meta_tensor = MetaTensor(image_tensor, affine_matrix)
        resampler = monai.transforms.Spacing(pixdim=new_spacing,
                                             mode="bilinear")
        resampled_tensor = resampler(meta_tensor)

        resample_transform = monai.transforms.ResizeWithPadOrCrop(
            spatial_size=target_shape)
        strict_resampled_tensor = resample_transform(resampled_tensor)

        resampled_array = strict_resampled_tensor
        return resampled_array  

    @staticmethod
    def create_affine(spacing, origin):
        affine = np.eye(4)  
        affine[:3, :3] = np.diag(spacing)  
        affine[:3, 3] = origin  
        return affine

class InfDataset(Dataset):

    def __init__(self,
                 label_dir,
                 img_size=256,
                 seq_len=8,
                 mode="train"):
        label_dir = pathlib.Path(label_dir)

        self.label_paths = sorted(
            [str(f) for f in list(label_dir.rglob("*.nrrd"))])

        self.seq_len = seq_len
        self.mode = mode

        # Preprocess
        self.label_lists = [sitk.ReadImage(path) for path in self.label_paths]
        self.resampled_images = []
        self.resampled_labels = []
        for label in self.label_lists:

            label_array = sitk.GetArrayFromImage(label).astype(np.int64)
            label_array[label_array!=0] = 1
            image_x, image_y, image_z = label_array.shape
            transform = ScaleIntensityRange(a_min=-1000,
                                            a_max=1000,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True)
            if image_x != img_size or image_y != img_size or image_z != img_size:
                resampled_label = self.resample(label,
                                                target_shape=(img_size,
                                                              img_size,
                                                              img_size))
                self.resampled_labels.append(resampled_label)
            else:

                self.resampled_labels.append(
                    torch.tensor(label_array).unsqueeze(0))
        print(f"Preprocessed {len(self.resampled_labels)} volumes.")

        # get slice
        self.label_slices = torch.cat(self.resampled_labels,
                                      dim=1)  # (1, N*D, H, W)
        self.label_slices = self.label_slices.squeeze(0)  # (N*D, H, W)
        self.slice_count = self.label_slices.shape[0] // self.seq_len

    def __len__(self):
        if self.mode == "train":
            return self.slice_count
        else:
            return len(self.label_lists)

    def __getitem__(self, idx):
        if self.mode == "train":
            label = self.label_slices[idx * self.seq_len:(idx + 1) *
                                                         self.seq_len]  # (seq_len, H, W)
            return label, label  # Return preprocessed volume
        else:
            # index = torch.randint(0, len(self.label_lists) - 1, (1, )).item()
            label = self.resampled_labels[idx].squeeze()  # (D, H, W)
            return self.label_paths[idx], label

    def resample(self, image, target_shape=(256, 256, 256)):
        origin = image.GetOrigin()
        old_spacing = image.GetSpacing()
        image_array = sitk.GetArrayFromImage(image)
        original_spacing = list(old_spacing)
        old_shape = image_array.shape
        new_spacing = [
            original_spacing[i] * old_shape[i] / target_shape[i]
            for i in range(3)
        ]

        affine_matrix = self.create_affine(original_spacing, origin)
        image_tensor = torch.tensor(image_array).unsqueeze(0)
        meta_tensor = MetaTensor(image_tensor, affine_matrix)
        resampler = monai.transforms.Spacing(pixdim=new_spacing,
                                             mode="bilinear")
        resampled_tensor = resampler(meta_tensor)

        resample_transform = monai.transforms.ResizeWithPadOrCrop(
            spatial_size=target_shape)
        strict_resampled_tensor = resample_transform(resampled_tensor)

        resampled_array = strict_resampled_tensor
        return resampled_array  # [1, D, H, W]

    @staticmethod
    def create_affine(spacing, origin):
        affine = np.eye(4)  
        affine[:3, :3] = np.diag(spacing)  
        affine[:3, 3] = origin  
        return affine
