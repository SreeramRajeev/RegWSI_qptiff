#!/usr/bin/env python3
"""
Direct QPTIFF Registration Pipeline
===================================
Works directly with QPTIFF files at full resolution
Preserves H&E colors in output
Maintains exact input dimensions

Optimized for vast.ai with NVIDIA H200 GPU
"""

### System Imports ###
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch as tc
import cv2
import matplotlib
matplotlib.use('Agg')
import tifffile
import SimpleITK as sitk
import matplotlib.pyplot as plt

import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

DEFAULT_IF_CHANNELS = [0, 1, 5]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def preprocess_for_registration(source, target):
    if source.shape[:2] != target.shape[:2]:
        source = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

    source_gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2GRAY) if source.ndim == 3 else source
    target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY) if target.ndim == 3 else target

    blur = lambda img: cv2.GaussianBlur(img, (5, 5), 1.0)
    source_blur, target_blur = blur(source_gray), blur(target_gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    enhance = lambda img: clahe.apply(img)
    source_enhanced, target_enhanced = enhance(source_blur), enhance(target_blur)

    edge = lambda img: cv2.Canny(img, 30, 100)
    source_combined = cv2.addWeighted(source_enhanced, 0.7, edge(source_enhanced), 0.3, 0)
    target_combined = cv2.addWeighted(target_enhanced, 0.7, edge(target_enhanced), 0.3, 0)

    return cv2.cvtColor(source_combined, cv2.COLOR_GRAY2RGB), cv2.cvtColor(target_combined, cv2.COLOR_GRAY2RGB)

def create_registration_params():
    params = default_initial_nonrigid()
    params['initial_alignment_params'] = {
        'type': 'feature_based', 'detector': 'superpoint', 'matcher': 'superglue',
        'ransac_threshold': 10.0, 'max_features': 10000, 'match_ratio': 0.9,
        'use_mutual_best': False, 'nms_radius': 4, 'keypoint_threshold': 0.005,
        'max_keypoints': -1, 'remove_borders': 4,
    }
    params['nonrigid_params'] = {
        'type': 'demons', 'iterations': [200,150,100,50],
        'smoothing_sigma': 3.0, 'update_field_sigma': 2.0, 'max_step_length': 5.0,
        'use_histogram_matching': True, 'use_symmetric_forces': True, 'use_gradient_type': 'symmetric'
    }
    params['multiresolution_params'] = {
        'levels': 5, 'shrink_factors': [16,8,4,2,1], 'smoothing_sigmas': [8.0,4.0,2.0,1.0,0.5]
    }
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information', 'number_of_bins': 32, 'optimizer': 'gradient_descent',
        'learning_rate': 2.0, 'min_step': 0.001, 'iterations': 500,
        'relaxation_factor': 0.8, 'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random', 'metric_sampling_percentage': 0.1
    }
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    params['save_displacement_field'] = True
    return params

def register_qptiff_direct(he_path: Path, if_path: Path, output_dir: Path, if_channels=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration"
    temp_dir = reg_dir / "TEMP"
    reg_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS
    results = {}

    with tifffile.TiffFile(if_path) as tif:
        if_data = tif.asarray()
        if if_data.ndim == 4:
            if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
        selected = []
        for ch in if_channels[:3]:
            ch_data = if_data[ch]
            p1, p99 = np.percentile(ch_data, [1, 99])
            ch_norm = np.clip((ch_data - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            selected.append(ch_norm)
        if_rgb = np.stack(selected, axis=-1)
        if_rgb_path = temp_dir / "if_rgb_temp.tiff"
        tifffile.imwrite(if_rgb_path, if_rgb, photometric='rgb', compression='lzw')
        target_shape = if_rgb.shape

    with tifffile.TiffFile(he_path) as tif:
        he_data = tif.pages[0].asarray()
        if he_data.ndim == 3 and he_data.shape[0] == 3:
            he_data = np.transpose(he_data, (1, 2, 0))
        if he_data.ndim == 2:
            he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
        if he_data.dtype != np.uint8:
            he_data = (he_data / 256).astype(np.uint8) if he_data.dtype == np.uint16 else he_data.astype(np.uint8)
        he_original = he_data.copy()
        if he_data.shape[:2] != target_shape[:2]:
            he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            he_original = cv2.resize(he_original, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        he_temp_path = temp_dir / "he_temp.tiff"
        tifffile.imwrite(he_temp_path, he_data, photometric='rgb', compression='lzw')

    he_prep, if_prep = preprocess_for_registration(he_data, if_rgb)
    he_prep_path = reg_dir / "he_preprocessed.tiff"
    if_prep_path = reg_dir / "if_preprocessed.tiff"
    tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw')
    tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw')

    config = {
        'source_path': str(he_prep_path), 'target_path': str(if_prep_path),
        'output_path': str(reg_dir), 'registration_parameters': create_registration_params(),
        'case_name': 'qptiff_reg', 'save_displacement_field': True,
        'copy_target': True, 'delete_temporary_results': False,
        'temporary_path': str(temp_dir)
    }
    start = time.time()
    deeperhistreg.run_registration(**config)
    results['elapsed_time'] = time.time() - start

    disp_field_path = temp_dir / 'displacement_field.npy'
    if not disp_field_path.exists():
        disp_field_path = reg_dir / 'displacement_field.mha'
        if not disp_field_path.exists():
            results['success'] = False
            results['error'] = 'Displacement field not found'
            return results

    if disp_field_path.suffix == '.npy':
        displacement_field = np.load(str(disp_field_path))
    else:
        disp_img = sitk.ReadImage(str(disp_field_path))
        displacement_field = sitk.GetArrayFromImage(disp_img)
        if displacement_field.shape[-1] == 2:
            displacement_field = displacement_field.transpose(2, 0, 1)

    h, w = target_shape[:2]
    flow = displacement_field.transpose(1, 2, 0)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = (x + flow[:, :, 0]).astype(np.float32), (y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(he_original, map_x, map_y, cv2.INTER_LINEAR)

    out_path = output_dir / "registered_HE_color.tiff"
    tifffile.imwrite(out_path, warped, photometric='rgb', compression='lzw', bigtiff=True)

    results['success'] = True
    results['registered_path'] = out_path
    results['output_shape'] = warped.shape
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--he-qptiff', type=str, required=True)
    parser.add_argument('--if-qptiff', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--if-channels', type=int, nargs='+', default=DEFAULT_IF_CHANNELS)
    args = parser.parse_args()

    he_path = Path(args.he_qptiff)
    if_path = Path(args.if_qptiff)
    output_dir = Path(args.output_dir)

    if not he_path.exists() or not if_path.exists():
        print("❌ Input files not found.")
        return 1

    if tc.cuda.is_available():
        print(f"✅ GPU detected: {tc.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not found. Expect slow performance.")

    results = register_qptiff_direct(he_path, if_path, output_dir, args.if_channels)

    if results.get('success'):
        print("\n✅ Registration complete")
        print(f"Output saved at: {results['registered_path']}")
        print(f"Dimensions: {results['output_shape']}")
    else:
        print(f"\n❌ Registration failed: {results.get('error')}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

