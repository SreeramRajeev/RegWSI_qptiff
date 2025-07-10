#!/usr/bin/env python3
"""
Direct QPTIFF Registration Pipeline (RGB inputs, dynamic warped discovery)
Feeds 3-channel RGB into DeepHistReg so the warped output stays in color.
"""

import os, sys, time, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch as tc
import cv2
import tifffile
import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163

def preprocess_for_registration(src: np.ndarray, tgt: np.ndarray):
    """
    Preserve RGB: CLAHE per-channel + blend in edges.
    """
    # Resize src→tgt
    if src.shape[:2] != tgt.shape[:2]:
        src = cv2.resize(src, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # CLAHE on each channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    def clahe_rgb(img):
        out = np.zeros_like(img)
        for c in range(3):
            ch = img[..., c]
            if ch.dtype != np.uint8:
                ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out[..., c] = clahe.apply(ch)
        return out

    src_c = clahe_rgb(src)
    tgt_c = clahe_rgb(tgt)

    # Edges from grayscale
    src_g = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    tgt_g = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    src_e = cv2.Canny(src_g, 30, 100)
    tgt_e = cv2.Canny(tgt_g, 30, 100)
    src_e3 = cv2.cvtColor(src_e, cv2.COLOR_GRAY2RGB)
    tgt_e3 = cv2.cvtColor(tgt_e, cv2.COLOR_GRAY2RGB)

    # Blend
    src_final = cv2.addWeighted(src_c, 0.7, src_e3, 0.3, 0)
    tgt_final = cv2.addWeighted(tgt_c, 0.7, tgt_e3, 0.3, 0)
    return src_final, tgt_final

def create_registration_params() -> Dict:
    params = default_initial_nonrigid()
    # ---- your original feature + nonrigid + multires + optimization params here ----
    params['initial_alignment_params'] = {
        'type': 'feature_based', 'detector': 'superpoint', 'matcher': 'superglue',
        'ransac_threshold': 10.0, 'max_features': 10000, 'match_ratio': 0.9,
        'use_mutual_best': False, 'nms_radius': 4, 'keypoint_threshold': 0.005,
        'max_keypoints': -1, 'remove_borders': 4,
    }
    params['nonrigid_params'] = {
        'type': 'demons', 'iterations': [200,150,100,50],
        'smoothing_sigma': 3.0, 'update_field_sigma': 2.0,
        'max_step_length': 5.0, 'use_histogram_matching': True,
        'use_symmetric_forces': True, 'use_gradient_type': 'symmetric',
    }
    params['multiresolution_params'] = {
        'levels': 5,
        'shrink_factors': [16,8,4,2,1],
        'smoothing_sigmas': [8.0,4.0,2.0,1.0,0.5],
    }
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 32,
        'optimizer': 'gradient_descent',
        'learning_rate': 2.0,
        'min_step': 0.001,
        'iterations': 500,
        'relaxation_factor': 0.8,
        'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.1,
    }
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    params['save_displacement_field'] = True
    return params

def register_qptiff_direct(
    he_qptiff_path: Path,
    if_qptiff_path: Path,
    output_dir: Path,
    if_channels: Optional[list[int]] = None
) -> Dict:
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS

    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir  = output_dir/"registration"; reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir/"TEMP";    temp_dir.mkdir(exist_ok=True)
    results: Dict = {}

    # Step 1: load & extract IF→RGB
    with tifffile.TiffFile(if_qptiff_path) as tif:
        if_data = tif.asarray()
    if if_data.ndim == 4:
        if if_data.shape[0] < if_data.shape[1]:
            if_data = if_data[0]
        else:
            if_data = if_data[:,0]
    if if_data.ndim == 3 and if_data.shape[0] >= max(if_channels)+1:
        chans = []
        for ch in if_channels[:3]:
            plane = if_data[ch]
            p1, p99 = np.percentile(plane, [1,99])
            norm = np.clip((plane - p1)/(p99-p1)*255,0,255).astype(np.uint8)
            chans.append(norm)
        if_rgb = np.stack(chans, axis=-1)
    else:
        if_rgb = if_data if if_data.ndim==3 else cv2.cvtColor(if_data, cv2.COLOR_GRAY2RGB)
    tifffile.imwrite(temp_dir/"if_rgb_temp.tiff", if_rgb, photometric='rgb', compression='lzw')
    target_shape = if_rgb.shape

    # Step 2: load H&E original & preprocess RGB
    with tifffile.TiffFile(he_qptiff_path) as tif:
        he_page = tif.pages[0]
        he_data = he_page.asarray()
    if he_data.ndim==3 and he_data.shape[0]==3:
        he_data = np.transpose(he_data, (1,2,0))
    if he_data.ndim==2:
        he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
    if he_data.dtype!=np.uint8:
        he_data = (he_data/np.max(he_data)*255).astype(np.uint8)
    if he_data.shape[:2] != target_shape[:2]:
        he_data = cv2.resize(he_data, (target_shape[1],target_shape[0]), interpolation=cv2.INTER_LINEAR)
    tifffile.imwrite(temp_dir/"he_rgb_temp.tiff", he_data, photometric='rgb', compression='lzw')

    # Step 3: RGB preprocessing for registration
    he_prep, if_prep = preprocess_for_registration(he_data, if_rgb)
    tifffile.imwrite(reg_dir/"he_preprocessed.tiff", he_prep, photometric='rgb', compression='lzw')
    tifffile.imwrite(reg_dir/"if_preprocessed.tiff", if_prep, photometric='rgb', compression='lzw')

    # Step 4: run DeepHistReg
    params = create_registration_params()
    config = {
        'source_path': str(reg_dir/"he_preprocessed.tiff"),
        'target_path': str(reg_dir/"if_preprocessed.tiff"),
        'output_path': str(reg_dir),
        'registration_parameters': params,
        'case_name': 'qptiff_rgb',
        'save_displacement_field': True,
        'copy_target': True,
        'delete_temporary_results': False,
        'temporary_path': str(temp_dir)
    }
    start = time.time()
    deeperhistreg.run_registration(**config)
    elapsed = time.time() - start
    print(f"   ✔ DeepHistReg took {elapsed:.1f}s")

    # Step 5: discover & copy the 3-channel warped TIFF
    warped_candidates = list(reg_dir.glob("**/*warped*.tif*"))
    if not warped_candidates:
        raise FileNotFoundError(f"No warped TIFF found in {reg_dir}")
    print("   🔍 Found warped files:")
    for p in warped_candidates:
        print("    -", p)
    warped_rgb_path = None
    for p in warped_candidates:
        img = tifffile.imread(str(p))
        if img.ndim == 3 and img.shape[2] == 3:
            warped_rgb_path = p
            break
    if warped_rgb_path is None:
        raise RuntimeError("Found warped files but none are 3-channel RGB")
    final_out = output_dir/"registered_HE_color_direct.tiff"
    shutil.copy(str(warped_rgb_path), str(final_out))
    print(f"✅ Color result copied from {warped_rgb_path.name} → {final_out.name}")
    results.update(success=True, registered_path=str(final_out), elapsed_time=elapsed)
    return results

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--he-qptiff", required=True)
    p.add_argument("--if-qptiff", required=True)
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--if-channels", type=int, nargs="+", default=[0,1,5])
    args = p.parse_args()

    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {tc.cuda.get_device_name(0) if tc.cuda.is_available() else 'None'}")
    out = register_qptiff_direct(
        Path(args.he_qptiff),
        Path(args.if_qptiff),
        Path(args.output_dir),
        args.if_channels
    )
    if out.get("success"):
        print("\n🎉 Registration complete!")
        print(f" Output: {out['registered_path']}")
        print(f" Time:   {out['elapsed_time']:.1f}s")
        sys.exit(0)
    else:
        print("❌ Registration failed.")
        sys.exit(1)

if __name__=="__main__":
    main()

