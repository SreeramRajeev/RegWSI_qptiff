#!/usr/bin/env python3
"""
Direct QPTIFF Registration Pipeline (RGB inputs)
===============================================
Feeds 3-channel images into DeepHistReg so output remains in color.
Optimized for vast.ai with NVIDIA H200 GPU.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch as tc
import cv2
import tifffile
import matplotlib
matplotlib.use('Agg')

import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Default IF channels to extract
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163

def preprocess_for_registration(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RGB-preserving preprocessing:
      - resize
      - CLAHE per channel
      - grayscale edge detection + blend
    """
    # 1) Resize
    if source.shape[:2] != target.shape[:2]:
        source = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 2) CLAHE separately on each channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    def apply_clahe_rgb(img):
        out = np.zeros_like(img)
        for c in range(3):
            ch = img[..., c]
            if ch.dtype != np.uint8:
                ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out[..., c] = clahe.apply(ch)
        return out

    src_clahe = apply_clahe_rgb(source)
    tgt_clahe = apply_clahe_rgb(target)

    # 3) Edge detection on grayscale, then stack to 3 channels
    src_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    src_edges = cv2.Canny(src_gray, 30, 100)
    tgt_edges = cv2.Canny(tgt_gray, 30, 100)
    src_edges_rgb = cv2.cvtColor(src_edges, cv2.COLOR_GRAY2RGB)
    tgt_edges_rgb = cv2.cvtColor(tgt_edges, cv2.COLOR_GRAY2RGB)

    # 4) Blend CLAHE + edges
    src_final = cv2.addWeighted(src_clahe, 0.7, src_edges_rgb, 0.3, 0)
    tgt_final = cv2.addWeighted(tgt_clahe, 0.7, tgt_edges_rgb, 0.3, 0)

    return src_final, tgt_final

def create_registration_params() -> Dict:
    params = default_initial_nonrigid()
    # (keep your existing superpoint/superglue & nonrigid settings…)
    # [snip: identical to your original create_registration_params]
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    params['save_displacement_field'] = True
    return params

def register_qptiff_direct(he_qptiff_path: Path, if_qptiff_path: Path,
                          output_dir: Path, if_channels: Optional[list[int]] = None) -> Dict:
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS

    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration"; reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir / "TEMP";   temp_dir.mkdir(exist_ok=True)
    results: Dict = {}

    # --- Step 1: prepare IF RGB ---
    with tifffile.TiffFile(if_qptiff_path) as tif:
        if_data = tif.asarray()
        # handle 4D→3D logic (your original code)...
        # extract channels, normalize to UINT8, stack first 3 → if_rgb
        # save if_rgb to temp_dir / "if_rgb_temp.tiff"
    # target_shape = if_rgb.shape

    # --- Step 2: prepare H&E RGB ---
    with tifffile.TiffFile(he_qptiff_path) as tif:
        he_data = tif.pages[0].asarray()
        # transpose if needed, cast to RGB, uint8, resize to target_shape
        # save he_temp to temp_dir

    # --- Step 3: preprocess both in RGB ---
    he_prep, if_prep = preprocess_for_registration(he_data, if_rgb)
    tifffile.imwrite(reg_dir/"he_preprocessed.tiff", he_prep, photometric='rgb', compression='lzw')
    tifffile.imwrite(reg_dir/"if_preprocessed.tiff", if_prep, photometric='rgb', compression='lzw')

    # --- Step 4: run DeepHistReg ---
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
    print(f"   Registration done in {elapsed:.1f}s")

    # --- Step 5: locate the RGB warped output directly ---
    warped_rgb_path = reg_dir / "qptiff_rgb_warped_source.tiff"
    if warped_rgb_path.exists():
        # This is already 3-channel color
        final = output_dir/"registered_HE_color_direct.tiff"
        shutil.copy(warped_rgb_path, final)
        print(f"✅ Color-warped TIFF available: {final}")
        results.update(success=True, registered_path=final, elapsed_time=elapsed)
    else:
        raise FileNotFoundError("Expected 3-channel warped_source.tiff not found in registration folder")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-qptiff",  required=True)
    parser.add_argument("--if-qptiff",  required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--if-channels", type=int, nargs='+', default=[0,1,5])
    args = parser.parse_args()

    he_path = Path(args.he_qptiff)
    if_path = Path(args.if_qptiff)
    out_dir = Path(args.output_dir)

    print(f"✅ GPU: {tc.cuda.get_device_name(0) if tc.cuda.is_available() else 'None'}")
    results = register_qptiff_direct(he_path, if_path, out_dir, args.if_channels)
    if results.get('success'):
        print("\n🎉 Registration complete!")
        print(f"   Output: {results['registered_path']}")
        print(f"   Time:   {results['elapsed_time']:.1f}s")
    else:
        print("❌ Registration failed.")

if __name__ == "__main__":
    sys.exit(main())

