#!/usr/bin/env python3
"""
Proven QPTIFF Registration Pipeline
===================================
Based on your successful approach from 2 days ago
Fixes pad_value error while maintaining excellent alignment quality
Works with H200 and large images
"""

### System Imports ###
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, Optional, Dict

### Scientific Computing ###
import numpy as np
import torch as tc
from scipy.ndimage import map_coordinates

### Image Processing ###
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

### DeepHistReg ###
import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

# For handling .mha files
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

#############################################################################
# CONFIGURATION
#############################################################################

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Default IF channels to extract
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163

#############################################################################
# UTILITY FUNCTIONS
#############################################################################

def save_tiff_safe(filepath, image, **kwargs):
    """
    Save TIFF with automatic BigTIFF detection for large files
    """
    # Calculate approximate file size
    size_estimate = image.nbytes
    
    # Use BigTIFF for files > 3GB (conservative threshold)
    if size_estimate > 3 * 1024**3:
        kwargs['bigtiff'] = True
        print(f"   Using BigTIFF for large file ({size_estimate/1e9:.1f} GB)")
    
    tifffile.imwrite(filepath, image, **kwargs)

#############################################################################
# PROVEN PREPROCESSING (Your Successful Approach)
#############################################################################

def preprocess_for_registration_proven(source, target):
    """
    Proven preprocessing approach that gave you excellent results
    Conservative enhancement that doesn't trigger DeepHistReg pad_value issues
    """
    print("  Applying proven preprocessing...")
    
    # Ensure same dimensions
    if source.shape[:2] != target.shape[:2]:
        print(f"  Resizing source from {source.shape} to match target {target.shape}")
        source = cv2.resize(source, (target.shape[1], target.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    # Ensure both are 3-channel RGB uint8
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    
    source = source.astype(np.uint8)
    target = target.astype(np.uint8)
    
    # Method that worked for you: Conservative enhancement per channel
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    # Apply light Gaussian blur for noise reduction
    source_blur = cv2.GaussianBlur(source, (5, 5), 1.0)
    target_blur = cv2.GaussianBlur(target, (5, 5), 1.0)
    
    # Apply CLAHE per channel with conservative settings
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    
    for channel in range(3):
        source_enhanced[:, :, channel] = clahe.apply(source_blur[:, :, channel])
        target_enhanced[:, :, channel] = clahe.apply(target_blur[:, :, channel])
    
    # Light edge enhancement using gradients (not Canny to avoid pad_value issues)
    source_edges = np.zeros_like(source_enhanced[:, :, 0])
    target_edges = np.zeros_like(target_enhanced[:, :, 0])
    
    # Simple gradient-based edge detection per channel
    for channel in range(3):
        # X and Y gradients
        grad_x = cv2.Sobel(source_enhanced[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(source_enhanced[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
        source_grad = np.sqrt(grad_x**2 + grad_y**2)
        source_edges = np.maximum(source_edges, source_grad)
        
        grad_x = cv2.Sobel(target_enhanced[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(target_enhanced[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
        target_grad = np.sqrt(grad_x**2 + grad_y**2)
        target_edges = np.maximum(target_edges, target_grad)
    
    # Normalize edges
    if source_edges.max() > 0:
        source_edges = (source_edges / source_edges.max() * 255).astype(np.uint8)
    if target_edges.max() > 0:
        target_edges = (target_edges / target_edges.max() * 255).astype(np.uint8)
    
    # Combine enhanced color with edge information (your successful approach)
    source_final = np.zeros_like(source_enhanced)
    target_final = np.zeros_like(target_enhanced)
    
    for channel in range(3):
        source_final[:, :, channel] = cv2.addWeighted(
            source_enhanced[:, :, channel], 0.7, 
            source_edges, 0.3, 0
        )
        target_final[:, :, channel] = cv2.addWeighted(
            target_enhanced[:, :, channel], 0.7, 
            target_edges, 0.3, 0
        )
    
    return source_final, target_final

def create_proven_registration_params():
    """Your proven registration parameters that gave excellent results"""
    params = default_initial_nonrigid()
    
    # Feature-based initial alignment (your successful settings)
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 10.0,
        'max_features': 10000,
        'match_ratio': 0.9,
        'use_mutual_best': False,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    # Nonrigid parameters (your successful settings)
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [200, 150, 100, 50],
        'smoothing_sigma': 3.0,
        'update_field_sigma': 2.0,
        'max_step_length': 5.0,
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric',
    }
    
    # Multi-resolution (your successful settings)
    params['multiresolution_params'] = {
        'levels': 5,
        'shrink_factors': [16, 8, 4, 2, 1],
        'smoothing_sigmas': [8.0, 4.0, 2.0, 1.0, 0.5],
    }
    
    # Optimization (your successful settings)
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
    
    # TIFF support
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    
    # Ensure displacement field is saved
    params['save_displacement_field'] = True
    
    return params

#############################################################################
# ROBUST DISPLACEMENT FIELD HANDLING
#############################################################################

def find_displacement_field_comprehensive(reg_dir, temp_dir, case_name='proven_reg'):
    """
    Comprehensive search for displacement field in all possible locations
    """
    search_dirs = [
        temp_dir,
        reg_dir,
        reg_dir / 'TEMP',
        temp_dir / case_name,
        reg_dir / case_name,
        temp_dir / case_name / 'Results_Final',
        reg_dir / case_name / 'Results_Final',
    ]
    
    search_patterns = [
        'displacement_field.*',
        f'{case_name}_displacement_field.*',
        'deformation_field.*',
        '*displacement*.*',
        '*deformation*.*',
    ]
    
    print(f"   Searching for displacement field...")
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for pattern in search_patterns:
            matches = list(search_dir.glob(pattern))
            for match in matches:
                if match.is_file() and match.stat().st_size > 1000:  # Must be substantial file
                    print(f"   Found displacement field: {match}")
                    return match
    
    # Last resort: list all files for debugging
    print("   No displacement field found. Available files:")
    for search_dir in search_dirs:
        if search_dir.exists():
            print(f"   {search_dir}:")
            for item in search_dir.iterdir():
                if item.is_file():
                    print(f"     {item.name} ({item.stat().st_size} bytes)")
    
    return None

def load_displacement_field_robust(filepath):
    """Load displacement field from various formats"""
    filepath = Path(filepath)
    print(f"   Loading displacement field: {filepath.name}")
    
    if filepath.suffix == '.npy':
        return np.load(str(filepath))
    
    elif filepath.suffix == '.mha':
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK required for .mha files")
        
        displacement_image = sitk.ReadImage(str(filepath))
        displacement_array = sitk.GetArrayFromImage(displacement_image)
        
        # Handle different array formats
        if displacement_array.ndim == 4:
            displacement_array = displacement_array[0]
        
        if displacement_array.shape[-1] == 2:
            return displacement_array
        elif displacement_array.shape[0] == 2:
            return displacement_array.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected displacement field shape: {displacement_array.shape}")
    
    else:
        raise ValueError(f"Unsupported displacement field format: {filepath.suffix}")

def apply_displacement_field_chunked(image, displacement_field, chunk_size=16384):
    """
    Apply displacement field with chunked processing for large images
    Handles H200 memory efficiently
    """
    h, w = image.shape[:2]
    
    # Resize displacement field if needed
    if displacement_field.shape[:2] != (h, w):
        print(f"   Resizing displacement field from {displacement_field.shape} to {(h, w, 2)}")
        displacement_field = cv2.resize(displacement_field, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Check if we need chunked processing (OpenCV size limit)
    max_opencv_size = 32767
    
    if h <= max_opencv_size and w <= max_opencv_size:
        print(f"   Applying displacement field directly (size: {h}x{w})")
        return _apply_displacement_direct(image, displacement_field)
    else:
        print(f"   Image too large for OpenCV ({h}x{w}), using chunked processing")
        return _apply_displacement_chunked(image, displacement_field, chunk_size)

def _apply_displacement_direct(image, displacement_field):
    """Direct application for smaller images"""
    h, w = image.shape[:2]
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + displacement_field[:, :, 0]).astype(np.float32)
    map_y = (y + displacement_field[:, :, 1]).astype(np.float32)
    
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def _apply_displacement_chunked(image, displacement_field, chunk_size):
    """Chunked processing for large images"""
    h, w = image.shape[:2]
    warped = np.zeros_like(image)
    
    h_chunks = (h + chunk_size - 1) // chunk_size
    w_chunks = (w + chunk_size - 1) // chunk_size
    
    print(f"   Processing in {h_chunks}x{w_chunks} = {h_chunks * w_chunks} chunks")
    
    for i in range(h_chunks):
        for j in range(w_chunks):
            h_start = i * chunk_size
            h_end = min((i + 1) * chunk_size, h)
            w_start = j * chunk_size
            w_end = min((j + 1) * chunk_size, w)
            
            if (i * w_chunks + j) % 5 == 0:
                print(f"   Processing chunk {i * w_chunks + j + 1}/{h_chunks * w_chunks}")
            
            # Extract displacement for this chunk
            disp_chunk = displacement_field[h_start:h_end, w_start:w_end]
            
            # Create coordinates for sampling
            y_coords, x_coords = np.mgrid[h_start:h_end, w_start:w_end]
            sample_x = np.clip(x_coords + disp_chunk[:, :, 0], 0, w - 1)
            sample_y = np.clip(y_coords + disp_chunk[:, :, 1], 0, h - 1)
            
            # Sample from full image
            try:
                if image.ndim == 3:
                    for c in range(image.shape[2]):
                        warped[h_start:h_end, w_start:w_end, c] = map_coordinates(
                            image[:, :, c], [sample_y, sample_x], 
                            order=1, mode='reflect', prefilter=False
                        )
                else:
                    warped[h_start:h_end, w_start:w_end] = map_coordinates(
                        image, [sample_y, sample_x], 
                        order=1, mode='reflect', prefilter=False
                    )
            except Exception as e:
                print(f"   Warning: Error in chunk ({i},{j}): {e}")
                warped[h_start:h_end, w_start:w_end] = image[h_start:h_end, w_start:w_end]
    
    return warped.astype(image.dtype)

#############################################################################
# MAIN REGISTRATION FUNCTION
#############################################################################

def register_qptiff_proven(he_qptiff_path: Path, if_qptiff_path: Path, 
                          output_dir: Path, if_channels: list = None) -> Dict:
    """
    Proven QPTIFF registration that gave you excellent results
    Fixes pad_value error while maintaining alignment quality
    """
    print("\n" + "="*70)
    print(" PROVEN QPTIFF REGISTRATION - EXCELLENT ALIGNMENT")
    print("="*70)
    
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration_proven"
    reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir / "TEMP"
    temp_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Prepare IF image
        print("\n1. Preparing IF image...")
        
        import tifffile
        
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            if if_data.ndim == 3 and if_data.shape[0] <= 16:
                selected = []
                for ch_idx in if_channels[:3]:
                    if ch_idx < if_data.shape[0]:
                        ch = if_data[ch_idx]
                        # Robust percentile normalization
                        p1, p99 = np.percentile(ch, [1, 99])
                        ch_norm = np.clip((ch - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                        selected.append(ch_norm)
                
                if_rgb = np.stack(selected[:3], axis=-1)
                if if_rgb.shape[-1] < 3:
                    padding = np.zeros((*if_rgb.shape[:2], 3 - if_rgb.shape[-1]), dtype=np.uint8)
                    if_rgb = np.concatenate([if_rgb, padding], axis=-1)
                
                print(f"   IF RGB shape: {if_rgb.shape}")
                
                # Save IF RGB
                if_rgb_path = temp_dir / "if_rgb.tiff"
                save_tiff_safe(if_rgb_path, if_rgb, photometric='rgb', compression='lzw')
                target_shape = if_rgb.shape
        
        # Step 2: Load H&E image
        print("\n2. Loading H&E image...")
        
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_data = tif.pages[0].asarray()
            print(f"   H&E shape: {he_data.shape}")
            
            if he_data.ndim == 3 and he_data.shape[0] == 3:
                he_data = np.transpose(he_data, (1, 2, 0))
            
            if he_data.ndim == 2:
                he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
            
            if he_data.dtype != np.uint8:
                if he_data.dtype == np.uint16:
                    he_data = (he_data / 256).astype(np.uint8)
                else:
                    he_data = he_data.astype(np.uint8)
            
            # Store original for final transformation
            he_original = he_data.copy()
            
            # Resize to match IF
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E from {he_data.shape} to match IF {target_shape}")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original = cv2.resize(he_original, (target_shape[1], target_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            
            # Save original H&E
            he_orig_path = temp_dir / "he_original.tiff"
            save_tiff_safe(he_orig_path, he_original, photometric='rgb', compression='lzw')
        
        # Step 3: Apply proven preprocessing
        print("\n3. Applying proven preprocessing...")
        he_prep, if_prep = preprocess_for_registration_proven(he_data, if_rgb)
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed.tiff"
        if_prep_path = reg_dir / "if_preprocessed.tiff"
        save_tiff_safe(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        save_tiff_safe(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        # Step 4: Run DeepHistReg with proven parameters
        print("\n4. Running DeepHistReg with proven parameters...")
        
        params = create_proven_registration_params()
        case_name = 'proven_reg'  # Simple case name
        
        config = {
            'source_path': str(he_prep_path),
            'target_path': str(if_prep_path),
            'output_path': str(reg_dir),
            'registration_parameters': params,
            'case_name': case_name,
            'save_displacement_field': True,
            'copy_target': True,
            'delete_temporary_results': False,
            'temporary_path': str(temp_dir)
        }
        
        start_time = time.time()
        deeperhistreg.run_registration(**config)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Registration completed in {elapsed:.1f} seconds")
        
        # Step 5: Find and apply displacement field
        print("\n5. Finding and applying displacement field...")
        
        disp_field_path = find_displacement_field_comprehensive(reg_dir, temp_dir, case_name)
        
        if disp_field_path:
            try:
                displacement_field = load_displacement_field_robust(disp_field_path)
                print(f"   Displacement field shape: {displacement_field.shape}")
                
                # Apply to original H&E
                warped_he = apply_displacement_field_chunked(he_original, displacement_field)
                
                # Save result
                final_output_path = output_dir / "registered_HE_proven.tiff"
                tifffile.imwrite(
                    final_output_path,
                    warped_he,
                    photometric='rgb',
                    compression='lzw',
                    bigtiff=True
                )
                
                print(f"   ✅ Registration saved: {final_output_path.name}")
                print(f"   Output dimensions: {warped_he.shape}")
                
                results['success'] = True
                results['registered_path'] = final_output_path
                results['displacement_field'] = disp_field_path
                results['elapsed_time'] = elapsed
                results['output_shape'] = warped_he.shape
                
                # Create visualizations
                print("\n6. Creating visualizations...")
                try:
                    create_visualizations(he_original, if_rgb, warped_he, output_dir)
                except Exception as e:
                    print(f"   Warning: Visualization failed: {e}")
                
            except Exception as e:
                print(f"   ❌ Error applying displacement field: {e}")
                import traceback
                traceback.print_exc()
                results['success'] = False
                results['error'] = f"Displacement field application failed: {e}"
        
        else:
            print("   ❌ No displacement field found!")
            results['success'] = False
            results['error'] = "Displacement field not found"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results

def create_visualizations(he_original, if_img, warped, output_dir):
    """Create quality check visualizations"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Ensure all images have same shape
    h, w = if_img.shape[:2]
    if he_original.shape[:2] != (h, w):
        he_original = cv2.resize(he_original, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # Side-by-side comparison
    comparison = np.hstack([he_original, if_img, warped])
    cv2.imwrite(str(viz_dir / "side_by_side.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Checkerboard overlay
    checker_size = max(100, min(h, w) // 20)
    checkerboard = np.zeros_like(if_img)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = warped[i:i+checker_size, j:j+checker_size]
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = if_img[i:i+checker_size, j:j+checker_size]
    
    cv2.imwrite(str(viz_dir / "checkerboard.jpg"), cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # Overlay blend
    overlay = cv2.addWeighted(if_img, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(viz_dir / "overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"   ✅ Visualizations saved to: {viz_dir}")

def evaluate_registration_quality(if_path, warped_path):
    """Calculate registration quality metrics"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        if_img = cv2.imread(str(if_path))
        warped_img = cv2.imread(str(warped_path))
        
        if if_img is None or warped_img is None:
            return None
        
        # Convert to grayscale
        if_gray = cv2.cvtColor(if_img, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        overall_ssim = ssim(if_gray, warped_gray)
        
        print(f"\n   Registration Quality:")
        print(f"   Overall SSIM: {overall_ssim:.4f}")
        
        return overall_ssim
    except Exception as e:
        print(f"   Could not calculate quality metrics: {e}")
        return None

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution"""
    import argparse
    
    print("\n" + "="*70)
    print(" PROVEN QPTIFF REGISTRATION - NO PAD_VALUE ERROR")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Proven QPTIFF registration with excellent alignment")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--if-channels", type=int, nargs='+', default=[0, 1, 5], 
                       help="IF channels to use (default: 0=DAPI, 1=CD8, 5=CD163)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    he_path = Path(args.he_qptiff)
    if_path = Path(args.if_qptiff)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not he_path.exists():
        print(f"❌ H&E file not found: {he_path}")
        return 1
    
    if not if_path.exists():
        print(f"❌ IF file not found: {if_path}")
        return 1
    
    print(f"\nInput files:")
    print(f"  H&E: {he_path.name} ({he_path.stat().st_size/1e9:.2f} GB)")
    print(f"  IF:  {if_path.name} ({if_path.stat().st_size/1e9:.2f} GB)")
    print(f"  Output: {output_dir}")
    
    # Check GPU
    if tc.cuda.is_available():
        print(f"\n✅ GPU available: {tc.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU detected - registration will be slower")
    
    # Run registration
    results = register_qptiff_proven(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        print(f"   Time: {results.get('elapsed_time', 0):.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        print(f"   Dimensions: {results.get('output_shape', 'unknown')}")
        
        # Evaluate quality
        if_temp = output_dir / "registration_proven" / "TEMP" / "if_rgb.tiff"
        if if_temp.exists() and results['registered_path'].exists():
            evaluate_registration_quality(if_temp, results['registered_path'])
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
