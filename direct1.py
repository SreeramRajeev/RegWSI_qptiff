#!/usr/bin/env python3
"""
Fine-Tuned QPTIFF Registration Pipeline - Maximum Precision
===========================================================
Optimized parameters for excellent alignment quality
Enhanced preprocessing and registration settings
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
# ENHANCED PREPROCESSING FOR MAXIMUM ALIGNMENT
#############################################################################

def preprocess_for_registration_enhanced(source, target):
    """
    Enhanced preprocessing for maximum alignment quality
    Stronger feature enhancement while avoiding pad_value issues
    """
    print("  Applying enhanced preprocessing for maximum alignment...")
    
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
    
    # Step 1: Multi-scale preprocessing
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    # Apply progressive Gaussian blur for multi-scale features
    source_blur1 = cv2.GaussianBlur(source, (3, 3), 0.8)  # Fine features
    source_blur2 = cv2.GaussianBlur(source, (7, 7), 1.5)  # Coarse features
    target_blur1 = cv2.GaussianBlur(target, (3, 3), 0.8)
    target_blur2 = cv2.GaussianBlur(target, (7, 7), 1.5)
    
    # Combine multi-scale information
    source_combined = cv2.addWeighted(source_blur1, 0.7, source_blur2, 0.3, 0)
    target_combined = cv2.addWeighted(target_blur1, 0.7, target_blur2, 0.3, 0)
    
    # Step 2: Enhanced CLAHE per channel with stronger settings
    clahe_fine = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))   # Stronger contrast
    clahe_coarse = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20)) # Smoother regions
    
    for channel in range(3):
        # Apply two-level CLAHE
        source_fine = clahe_fine.apply(source_combined[:, :, channel])
        source_coarse = clahe_coarse.apply(source_combined[:, :, channel])
        source_enhanced[:, :, channel] = cv2.addWeighted(source_fine, 0.8, source_coarse, 0.2, 0)
        
        target_fine = clahe_fine.apply(target_combined[:, :, channel])
        target_coarse = clahe_coarse.apply(target_combined[:, :, channel])
        target_enhanced[:, :, channel] = cv2.addWeighted(target_fine, 0.8, target_coarse, 0.2, 0)
    
    # Step 3: Multi-direction edge enhancement using Sobel
    source_edges = np.zeros_like(source_enhanced[:, :, 0], dtype=np.float64)
    target_edges = np.zeros_like(target_enhanced[:, :, 0], dtype=np.float64)
    
    # Multiple Sobel kernel sizes for different edge scales
    sobel_kernels = [3, 5]  # Different scales
    
    for kernel_size in sobel_kernels:
        for channel in range(3):
            # X and Y gradients
            grad_x = cv2.Sobel(source_enhanced[:, :, channel], cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(source_enhanced[:, :, channel], cv2.CV_64F, 0, 1, ksize=kernel_size)
            source_grad = np.sqrt(grad_x**2 + grad_y**2)
            source_edges = np.maximum(source_edges, source_grad)
            
            grad_x = cv2.Sobel(target_enhanced[:, :, channel], cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(target_enhanced[:, :, channel], cv2.CV_64F, 0, 1, ksize=kernel_size)
            target_grad = np.sqrt(grad_x**2 + grad_y**2)
            target_edges = np.maximum(target_edges, target_grad)
    
    # Normalize edges with better dynamic range
    if source_edges.max() > 0:
        source_edges = (source_edges / source_edges.max() * 255).astype(np.uint8)
        # Apply mild smoothing to edges
        source_edges = cv2.GaussianBlur(source_edges, (3, 3), 0.5)
    if target_edges.max() > 0:
        target_edges = (target_edges / target_edges.max() * 255).astype(np.uint8)
        target_edges = cv2.GaussianBlur(target_edges, (3, 3), 0.5)
    
    # Step 4: Optimal combination of enhanced color and edges
    source_final = np.zeros_like(source_enhanced)
    target_final = np.zeros_like(target_enhanced)
    
    # Stronger edge emphasis for better feature matching
    for channel in range(3):
        source_final[:, :, channel] = cv2.addWeighted(
            source_enhanced[:, :, channel], 0.6,  # Color information
            source_edges, 0.4, 0                  # Edge information (stronger)
        )
        target_final[:, :, channel] = cv2.addWeighted(
            target_enhanced[:, :, channel], 0.6,
            target_edges, 0.4, 0
        )
    
    # Step 5: Final smoothing to reduce noise while preserving features
    source_final = cv2.bilateralFilter(source_final, 5, 50, 50)
    target_final = cv2.bilateralFilter(target_final, 5, 50, 50)
    
    return source_final, target_final

def create_finetuned_registration_params():
    """
    Fine-tuned registration parameters for maximum precision
    """
    params = default_initial_nonrigid()
    
    # Enhanced feature-based initial alignment
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 8.0,        # Stricter threshold
        'max_features': 15000,          # More features
        'match_ratio': 0.82,            # More selective matching
        'use_mutual_best': True,        # Bidirectional matching
        'nms_radius': 3,                # Smaller NMS radius for more features
        'keypoint_threshold': 0.003,    # Lower threshold for more keypoints
        'max_keypoints': -1,
        'remove_borders': 6,            # Larger border removal
    }
    
    # Enhanced nonrigid parameters for finer alignment
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [400, 300, 200, 150, 100],  # More iterations at each level
        'smoothing_sigma': 2.5,                   # Slightly less smoothing for finer details
        'update_field_sigma': 1.8,                # Smaller updates for precision
        'max_step_length': 4.0,                   # Smaller steps
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric',
    }
    
    # Enhanced multi-resolution with more levels
    params['multiresolution_params'] = {
        'levels': 6,                                        # More levels for finer detail
        'shrink_factors': [32, 16, 8, 4, 2, 1],           # Finer progression
        'smoothing_sigmas': [12.0, 6.0, 3.0, 1.5, 0.75, 0.25],  # Finer smoothing progression
    }
    
    # Enhanced optimization for better convergence
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 64,                    # More bins for finer matching
        'optimizer': 'gradient_descent',
        'learning_rate': 1.5,                    # Slightly slower learning
        'min_step': 0.0005,                      # Smaller minimum step
        'iterations': 800,                       # More optimization iterations
        'relaxation_factor': 0.75,               # More relaxation
        'gradient_magnitude_tolerance': 5e-7,    # Tighter tolerance
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.2,       # More sampling for better statistics
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

def find_displacement_field_comprehensive(reg_dir, temp_dir, case_name='finetuned_reg'):
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
                if match.is_file() and match.stat().st_size > 1000:
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
    """
    h, w = image.shape[:2]
    
    # Resize displacement field if needed
    if displacement_field.shape[:2] != (h, w):
        print(f"   Resizing displacement field from {displacement_field.shape} to {(h, w, 2)}")
        displacement_field = cv2.resize(displacement_field, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Check if we need chunked processing
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

def register_qptiff_finetuned(he_qptiff_path: Path, if_qptiff_path: Path, 
                             output_dir: Path, if_channels: list = None) -> Dict:
    """
    Fine-tuned QPTIFF registration for maximum precision alignment
    """
    print("\n" + "="*70)
    print(" FINE-TUNED QPTIFF REGISTRATION - MAXIMUM PRECISION")
    print("="*70)
    
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration_finetuned"
    reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir / "TEMP"
    temp_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Prepare IF image
        print("\n1. Preparing IF image...")
        
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
                        # Enhanced percentile normalization
                        p0p5, p99p5 = np.percentile(ch, [0.5, 99.5])  # Tighter percentiles
                        ch_norm = np.clip((ch - p0p5) / (p99p5 - p0p5) * 255, 0, 255).astype(np.uint8)
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
        
        # Step 3: Apply enhanced preprocessing
        print("\n3. Applying enhanced preprocessing...")
        he_prep, if_prep = preprocess_for_registration_enhanced(he_data, if_rgb)
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed.tiff"
        if_prep_path = reg_dir / "if_preprocessed.tiff"
        save_tiff_safe(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        save_tiff_safe(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        # Step 4: Run DeepHistReg with fine-tuned parameters
        print("\n4. Running DeepHistReg with fine-tuned parameters...")
        print("   Using enhanced settings: 6 levels, 400+ iterations, 15k features")
        
        params = create_finetuned_registration_params()
        case_name = 'finetuned_reg'
        
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
        
        print(f"   ✅ Fine-tuned registration completed in {elapsed:.1f} seconds")
        
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
                final_output_path = output_dir / "registered_HE_finetuned.tiff"
                save_tiff_safe(
                    final_output_path,
                    warped_he,
                    photometric='rgb',
                    compression='lzw'
                )
                
                print(f"   ✅ Fine-tuned registration saved: {final_output_path.name}")
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
    viz_dir = output_dir / "visualizations_finetuned"
    viz_dir.mkdir(exist_ok=True)
    
    # Ensure all images have same shape
    h, w = if_img.shape[:2]
    if he_original.shape[:2] != (h, w):
        he_original = cv2.resize(he_original, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # Side-by-side comparison
    comparison = np.hstack([he_original, if_img, warped])
    cv2.imwrite(str(viz_dir / "side_by_side_finetuned.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Detailed checkerboard with smaller squares
    checker_size = max(150, min(h, w) // 30)  # Smaller for finer detail
    checkerboard = np.zeros_like(if_img)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = warped[i:i+checker_size, j:j+checker_size]
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = if_img[i:i+checker_size, j:j+checker_size]
    
    cv2.imwrite(str(viz_dir / "checkerboard_finetuned.jpg"), cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # Overlay blend
    overlay = cv2.addWeighted(if_img, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(viz_dir / "overlay_finetuned.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"   ✅ Fine-tuned visualizations saved to: {viz_dir}")

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution"""
    import argparse
    
    print("\n" + "="*70)
    print(" FINE-TUNED QPTIFF REGISTRATION - MAXIMUM PRECISION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Fine-tuned QPTIFF registration for maximum precision")
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
    print(f"\nFine-tuned settings:")
    print(f"  🎯 6 resolution levels for maximum detail")
    print(f"  🎯 400+ iterations per level")
    print(f"  🎯 15,000 features for better matching")
    print(f"  🎯 Enhanced multi-scale preprocessing")
    print(f"  🎯 Bilateral filtering for noise reduction")
    
    # Check GPU
    if tc.cuda.is_available():
        print(f"\n✅ GPU available: {tc.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU detected - registration will be slower")
    
    # Run fine-tuned registration
    results = register_qptiff_finetuned(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" FINE-TUNED REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Fine-tuned registration completed successfully!")
        print(f"   Time: {results.get('elapsed_time', 0):.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        print(f"   Dimensions: {results.get('output_shape', 'unknown')}")
        print(f"   Expected improvement: Finer detail alignment")
        print(f"   Check visualizations_finetuned/ for quality assessment")
    else:
        print(f"❌ Fine-tuned registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
