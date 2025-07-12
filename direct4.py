#!/usr/bin/env python3
"""
Color-Direct QPTIFF Registration Pipeline
==========================================
Registers color images directly without grayscale conversion
Properly handles .mha displacement fields
Preserves colors naturally by working with color throughout
"""

### System Imports ###
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, Optional, Dict
import glob

### Scientific Computing ###
import numpy as np
import torch as tc
from scipy.ndimage import map_coordinates

### Image Processing ###
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### DeepHistReg ###
import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

# For handling .mha files
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("Warning: SimpleITK not available. .mha displacement fields may not be handled.")

#############################################################################
# CONFIGURATION
#############################################################################

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Default IF channels to extract
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163

#############################################################################
# PREPROCESSING FUNCTIONS (MODIFIED FOR COLOR)
#############################################################################

def preprocess_for_registration_color(source, target, use_color=True):
    """
    Color-preserving preprocessing for better alignment between H&E and IF
    Option to keep color throughout the pipeline
    """
    print(f"  Applying preprocessing (color={use_color})...")
    
    # Ensure same dimensions
    if source.shape[:2] != target.shape[:2]:
        print(f"  Resizing source from {source.shape} to match target {target.shape}")
        source = cv2.resize(source, (target.shape[1], target.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    if use_color:
        # Keep color but enhance contrast and edges
        source_enhanced = source.copy()
        target_enhanced = target.copy()
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):
            source_enhanced[:, :, i] = clahe.apply(source_enhanced[:, :, i])
            target_enhanced[:, :, i] = clahe.apply(target_enhanced[:, :, i])
        
        # Slight Gaussian blur for noise reduction
        source_final = cv2.GaussianBlur(source_enhanced, (3, 3), 0.5)
        target_final = cv2.GaussianBlur(target_enhanced, (3, 3), 0.5)
        
        return source_final, target_final
    
    else:
        # Original grayscale method
        # Convert to grayscale for processing
        if source.ndim == 3:
            source_gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            source_gray = source.astype(np.uint8)
        
        if target.ndim == 3:
            target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target.astype(np.uint8)
        
        # Apply Gaussian blur
        source_blur = cv2.GaussianBlur(source_gray, (5, 5), 1.0)
        target_blur = cv2.GaussianBlur(target_gray, (5, 5), 1.0)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        source_enhanced = clahe.apply(source_blur)
        target_enhanced = clahe.apply(target_blur)
        
        # Edge enhancement
        source_edges = cv2.Canny(source_enhanced, 30, 100)
        target_edges = cv2.Canny(target_enhanced, 30, 100)
        
        # Combine enhanced and edge information
        source_combined = cv2.addWeighted(source_enhanced, 0.7, source_edges, 0.3, 0)
        target_combined = cv2.addWeighted(target_enhanced, 0.7, target_edges, 0.3, 0)
        
        # Convert back to RGB
        source_final = cv2.cvtColor(source_combined, cv2.COLOR_GRAY2RGB)
        target_final = cv2.cvtColor(target_combined, cv2.COLOR_GRAY2RGB)
        
        return source_final, target_final

def create_registration_params(use_color=True):
    """Registration parameters optimized for color or grayscale"""
    params = default_initial_nonrigid()
    
    # Feature-based initial alignment
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 10.0,
        'max_features': 8000 if use_color else 10000,
        'match_ratio': 0.85 if use_color else 0.9,
        'use_mutual_best': False,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    # Nonrigid parameters (adjusted for color)
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [150, 100, 75, 50] if use_color else [200, 150, 100, 50],
        'smoothing_sigma': 2.5 if use_color else 3.0,
        'update_field_sigma': 1.5 if use_color else 2.0,
        'max_step_length': 4.0 if use_color else 5.0,
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric',
    }
    
    # Multi-resolution
    params['multiresolution_params'] = {
        'levels': 4 if use_color else 5,
        'shrink_factors': [8, 4, 2, 1] if use_color else [16, 8, 4, 2, 1],
        'smoothing_sigmas': [4.0, 2.0, 1.0, 0.5] if use_color else [8.0, 4.0, 2.0, 1.0, 0.5],
    }
    
    # Optimization (adjusted for color)
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 64 if use_color else 32,
        'optimizer': 'gradient_descent',
        'learning_rate': 1.5 if use_color else 2.0,
        'min_step': 0.001,
        'iterations': 400 if use_color else 500,
        'relaxation_factor': 0.85 if use_color else 0.8,
        'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.15 if use_color else 0.1,
    }
    
    # IMPORTANT: Tell DeepHistReg to work with QPTIFF directly
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    
    # Ensure displacement field is saved
    params['save_displacement_field'] = True
    
    return params

def find_displacement_field(temp_dir, reg_dir, case_name='qptiff_reg'):
    """
    Find displacement field file in various possible locations and formats
    """
    possible_names = [
        'displacement_field.npy',
        'displacement_field.mha',
        f'{case_name}_displacement_field.npy',
        f'{case_name}_displacement_field.mha',
        'deformation_field.npy',
        'deformation_field.mha',
    ]
    
    possible_dirs = [temp_dir, reg_dir, reg_dir / 'TEMP']
    
    for directory in possible_dirs:
        if not directory.exists():
            continue
            
        for name in possible_names:
            filepath = directory / name
            if filepath.exists():
                print(f"   Found displacement field: {filepath}")
                return filepath
    
    # Last resort: search for any .npy or .mha files containing "displacement" or "deformation"
    for directory in possible_dirs:
        if not directory.exists():
            continue
            
        for pattern in ['*displacement*.npy', '*displacement*.mha', '*deformation*.npy', '*deformation*.mha']:
            matches = list(directory.glob(pattern))
            if matches:
                print(f"   Found displacement field (pattern match): {matches[0]}")
                return matches[0]
    
    return None

def load_displacement_field(filepath):
    """
    Load displacement field from .npy or .mha file
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        return np.load(str(filepath))
    
    elif filepath.suffix == '.mha':
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK required to read .mha files. Install with: pip install SimpleITK")
        
        # Load using SimpleITK
        displacement_image = sitk.ReadImage(str(filepath))
        displacement_array = sitk.GetArrayFromImage(displacement_image)
        
        # SimpleITK arrays are typically (Z, Y, X, Components)
        # We need (Y, X, Components) for 2D displacement
        if displacement_array.ndim == 4:
            displacement_array = displacement_array[0]  # Take first slice if 3D
        
        # Ensure correct format: (Y, X, 2) for 2D displacement
        if displacement_array.shape[-1] == 2:
            return displacement_array
        elif displacement_array.ndim == 3 and displacement_array.shape[0] == 2:
            # If it's (2, Y, X), transpose to (Y, X, 2)
            return displacement_array.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected displacement field shape: {displacement_array.shape}")
    
    else:
        raise ValueError(f"Unsupported displacement field format: {filepath.suffix}")

def apply_displacement_field(image, displacement_field, chunk_size=16384):
    """
    Apply displacement field to warp an image
    Handles large images by processing in chunks to avoid OpenCV size limitations
    """
    h, w = image.shape[:2]
    
    # Ensure displacement field has correct shape
    if displacement_field.shape[:2] != (h, w):
        print(f"   Resizing displacement field from {displacement_field.shape} to {(h, w, 2)}")
        displacement_field = cv2.resize(displacement_field, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Check if image is too large for OpenCV (SHRT_MAX = 32767)
    max_opencv_size = 32767
    
    if h <= max_opencv_size and w <= max_opencv_size:
        # Small enough for direct processing
        print(f"   Applying displacement field directly (size: {h}x{w})")
        return _apply_displacement_direct(image, displacement_field)
    else:
        # Too large - use chunked processing
        print(f"   Image too large for OpenCV direct processing ({h}x{w})")
        print(f"   Using chunked processing with chunk size: {chunk_size}")
        return _apply_displacement_chunked(image, displacement_field, chunk_size)

def _apply_displacement_direct(image, displacement_field):
    """Direct application for smaller images"""
    h, w = image.shape[:2]
    
    # Create coordinate maps
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + displacement_field[:, :, 0]).astype(np.float32)
    map_y = (y + displacement_field[:, :, 1]).astype(np.float32)
    
    # Apply warping
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def _apply_displacement_chunked(image, displacement_field, chunk_size):
    """Chunked processing for large images"""
    h, w = image.shape[:2]
    
    # Create output array
    if image.ndim == 3:
        warped = np.zeros_like(image)
    else:
        warped = np.zeros_like(image)
    
    # Calculate number of chunks
    h_chunks = (h + chunk_size - 1) // chunk_size
    w_chunks = (w + chunk_size - 1) // chunk_size
    
    print(f"   Processing in {h_chunks}x{w_chunks} = {h_chunks * w_chunks} chunks")
    
    # Process each chunk
    for i in range(h_chunks):
        for j in range(w_chunks):
            # Calculate chunk boundaries
            h_start = i * chunk_size
            h_end = min((i + 1) * chunk_size, h)
            w_start = j * chunk_size
            w_end = min((j + 1) * chunk_size, w)
            
            if (i * w_chunks + j) % 10 == 0:  # Progress indicator
                print(f"   Processing chunk {i * w_chunks + j + 1}/{h_chunks * w_chunks}")
            
            # Extract displacement chunk
            disp_chunk = displacement_field[h_start:h_end, w_start:w_end]
            
            # Create coordinate grids for this chunk
            chunk_h, chunk_w = disp_chunk.shape[:2]
            y_coords, x_coords = np.mgrid[h_start:h_end, w_start:w_end]
            
            # Apply displacement to get sampling coordinates
            sample_x = x_coords + disp_chunk[:, :, 0]
            sample_y = y_coords + disp_chunk[:, :, 1]
            
            # Clamp coordinates to image boundaries
            sample_x = np.clip(sample_x, 0, w - 1)
            sample_y = np.clip(sample_y, 0, h - 1)
            
            # Sample from the full image at these coordinates
            try:
                warped_chunk = _sample_image_at_coordinates(image, sample_x, sample_y)
                warped[h_start:h_end, w_start:w_end] = warped_chunk
                
            except Exception as e:
                print(f"   Warning: Error processing chunk ({i},{j}): {e}")
                # Fall back to copying original chunk
                warped[h_start:h_end, w_start:w_end] = image[h_start:h_end, w_start:w_end]
    
    return warped

def _sample_image_at_coordinates(image, map_x, map_y):
    """
    Sample image at given coordinates using bilinear interpolation
    Alternative to cv2.remap for when coordinates are already computed
    """
    h, w = image.shape[:2]
    
    # Flatten coordinates for map_coordinates
    coords_y = map_y.flatten()
    coords_x = map_x.flatten()
    
    if image.ndim == 3:
        # Color image
        result = np.zeros((map_y.size, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            result[:, c] = map_coordinates(
                image[:, :, c], 
                [coords_y, coords_x], 
                order=1,  # Linear interpolation
                mode='reflect',
                prefilter=False
            )
        result = result.reshape(map_y.shape + (image.shape[2],))
    else:
        # Grayscale image
        result = map_coordinates(
            image, 
            [coords_y, coords_x], 
            order=1,  # Linear interpolation
            mode='reflect',
            prefilter=False
        )
        result = result.reshape(map_y.shape)
    
    return result.astype(image.dtype)

#############################################################################
# MAIN REGISTRATION FUNCTION
#############################################################################

def register_qptiff_color_direct(he_qptiff_path: Path, if_qptiff_path: Path, 
                                output_dir: Path, if_channels: list = None, 
                                use_color: bool = True, chunk_size: int = 16384) -> Dict:
    """
    Register H&E to IF QPTIFF directly at full resolution using color
    
    Args:
        use_color: If True, registers color images directly. If False, uses grayscale method.
    """
    print("\n" + "="*70)
    print(f" {'COLOR' if use_color else 'GRAYSCALE'} QPTIFF REGISTRATION")
    print("="*70)
    
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration"
    reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir / "TEMP"
    temp_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Prepare IF image (extract channels as RGB)
        print("\n1. Preparing IF image...")
        
        import tifffile
        
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            # Handle 4D data (Z, C, Y, X)
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            # Extract channels
            if if_data.ndim == 3 and if_data.shape[0] <= 16:  # C, Y, X format
                selected = []
                for ch_idx in if_channels[:3]:  # Take first 3 for RGB
                    if ch_idx < if_data.shape[0]:
                        ch = if_data[ch_idx]
                        # Normalize to 8-bit
                        p1, p99 = np.percentile(ch, [1, 99])
                        ch_norm = np.clip((ch - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                        selected.append(ch_norm)
                
                # Create RGB
                if_rgb = np.stack(selected[:3], axis=-1)
                if if_rgb.shape[-1] < 3:
                    # Pad with zeros if less than 3 channels
                    padding = np.zeros((*if_rgb.shape[:2], 3 - if_rgb.shape[-1]), dtype=np.uint8)
                    if_rgb = np.concatenate([if_rgb, padding], axis=-1)
                    
                print(f"   IF RGB shape: {if_rgb.shape}")
                
                # Save temporary IF RGB
                if_rgb_path = temp_dir / "if_rgb_temp.tiff"
                tifffile.imwrite(if_rgb_path, if_rgb, photometric='rgb', compression='lzw')
                
                # Store dimensions
                target_shape = if_rgb.shape
        
        # Step 2: Load H&E and prepare for registration
        print("\n2. Loading H&E image...")
        
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_page = tif.pages[0]
            he_data = he_page.asarray()
            print(f"   H&E shape: {he_data.shape}")
            
            # Handle planar configuration if needed
            if he_data.ndim == 3 and he_data.shape[0] == 3:
                he_data = np.transpose(he_data, (1, 2, 0))
            
            # Ensure RGB
            if he_data.ndim == 2:
                he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
            
            # Convert to uint8 if needed
            if he_data.dtype != np.uint8:
                if he_data.dtype == np.uint16:
                    he_data = (he_data / 256).astype(np.uint8)
                else:
                    he_data = he_data.astype(np.uint8)
            
            # Resize H&E to match IF if needed
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E from {he_data.shape} to match IF {target_shape}")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
            
            # Save original H&E (this will be warped)
            he_original_path = temp_dir / "he_original.tiff"
            tifffile.imwrite(he_original_path, he_data, photometric='rgb', compression='lzw')
        
        # Step 3: Preprocess both images
        print("\n3. Preprocessing images...")
        he_prep, if_prep = preprocess_for_registration_color(he_data, if_rgb, use_color=use_color)
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed.tiff"
        if_prep_path = reg_dir / "if_preprocessed.tiff"
        tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        # Step 4: Run DeepHistReg
        print("\n4. Running DeepHistReg registration...")
        
        params = create_registration_params(use_color=use_color)
        
        # Configure DeepHistReg
        config = {
            'source_path': str(he_prep_path),
            'target_path': str(if_prep_path),
            'output_path': str(reg_dir),
            'registration_parameters': params,
            'case_name': 'qptiff_reg',
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
        
        # Find displacement field
        disp_field_path = find_displacement_field(temp_dir, reg_dir, 'qptiff_reg')
        
        if disp_field_path:
            try:
                # Load displacement field
                displacement_field = load_displacement_field(disp_field_path)
                print(f"   Displacement field shape: {displacement_field.shape}")
                
                # Apply to original H&E image
                warped_he = apply_displacement_field(he_data, displacement_field, chunk_size)
                
                # Save final result
                final_output_path = output_dir / f"registered_HE_{'color' if use_color else 'enhanced'}.tiff"
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
                results['method'] = 'color_direct' if use_color else 'grayscale_enhanced'
                
                # Step 6: Create visualizations
                print("\n6. Creating visualizations...")
                try:
                    create_visualizations(he_data, if_rgb, warped_he, output_dir, use_color)
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
            
            # List all files in temp and reg directories for debugging
            print("\n   Debug: Files in temp directory:")
            if temp_dir.exists():
                for f in temp_dir.iterdir():
                    print(f"     {f.name}")
            
            print("\n   Debug: Files in registration directory:")
            if reg_dir.exists():
                for f in reg_dir.iterdir():
                    print(f"     {f.name}")
            
            results['success'] = False
            results['error'] = "Displacement field not found"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results

def create_visualizations(he_original, if_img, warped, output_dir, use_color=True):
    """Create quality check visualizations"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Ensure all images have same shape
    h, w = if_img.shape[:2]
    if he_original.shape[:2] != (h, w):
        he_original = cv2.resize(he_original, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # 1. Side-by-side comparison
    comparison = np.hstack([he_original, if_img, warped])
    cv2.imwrite(str(viz_dir / "side_by_side.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 2. Checkerboard overlay
    checker_size = max(100, min(h, w) // 20)
    checkerboard = np.zeros_like(if_img)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = warped[i:i+checker_size, j:j+checker_size]
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = if_img[i:i+checker_size, j:j+checker_size]
    
    cv2.imwrite(str(viz_dir / "checkerboard.jpg"), cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # 3. Overlay blend
    overlay = cv2.addWeighted(if_img, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(viz_dir / "overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 4. Difference map (for quality assessment)
    if_gray = cv2.cvtColor(if_img, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(if_gray, warped_gray)
    cv2.imwrite(str(viz_dir / "difference_map.jpg"), diff)
    
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
    print(" COLOR-DIRECT QPTIFF REGISTRATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Color-direct QPTIFF registration")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--if-channels", type=int, nargs='+', default=[0, 1, 5], 
                       help="IF channels to use (default: 0=DAPI, 1=CD8, 5=CD163)")
    parser.add_argument("--use-color", action='store_true', default=True,
                       help="Use color registration (default: True)")
    parser.add_argument("--use-grayscale", action='store_true', default=False,
                       help="Force grayscale registration")
    parser.add_argument("--chunk-size", type=int, default=16384,
                       help="Chunk size for large image processing (default: 16384)")
    
    args = parser.parse_args()
    
    # Handle color vs grayscale option
    use_color = args.use_color and not args.use_grayscale
    
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
    print(f"  Mode: {'Color' if use_color else 'Grayscale'} registration")
    
    # Check dependencies
    if tc.cuda.is_available():
        print(f"\n✅ GPU available: {tc.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU detected - registration will be slower")
    
    if not SITK_AVAILABLE:
        print("\n⚠️  SimpleITK not available - install with 'pip install SimpleITK' for .mha support")
    
    # Run registration
    results = register_qptiff_color_direct(he_path, if_path, output_dir, args.if_channels, use_color, args.chunk_size)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        print(f"   Method: {results.get('method', 'unknown')}")
        print(f"   Time: {results.get('elapsed_time', 0):.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        print(f"   Dimensions: {results.get('output_shape', 'unknown')}")
        
        # Evaluate quality
        if_temp = output_dir / "registration" / "TEMP" / "if_rgb_temp.tiff"
        if if_temp.exists() and results['registered_path'].exists():
            evaluate_registration_quality(if_temp, results['registered_path'])
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
