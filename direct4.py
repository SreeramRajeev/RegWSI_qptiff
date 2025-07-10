#!/usr/bin/env python3
"""
Direct QPTIFF Registration Pipeline - COLOR INPUT VERSION
=========================================================
Fixed to work with color images throughout the pipeline
Preserves original color information for better results

Key fixes:
- Modified preprocessing to work with color images
- Better color preservation strategy
- Improved displacement field handling
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

### Image Processing ###
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### DeepHistReg ###
import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

#############################################################################
# CONFIGURATION
#############################################################################

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Default IF channels to extract
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163

#############################################################################
# IMPROVED PREPROCESSING - WORKS WITH COLOR IMAGES
#############################################################################

def preprocess_for_registration_color(source, target):
    """
    Advanced preprocessing that preserves color information
    This version works with color images throughout the pipeline
    """
    print("  Applying color-preserving preprocessing...")
    
    # Ensure same dimensions
    if source.shape[:2] != target.shape[:2]:
        print(f"  Resizing source from {source.shape} to match target {target.shape}")
        source = cv2.resize(source, (target.shape[1], target.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    # Ensure both are 3-channel RGB
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    
    # Convert to uint8 if needed
    source = source.astype(np.uint8)
    target = target.astype(np.uint8)
    
    # METHOD 1: Enhanced color processing per channel
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    for channel in range(3):
        # Apply Gaussian blur per channel
        src_ch = cv2.GaussianBlur(source[:, :, channel], (5, 5), 1.0)
        tgt_ch = cv2.GaussianBlur(target[:, :, channel], (5, 5), 1.0)
        
        # Apply CLAHE per channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        src_ch_enhanced = clahe.apply(src_ch)
        tgt_ch_enhanced = clahe.apply(tgt_ch)
        
        source_enhanced[:, :, channel] = src_ch_enhanced
        target_enhanced[:, :, channel] = tgt_ch_enhanced
    
    # METHOD 2: Add structural enhancement using luminance
    # Convert to LAB color space for better luminance processing
    source_lab = cv2.cvtColor(source_enhanced, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_enhanced, cv2.COLOR_RGB2LAB)
    
    # Apply edge enhancement to L channel only
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]
    
    # Edge detection on luminance
    source_edges = cv2.Canny(source_l, 30, 100)
    target_edges = cv2.Canny(target_l, 30, 100)
    
    # Combine enhanced luminance with edge information
    source_l_final = cv2.addWeighted(source_l, 0.8, source_edges, 0.2, 0)
    target_l_final = cv2.addWeighted(target_l, 0.8, target_edges, 0.2, 0)
    
    # Put back the enhanced L channel
    source_lab[:, :, 0] = source_l_final
    target_lab[:, :, 0] = target_l_final
    
    # Convert back to RGB
    source_final = cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)
    target_final = cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)
    
    return source_final, target_final

def preprocess_for_registration_simple_color(source, target):
    """
    Simpler color preprocessing that maintains color structure
    Good fallback if LAB conversion causes issues
    """
    print("  Applying simple color preprocessing...")
    
    # Ensure same dimensions
    if source.shape[:2] != target.shape[:2]:
        print(f"  Resizing source from {source.shape} to match target {target.shape}")
        source = cv2.resize(source, (target.shape[1], target.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    # Ensure both are 3-channel RGB
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
    
    # Convert to uint8 if needed
    source = source.astype(np.uint8)
    target = target.astype(np.uint8)
    
    # Apply light enhancement while preserving color relationships
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    # Apply CLAHE per channel with gentler settings
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for channel in range(3):
        source_enhanced[:, :, channel] = clahe.apply(source[:, :, channel])
        target_enhanced[:, :, channel] = clahe.apply(target[:, :, channel])
    
    # Light Gaussian blur to reduce noise
    source_final = cv2.GaussianBlur(source_enhanced, (3, 3), 0.5)
    target_final = cv2.GaussianBlur(target_enhanced, (3, 3), 0.5)
    
    return source_final, target_final

def create_registration_params():
    """Registration parameters optimized for color images"""
    params = default_initial_nonrigid()
    
    # Feature-based initial alignment
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 8.0,  # Slightly more tolerant
        'max_features': 8000,     # Reduced for color images
        'match_ratio': 0.85,      # More selective matching
        'use_mutual_best': True,  # Better for color images
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    # Nonrigid parameters - optimized for color
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [150, 100, 75, 50],  # Reduced iterations
        'smoothing_sigma': 2.5,            # Slightly less smoothing
        'update_field_sigma': 1.5,         # Smaller updates
        'max_step_length': 3.0,            # Smaller steps
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric',
    }
    
    # Multi-resolution
    params['multiresolution_params'] = {
        'levels': 4,  # Fewer levels for color
        'shrink_factors': [8, 4, 2, 1],
        'smoothing_sigmas': [4.0, 2.0, 1.0, 0.5],
    }
    
    # Optimization - better for color images
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 64,              # More bins for color
        'optimizer': 'gradient_descent',
        'learning_rate': 1.5,              # Slower learning
        'min_step': 0.0001,                # Smaller minimum step
        'iterations': 300,                 # Fewer iterations
        'relaxation_factor': 0.7,          # More relaxation
        'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.15,  # More sampling
    }
    
    # TIFF support
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    
    # Ensure displacement field is saved
    params['save_displacement_field'] = True
    
    return params

#############################################################################
# IMPROVED DISPLACEMENT FIELD HANDLING
#############################################################################

def find_displacement_field(reg_dir, temp_dir, case_name='qptiff_reg'):
    """
    Find displacement field in various possible locations
    """
    possible_paths = [
        temp_dir / 'displacement_field.npy',
        temp_dir / f'{case_name}_displacement_field.npy',
        reg_dir / 'displacement_field.npy',
        reg_dir / f'{case_name}_displacement_field.npy',
        reg_dir / 'TEMP' / 'displacement_field.npy',
        reg_dir / 'TEMP' / f'{case_name}_displacement_field.npy',
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"   Found displacement field: {path}")
            return path
    
    print("   Displacement field not found in any expected location")
    return None

def apply_displacement_field_to_color(color_image, displacement_field):
    """
    Apply displacement field to color image with proper handling
    """
    print("   Applying displacement field to color image...")
    
    h, w = color_image.shape[:2]
    
    # Handle different displacement field formats
    if displacement_field.shape[0] == 2:  # Format: (2, H, W)
        flow = displacement_field.transpose(1, 2, 0)
    elif displacement_field.shape[-1] == 2:  # Format: (H, W, 2)
        flow = displacement_field
    else:
        print(f"   Unexpected displacement field shape: {displacement_field.shape}")
        return None
    
    # Ensure flow matches image dimensions
    if flow.shape[:2] != (h, w):
        print(f"   Resizing flow from {flow.shape[:2]} to {(h, w)}")
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate maps
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[:, :, 0]).astype(np.float32)
    map_y = (y + flow[:, :, 1]).astype(np.float32)
    
    # Apply transformation
    warped = cv2.remap(color_image, map_x, map_y, 
                       interpolation=cv2.INTER_LINEAR, 
                       borderMode=cv2.BORDER_CONSTANT, 
                       borderValue=0)
    
    return warped

def find_warped_result(reg_dir, case_name='qptiff_reg'):
    """
    Find the warped result file
    """
    possible_names = [
        'warped_source.tiff',
        f'{case_name}_warped_source.tiff',
        'warped_source.tif',
        f'{case_name}_warped_source.tif',
    ]
    
    for name in possible_names:
        path = reg_dir / name
        if path.exists():
            print(f"   Found warped result: {path}")
            return path
    
    # Check for any warped files
    warped_files = list(reg_dir.glob("*warped*"))
    if warped_files:
        print(f"   Found warped file: {warped_files[0]}")
        return warped_files[0]
    
    print("   No warped result found")
    return None

#############################################################################
# MAIN REGISTRATION FUNCTION - IMPROVED
#############################################################################

def register_qptiff_direct_color(he_qptiff_path: Path, if_qptiff_path: Path, 
                                output_dir: Path, if_channels: list = None) -> Dict:
    """
    Register H&E to IF QPTIFF directly at full resolution with color preservation
    """
    print("\n" + "="*70)
    print(" DIRECT QPTIFF REGISTRATION - COLOR VERSION")
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
        # Step 1: Load and prepare IF image
        print("\n1. Preparing IF image...")
        import tifffile
        
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            # Handle 4D data
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            # Extract and create RGB
            if if_data.ndim == 3 and if_data.shape[0] <= 16:
                selected = []
                for ch_idx in if_channels[:3]:
                    if ch_idx < if_data.shape[0]:
                        ch = if_data[ch_idx]
                        p1, p99 = np.percentile(ch, [1, 99])
                        ch_norm = np.clip((ch - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                        selected.append(ch_norm)
                
                if_rgb = np.stack(selected[:3], axis=-1)
                if if_rgb.shape[-1] < 3:
                    padding = np.zeros((*if_rgb.shape[:2], 3 - if_rgb.shape[-1]), dtype=np.uint8)
                    if_rgb = np.concatenate([if_rgb, padding], axis=-1)
                
                print(f"   IF RGB shape: {if_rgb.shape}")
                target_shape = if_rgb.shape
        
        # Step 2: Load H&E image
        print("\n2. Loading H&E image...")
        
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_data = tif.pages[0].asarray()
            print(f"   H&E shape: {he_data.shape}")
            
            # Handle planar configuration
            if he_data.ndim == 3 and he_data.shape[0] == 3:
                he_data = np.transpose(he_data, (1, 2, 0))
            
            # Ensure RGB
            if he_data.ndim == 2:
                he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
            
            # Convert to uint8
            if he_data.dtype != np.uint8:
                if he_data.dtype == np.uint16:
                    he_data = (he_data / 256).astype(np.uint8)
                else:
                    he_data = he_data.astype(np.uint8)
            
            # Store original color
            he_original_color = he_data.copy()
            
            # Resize to match IF
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E from {he_data.shape} to match IF {target_shape}")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original_color = cv2.resize(he_original_color, (target_shape[1], target_shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Preprocess with color preservation
        print("\n3. Preprocessing images with color preservation...")
        
        try:
            # Try advanced color preprocessing first
            he_prep, if_prep = preprocess_for_registration_color(he_data, if_rgb)
            preprocessing_method = "advanced_color"
        except Exception as e:
            print(f"   Advanced preprocessing failed: {e}")
            print("   Falling back to simple color preprocessing...")
            he_prep, if_prep = preprocess_for_registration_simple_color(he_data, if_rgb)
            preprocessing_method = "simple_color"
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed_color.tiff"
        if_prep_path = reg_dir / "if_preprocessed_color.tiff"
        tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        # Step 4: Run DeepHistReg with color images
        print("\n4. Running DeepHistReg with color images...")
        
        params = create_registration_params()
        
        config = {
            'source_path': str(he_prep_path),
            'target_path': str(if_prep_path),
            'output_path': str(reg_dir),
            'registration_parameters': params,
            'case_name': 'qptiff_color_reg',
            'save_displacement_field': True,
            'copy_target': True,
            'delete_temporary_results': False,
            'temporary_path': str(temp_dir)
        }
        
        start_time = time.time()
        deeperhistreg.run_registration(**config)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Registration completed in {elapsed:.1f} seconds")
        
        # Step 5: Apply transformation with improved handling
        print("\n5. Applying transformation to original color H&E...")
        
        # Try to find displacement field
        disp_field_path = find_displacement_field(reg_dir, temp_dir, 'qptiff_color_reg')
        warped_color = None
        
        if disp_field_path:
            try:
                displacement_field = np.load(str(disp_field_path))
                print(f"   Displacement field shape: {displacement_field.shape}")
                
                # Apply to original color image
                warped_color = apply_displacement_field_to_color(he_original_color, displacement_field)
                
                if warped_color is not None:
                    # Save final result
                    final_output_path = output_dir / "registered_HE_color.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Color registration saved: {final_output_path.name}")
                    print(f"   Output dimensions: {warped_color.shape}")
                    
                    results['success'] = True
                    results['registered_path'] = final_output_path
                    results['displacement_field'] = disp_field_path
                    results['elapsed_time'] = elapsed
                    results['output_shape'] = warped_color.shape
                    results['preprocessing_method'] = preprocessing_method
                
            except Exception as e:
                print(f"   ❌ Error applying displacement field: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: Try to find and use warped result directly
        if warped_color is None:
            print("\n   Trying to find warped result directly...")
            warped_path = find_warped_result(reg_dir, 'qptiff_color_reg')
            
            if warped_path:
                try:
                    warped_color = tifffile.imread(str(warped_path))
                    
                    # Save as color result
                    final_output_path = output_dir / "registered_HE_from_warped.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Warped result saved: {final_output_path.name}")
                    results['success'] = True
                    results['registered_path'] = final_output_path
                    results['method'] = 'direct_warped'
                    results['preprocessing_method'] = preprocessing_method
                    
                except Exception as e:
                    print(f"   ❌ Error loading warped result: {e}")
        
        if warped_color is None:
            print("   ❌ No usable registration result found")
            results['success'] = False
            results['error'] = "No displacement field or warped result found"
        
        # Step 6: Create visualizations
        if warped_color is not None:
            print("\n6. Creating visualizations...")
            try:
                create_visualizations(he_original_color, if_rgb, warped_color, output_dir)
                print("   ✅ Visualizations created")
            except Exception as e:
                print(f"   Warning: Visualization failed: {e}")
        
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

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution with color support"""
    import argparse
    
    print("\n" + "="*70)
    print(" QPTIFF REGISTRATION PIPELINE - COLOR VERSION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Direct QPTIFF registration with color preservation")
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
    
    # Run registration with color support
    results = register_qptiff_direct_color(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        print(f"   Method: {results.get('method', 'displacement_field')}")
        print(f"   Preprocessing: {results.get('preprocessing_method', 'unknown')}")
        print(f"   Time: {results.get('elapsed_time', 'unknown'):.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        if 'output_shape' in results:
            print(f"   Dimensions: {results['output_shape']}")
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
