#!/usr/bin/env python3
"""
Fixed Direct QPTIFF Registration Pipeline
========================================
Now supports MHA displacement fields from DeepHistReg
Preserves H&E colors in output at full resolution
"""

### System Imports ###
import os
import sys
import time
import shutil
import glob
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

### For MHA file reading ###
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

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

def read_mha_displacement_field(mha_path):
    """
    Read MHA displacement field using SimpleITK
    Returns numpy array in the format expected by cv2.remap
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK not available. Install with: pip install SimpleITK")
    
    print(f"   Reading MHA displacement field: {mha_path}")
    
    # Read the MHA file
    sitk_image = sitk.ReadImage(str(mha_path))
    
    # Convert to numpy array
    displacement_array = sitk.GetArrayFromImage(sitk_image)
    
    print(f"   MHA displacement shape: {displacement_array.shape}")
    print(f"   MHA spacing: {sitk_image.GetSpacing()}")
    print(f"   MHA origin: {sitk_image.GetOrigin()}")
    
    # MHA format is typically (Z, Y, X, Components) or (Y, X, Components)
    # We need to transpose to (Y, X, Components) for cv2.remap
    if displacement_array.ndim == 4:
        # Remove Z dimension if present
        if displacement_array.shape[0] == 1:
            displacement_array = displacement_array[0]
        else:
            displacement_array = displacement_array[0]  # Take first slice
    
    # Now should be (Y, X, Components)
    if displacement_array.ndim == 3 and displacement_array.shape[2] >= 2:
        print(f"   Final displacement shape: {displacement_array.shape}")
        return displacement_array
    else:
        raise ValueError(f"Unexpected displacement field shape: {displacement_array.shape}")

def find_displacement_field(reg_dir, temp_dir):
    """
    Find displacement field in various possible locations and formats
    """
    possible_paths = [
        # MHA files (most likely)
        reg_dir / "displacement_field.mha",
        temp_dir / "displacement_field.mha",
        
        # NPY files (fallback)
        temp_dir / "displacement_field.npy",
        reg_dir / "displacement_field.npy",
        
        # Check for case-specific names
        reg_dir / "qptiff_reg_displacement_field.mha",
        temp_dir / "qptiff_reg_displacement_field.mha",
        reg_dir / "qptiff_reg_displacement_field.npy",
        temp_dir / "qptiff_reg_displacement_field.npy",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"   Found displacement field: {path}")
            return path
    
    # If not found, search recursively
    print("   Searching for displacement field recursively...")
    for pattern in ["*displacement*.mha", "*displacement*.npy"]:
        matches = list(reg_dir.glob(f"**/{pattern}"))
        if matches:
            print(f"   Found displacement field: {matches[0]}")
            return matches[0]
    
    return None

def apply_displacement_field(image, displacement_field_path):
    """
    Apply displacement field to image using cv2.remap
    """
    print(f"   Applying displacement field: {displacement_field_path.suffix}")
    
    if displacement_field_path.suffix == '.mha':
        # Read MHA file
        displacement_field = read_mha_displacement_field(displacement_field_path)
    elif displacement_field_path.suffix == '.npy':
        # Read NPY file
        displacement_field = np.load(str(displacement_field_path))
        # Transpose if needed (from C, H, W to H, W, C)
        if displacement_field.ndim == 3 and displacement_field.shape[0] == 2:
            displacement_field = displacement_field.transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported displacement field format: {displacement_field_path.suffix}")
    
    h, w = image.shape[:2]
    
    # Extract displacement components
    if displacement_field.shape[2] >= 2:
        disp_x = displacement_field[:, :, 0]
        disp_y = displacement_field[:, :, 1]
    else:
        raise ValueError("Displacement field must have at least 2 components")
    
    # Resize displacement field if needed
    if disp_x.shape != (h, w):
        print(f"   Resizing displacement field from {disp_x.shape} to {(h, w)}")
        disp_x = cv2.resize(disp_x, (w, h), interpolation=cv2.INTER_LINEAR)
        disp_y = cv2.resize(disp_y, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate maps
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + disp_x).astype(np.float32)
    map_y = (y + disp_y).astype(np.float32)
    
    # Apply transformation
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return warped

def find_warped_result(reg_dir):
    """
    Find the warped result image in various possible locations
    """
    possible_names = [
        "warped_source.tiff",
        "warped_source.tif",
        "qptiff_reg_warped_source.tiff",
        "qptiff_reg_warped_source.tif",
        "registered_source.tiff",
        "registered_source.tif",
    ]
    
    for name in possible_names:
        path = reg_dir / name
        if path.exists():
            print(f"   Found warped result: {path}")
            return path
    
    # Search recursively
    print("   Searching for warped result recursively...")
    for pattern in ["*warped*.tiff", "*warped*.tif", "*registered*.tiff", "*registered*.tif"]:
        matches = list(reg_dir.glob(f"**/{pattern}"))
        if matches:
            print(f"   Found warped result: {matches[0]}")
            return matches[0]
    
    return None

#############################################################################
# PREPROCESSING FUNCTIONS
#############################################################################

def preprocess_for_registration(source, target):
    """
    Advanced preprocessing for better alignment between H&E and IF
    """
    print("  Applying advanced preprocessing...")
    
    # Ensure same dimensions
    if source.shape[:2] != target.shape[:2]:
        print(f"  Resizing source from {source.shape} to match target {target.shape}")
        source = cv2.resize(source, (target.shape[1], target.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
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

def create_registration_params():
    """Registration parameters optimized for QPTIFF"""
    params = default_initial_nonrigid()
    
    # Feature-based initial alignment
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
    
    # Nonrigid parameters
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
    
    # Multi-resolution
    params['multiresolution_params'] = {
        'levels': 5,
        'shrink_factors': [16, 8, 4, 2, 1],
        'smoothing_sigmas': [8.0, 4.0, 2.0, 1.0, 0.5],
    }
    
    # Optimization
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
# MAIN REGISTRATION FUNCTION
#############################################################################

def register_qptiff_direct(he_qptiff_path: Path, if_qptiff_path: Path, 
                          output_dir: Path, if_channels: list = None) -> Dict:
    """
    Register H&E to IF QPTIFF directly at full resolution
    """
    print("\n" + "="*70)
    print(" DIRECT QPTIFF REGISTRATION")
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
        # Step 1: Prepare IF image
        print("\n1. Preparing IF image...")
        import tifffile
        
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            # Handle 4D data
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            # Extract channels
            if if_data.ndim == 3 and if_data.shape[0] <= 16:
                selected = []
                for ch_idx in if_channels[:3]:
                    if ch_idx < if_data.shape[0]:
                        ch = if_data[ch_idx]
                        p1, p99 = np.percentile(ch, [1, 99])
                        ch_norm = np.clip((ch - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                        selected.append(ch_norm)
                
                # Create RGB
                if_rgb = np.stack(selected[:3], axis=-1)
                if if_rgb.shape[-1] < 3:
                    padding = np.zeros((*if_rgb.shape[:2], 3 - if_rgb.shape[-1]), dtype=np.uint8)
                    if_rgb = np.concatenate([if_rgb, padding], axis=-1)
                    
                print(f"   IF RGB shape: {if_rgb.shape}")
                
                # Save temporary IF RGB
                if_rgb_path = temp_dir / "if_rgb_temp.tiff"
                tifffile.imwrite(if_rgb_path, if_rgb, photometric='rgb', compression='lzw')
                
                target_shape = if_rgb.shape
        
        # Step 2: Load H&E
        print("\n2. Loading H&E image...")
        
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_page = tif.pages[0]
            he_data = he_page.asarray()
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
            
            # Store original for color preservation
            he_original_color = he_data.copy()
            
            # Resize if needed
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E from {he_data.shape} to match IF {target_shape}")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original_color = cv2.resize(he_original_color, (target_shape[1], target_shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            
            # Save temporary H&E
            he_temp_path = temp_dir / "he_temp.tiff"
            tifffile.imwrite(he_temp_path, he_data, photometric='rgb', compression='lzw')
        
        # Step 3: Preprocess
        print("\n3. Preprocessing images...")
        he_prep, if_prep = preprocess_for_registration(he_data, if_rgb)
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed.tiff"
        if_prep_path = reg_dir / "if_preprocessed.tiff"
        tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        # Step 4: Run DeepHistReg
        print("\n4. Running DeepHistReg registration...")
        
        params = create_registration_params()
        
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
        
        # Step 5: Apply transformation to original color H&E
        print("\n5. Applying transformation to original color H&E...")
        
        # Find displacement field
        disp_field_path = find_displacement_field(reg_dir, temp_dir)
        warped_color = None
        
        if disp_field_path:
            try:
                # Apply displacement field to original color H&E
                warped_color = apply_displacement_field(he_original_color, disp_field_path)
                
                # Save result
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
                
            except Exception as e:
                print(f"   ❌ Error applying displacement field: {e}")
                import traceback
                traceback.print_exc()
                warped_color = None
        else:
            print("   ❌ No displacement field found!")
        
        # Fallback: try to find and use warped grayscale result
        if warped_color is None:
            print("\n   Trying to find warped grayscale result...")
            warped_gray_path = find_warped_result(reg_dir)
            
            if warped_gray_path:
                try:
                    warped_gray = tifffile.imread(str(warped_gray_path))
                    print(f"   Found warped result: {warped_gray.shape}")
                    
                    # If grayscale, convert to RGB
                    if warped_gray.ndim == 2:
                        warped_color = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2RGB)
                    else:
                        warped_color = warped_gray
                    
                    # Save as fallback result
                    fallback_path = output_dir / "registered_HE_grayscale.tiff"
                    tifffile.imwrite(
                        fallback_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Fallback result saved: {fallback_path.name}")
                    
                    results['success'] = True
                    results['registered_path'] = fallback_path
                    results['elapsed_time'] = elapsed
                    results['output_shape'] = warped_color.shape
                    results['method'] = 'grayscale_fallback'
                    
                except Exception as e:
                    print(f"   ❌ Error with grayscale fallback: {e}")
                    results['success'] = False
                    results['error'] = f"All methods failed: {e}"
            else:
                print("   ❌ No warped result found!")
                results['success'] = False
                results['error'] = "No displacement field or warped result found"
        
        # Step 6: Create visualizations
        if warped_color is not None:
            print("\n6. Creating visualizations...")
            try:
                create_visualizations(he_original_color, if_rgb, warped_color, output_dir)
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
    
    # Ensure same dimensions
    h, w = if_img.shape[:2]
    if he_original.shape[:2] != (h, w):
        he_original = cv2.resize(he_original, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # Create downsampled versions for visualization
    scale = min(1.0, 2000 / max(h, w))  # Limit to 2000px max dimension
    new_h, new_w = int(h * scale), int(w * scale)
    
    he_viz = cv2.resize(he_original, (new_w, new_h))
    if_viz = cv2.resize(if_img, (new_w, new_h))
    warped_viz = cv2.resize(warped, (new_w, new_h))
    
    # 1. Side-by-side comparison
    comparison = np.hstack([he_viz, if_viz, warped_viz])
    cv2.imwrite(str(viz_dir / "side_by_side.jpg"), 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 2. Checkerboard overlay
    checker_size = max(50, min(new_h, new_w) // 20)
    checkerboard = np.zeros_like(if_viz)
    
    for i in range(0, new_h, checker_size):
        for j in range(0, new_w, checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = warped_viz[i:i+checker_size, j:j+checker_size]
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = if_viz[i:i+checker_size, j:j+checker_size]
    
    cv2.imwrite(str(viz_dir / "checkerboard.jpg"), 
                cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # 3. Overlay blend
    overlay = cv2.addWeighted(if_viz, 0.5, warped_viz, 0.5, 0)
    cv2.imwrite(str(viz_dir / "overlay.jpg"), 
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"   ✅ Visualizations saved to: {viz_dir}")

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution with improved error handling"""
    import argparse
    
    print("\n" + "="*70)
    print(" QPTIFF REGISTRATION PIPELINE (FULL RESOLUTION)")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Direct QPTIFF registration at full resolution")
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
    
    # Check SimpleITK availability
    if not HAS_SITK:
        print("\n⚠️  SimpleITK not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "SimpleITK"])
            import SimpleITK as sitk
            global HAS_SITK
            HAS_SITK = True
            print("✅ SimpleITK installed successfully")
        except Exception as e:
            print(f"❌ Failed to install SimpleITK: {e}")
            return 1
    
    # Check GPU
    if tc.cuda.is_available():
        print(f"\n✅ GPU available: {tc.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  No GPU detected - registration will be slower")
    
    # Run registration
    results = register_qptiff_direct(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        print(f"   Time: {results['elapsed_time']:.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        print(f"   Dimensions: {results['output_shape']}")
        if 'method' in results:
            print(f"   Method: {results['method']}")
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
