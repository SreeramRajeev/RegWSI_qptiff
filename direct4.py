#!/usr/bin/env python3
"""
Robust QPTIFF Registration Pipeline
==================================
Enhanced error handling and fallback mechanisms
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
# ROBUST PREPROCESSING FUNCTIONS
#############################################################################

def preprocess_for_registration_simple_robust(source, target):
    """
    Simple but robust preprocessing that avoids complex operations
    """
    print("  Applying simple robust preprocessing...")
    
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
    
    # Light enhancement - avoid complex LAB operations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    for channel in range(3):
        source_enhanced[:, :, channel] = clahe.apply(source[:, :, channel])
        target_enhanced[:, :, channel] = clahe.apply(target[:, :, channel])
    
    # Very light smoothing
    source_final = cv2.GaussianBlur(source_enhanced, (3, 3), 0.5)
    target_final = cv2.GaussianBlur(target_enhanced, (3, 3), 0.5)
    
    return source_final, target_final

def create_robust_registration_params():
    """
    More conservative registration parameters that are less likely to fail
    """
    params = default_initial_nonrigid()
    
    # Simpler initial alignment
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 10.0,
        'max_features': 5000,      # Reduced
        'match_ratio': 0.8,
        'use_mutual_best': False,  # More conservative
        'nms_radius': 4,
        'keypoint_threshold': 0.01,  # Less sensitive
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    # Conservative nonrigid parameters
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [100, 75, 50, 25],  # Reduced iterations
        'smoothing_sigma': 3.0,           # More smoothing
        'update_field_sigma': 2.0,        # More conservative updates
        'max_step_length': 2.0,           # Smaller steps
        'use_histogram_matching': True,
        'use_symmetric_forces': False,    # Simpler
        'use_gradient_type': 'fixed',     # More stable
    }
    
    # Fewer resolution levels
    params['multiresolution_params'] = {
        'levels': 3,  # Fewer levels
        'shrink_factors': [4, 2, 1],
        'smoothing_sigmas': [2.0, 1.0, 0.5],
    }
    
    # Conservative optimization
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 32,              # Fewer bins
        'optimizer': 'gradient_descent',
        'learning_rate': 1.0,              # Slower
        'min_step': 0.001,                 # Larger minimum step
        'iterations': 200,                 # Fewer iterations
        'relaxation_factor': 0.8,          # More relaxation
        'gradient_magnitude_tolerance': 1e-5,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.1,  # Less sampling
    }
    
    # Simple loading parameters
    params['loading_params'] = {
        'loader': 'tiff',
        'downsample_factor': 1,
    }
    
    # Output parameters
    params['save_displacement_field'] = True
    
    return params

def run_deephistreg_robust(he_path, if_path, output_dir, case_name='robust_reg'):
    """
    Run DeepHistReg with better error handling and debugging
    """
    print("  Running DeepHistReg with robust configuration...")
    
    # Clean output directory structure
    reg_dir = output_dir / "registration"
    temp_dir = reg_dir / "TEMP"
    
    # Ensure clean state
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        params = create_robust_registration_params()
        
        # Simple configuration
        config = {
            'source_path': str(he_path),
            'target_path': str(if_path),
            'output_path': str(reg_dir),
            'registration_parameters': params,
            'case_name': case_name,
            'save_displacement_field': True,
            'copy_target': True,
            'delete_temporary_results': False,  # Keep for debugging
            'temporary_path': str(temp_dir)
        }
        
        print(f"  Config: {config}")
        
        start_time = time.time()
        
        # Try the registration
        deeperhistreg.run_registration(**config)
        
        elapsed = time.time() - start_time
        print(f"  ✅ DeepHistReg completed in {elapsed:.1f} seconds")
        
        return True, elapsed, None
        
    except Exception as e:
        print(f"  ❌ DeepHistReg failed: {e}")
        
        # Debug: Show what was created
        print("  Debug - Files in registration directory:")
        if reg_dir.exists():
            for item in reg_dir.rglob("*"):
                if item.is_file():
                    print(f"    {item.relative_to(reg_dir)} ({item.stat().st_size} bytes)")
        
        print("  Debug - Files in temp directory:")
        if temp_dir.exists():
            for item in temp_dir.rglob("*"):
                if item.is_file():
                    print(f"    {item.relative_to(temp_dir)} ({item.stat().st_size} bytes)")
        
        return False, 0, str(e)

def find_any_output_files(reg_dir, temp_dir):
    """
    Find any usable output files from the registration attempt
    """
    print("  Searching for any usable output files...")
    
    # Look for warped files
    search_dirs = [reg_dir, temp_dir]
    warped_files = []
    displacement_files = []
    
    for search_dir in search_dirs:
        if search_dir.exists():
            # Find warped files
            warped_files.extend(list(search_dir.rglob("*warped*")))
            warped_files.extend(list(search_dir.rglob("*registered*")))
            
            # Find displacement files
            displacement_files.extend(list(search_dir.rglob("*.npy")))
            displacement_files.extend(list(search_dir.rglob("*displacement*")))
            displacement_files.extend(list(search_dir.rglob("*deformation*")))
    
    print(f"  Found {len(warped_files)} potential warped files")
    print(f"  Found {len(displacement_files)} potential displacement files")
    
    for wf in warped_files:
        print(f"    Warped: {wf}")
    for df in displacement_files:
        print(f"    Displacement: {df}")
    
    return warped_files, displacement_files

def load_image_robust(image_path):
    """
    Load image with multiple fallback methods
    """
    print(f"  Loading image: {image_path}")
    
    methods = [
        ("tifffile", lambda p: __import__('tifffile').imread(str(p))),
        ("cv2", lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)),
        ("PIL", lambda p: np.array(__import__('PIL.Image', fromlist=['Image']).open(str(p)))),
        ("skimage", lambda p: __import__('skimage.io', fromlist=['io']).imread(str(p))),
    ]
    
    for method_name, loader in methods:
        try:
            img = loader(image_path)
            if img is not None:
                print(f"  ✅ Loaded with {method_name}, shape: {img.shape}")
                return img
        except Exception as e:
            print(f"  ❌ {method_name} failed: {e}")
    
    return None

#############################################################################
# MAIN ROBUST REGISTRATION FUNCTION
#############################################################################

def register_qptiff_robust(he_qptiff_path: Path, if_qptiff_path: Path, 
                          output_dir: Path, if_channels: list = None) -> Dict:
    """
    Robust QPTIFF registration with extensive error handling
    """
    print("\n" + "="*70)
    print(" ROBUST QPTIFF REGISTRATION")
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
        print("\n1. Loading IF image...")
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
        
        # Step 3: Simple preprocessing
        print("\n3. Preprocessing with simple robust method...")
        he_prep, if_prep = preprocess_for_registration_simple_robust(he_data, if_rgb)
        
        # Save preprocessed versions
        he_prep_path = reg_dir / "he_preprocessed_robust.tiff"
        if_prep_path = reg_dir / "if_preprocessed_robust.tiff"
        tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw')
        tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw')
        
        print(f"   Preprocessed images saved")
        
        # Step 4: Run robust registration
        print("\n4. Running robust registration...")
        
        success, elapsed, error = run_deephistreg_robust(
            he_prep_path, if_prep_path, output_dir, 'robust_reg'
        )
        
        if success:
            print(f"   ✅ Registration completed successfully")
            results['registration_success'] = True
            results['elapsed_time'] = elapsed
        else:
            print(f"   ❌ Registration failed: {error}")
            results['registration_success'] = False
            results['registration_error'] = error
        
        # Step 5: Try to find and use any output
        print("\n5. Looking for any usable output...")
        
        warped_files, displacement_files = find_any_output_files(reg_dir, temp_dir)
        
        final_result = None
        
        # Try displacement field method first
        for disp_file in displacement_files:
            try:
                print(f"   Trying displacement file: {disp_file}")
                disp_field = np.load(str(disp_file))
                print(f"   Displacement shape: {disp_field.shape}")
                
                # Apply to original color image
                h, w = target_shape[:2]
                if disp_field.shape[0] == 2:
                    flow = disp_field.transpose(1, 2, 0)
                else:
                    flow = disp_field
                
                if flow.shape[:2] != (h, w):
                    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + flow[:, :, 0]).astype(np.float32)
                map_y = (y + flow[:, :, 1]).astype(np.float32)
                
                warped = cv2.remap(he_original_color, map_x, map_y, cv2.INTER_LINEAR)
                
                final_output_path = output_dir / "registered_HE_robust.tiff"
                tifffile.imwrite(
                    final_output_path,
                    warped,
                    photometric='rgb',
                    compression='lzw',
                    bigtiff=True
                )
                
                print(f"   ✅ Applied displacement field successfully")
                final_result = final_output_path
                results['method'] = 'displacement_field'
                break
                
            except Exception as e:
                print(f"   ❌ Displacement method failed: {e}")
        
        # Try warped files method
        if final_result is None:
            for warped_file in warped_files:
                warped = load_image_robust(warped_file)
                if warped is not None:
                    final_output_path = output_dir / f"registered_HE_from_{warped_file.stem}.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Used warped file: {warped_file.name}")
                    final_result = final_output_path
                    results['method'] = f'warped_file_{warped_file.stem}'
                    break
        
        if final_result:
            results['success'] = True
            results['registered_path'] = final_result
            print(f"   ✅ Final result saved: {final_result}")
        else:
            results['success'] = False
            results['error'] = "No usable output could be generated"
            print("   ❌ No usable output could be generated")
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution with robust error handling"""
    import argparse
    
    print("\n" + "="*70)
    print(" ROBUST QPTIFF REGISTRATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Robust QPTIFF registration")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default="./output_robust", help="Output directory")
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
    
    # Run robust registration
    results = register_qptiff_robust(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        if 'method' in results:
            print(f"   Method: {results['method']}")
        if 'elapsed_time' in results:
            print(f"   Time: {results['elapsed_time']:.1f} seconds")
        print(f"   Output: {results['registered_path']}")
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        if 'registration_error' in results:
            print(f"   DeepHistReg error: {results['registration_error']}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
