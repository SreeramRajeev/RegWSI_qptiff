#!/usr/bin/env python3
"""
Grayscale QPTIFF Registration Pipeline
=====================================
Works with DeepHistReg's grayscale expectations
Applies transformation to original color H&E for final output
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
# GRAYSCALE PREPROCESSING FUNCTIONS
#############################################################################

def create_optimal_grayscale_from_color(color_image, method='weighted'):
    """
    Create optimal grayscale from color image for registration
    """
    if color_image.ndim == 2:
        return color_image
    
    if method == 'weighted':
        # Optimized weights for H&E images
        # Higher weight on green (eosin) and blue (hematoxylin)
        weights = np.array([0.2, 0.5, 0.3])  # R, G, B
        gray = np.dot(color_image, weights)
    elif method == 'luminance':
        # Standard luminance conversion
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    elif method == 'max_contrast':
        # Use channel with maximum contrast
        contrasts = [np.std(color_image[:,:,i]) for i in range(3)]
        best_channel = np.argmax(contrasts)
        gray = color_image[:,:,best_channel]
    else:
        # Default to standard conversion
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    
    return gray.astype(np.uint8)

def create_optimal_if_grayscale(if_channels_data, channels=[0, 1, 5]):
    """
    Create optimal grayscale from IF channels for registration
    """
    # Extract the specified channels
    selected_channels = []
    for ch_idx in channels:
        if ch_idx < if_channels_data.shape[0]:
            ch = if_channels_data[ch_idx]
            # Normalize each channel
            p1, p99 = np.percentile(ch, [1, 99])
            ch_norm = np.clip((ch - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            selected_channels.append(ch_norm)
    
    if len(selected_channels) == 0:
        raise ValueError("No valid channels found")
    
    if len(selected_channels) == 1:
        return selected_channels[0]
    
    # Combine channels with weights optimized for registration
    # DAPI (nuclei) gets higher weight for structural information
    if len(channels) >= 3:
        weights = np.array([0.5, 0.3, 0.2])  # DAPI, CD8, CD163
    else:
        weights = np.ones(len(selected_channels)) / len(selected_channels)
    
    # Create weighted combination
    gray_if = np.zeros_like(selected_channels[0], dtype=np.float32)
    for i, channel in enumerate(selected_channels):
        gray_if += channel.astype(np.float32) * weights[i]
    
    return np.clip(gray_if, 0, 255).astype(np.uint8)

def enhance_grayscale_for_registration(gray_image):
    """
    Enhance grayscale image for better registration
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray_image)
    
    # Light denoising
    denoised = cv2.medianBlur(enhanced, 3)
    
    # Edge preservation with bilateral filter
    final = cv2.bilateralFilter(denoised, 5, 50, 50)
    
    return final

def create_simple_registration_params():
    """
    Simple registration parameters that work reliably with DeepHistReg
    """
    # Start with minimal default parameters
    params = {
        'device': 'cuda:0',
        'echo': False,
        
        # Simple loading - let DeepHistReg handle preprocessing
        'loading_params': {
            'loader': 'tiff',
            'downsample_factor': 1,
        },
        
        # Minimal saving parameters
        'saving_params': {
            'saver': 'tiff',
            'save_params': 'pil',
        },
        
        # Let DeepHistReg use its default preprocessing
        # This avoids the pad_value error
        
        # Simple initial alignment
        'run_initial_registration': True,
        'initial_registration_params': {
            'save_results': True,
            'run_superpoint_superglue': True,
            'run_sift_ransac': False,  # Disable SIFT to avoid conflicts
            'registration_size': 512,  # Smaller for stability
            'cuda': True,
            'device': 'cuda:0',
        },
        
        # Simple nonrigid registration
        'run_nonrigid_registration': True,
        'nonrigid_registration_params': {
            'save_results': True,
            'device': 'cuda:0',
            'registration_size': 2048,  # Moderate size
            'num_levels': 6,  # Fewer levels
            'used_levels': 6,
            'iterations': [50, 50, 50, 50, 50, 50],  # Fewer iterations
            'learning_rates': [0.01, 0.005, 0.005, 0.005, 0.005, 0.005],
        },
        
        # Ensure outputs are saved
        'save_displacement_field': True,
        'save_final_displacement_field': True,
        'save_final_images': True,
    }
    
    return params

def run_deephistreg_grayscale(he_gray_path, if_gray_path, output_dir, case_name='grayscale_reg'):
    """
    Run DeepHistReg with grayscale images and simple parameters
    """
    print("  Running DeepHistReg with grayscale images...")
    
    reg_dir = output_dir / "registration"
    temp_dir = reg_dir / "TEMP"
    
    # Clean slate
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use minimal configuration that DeepHistReg can handle
        config = {
            'source_path': str(he_gray_path),
            'target_path': str(if_gray_path),
            'output_path': str(reg_dir),
            'case_name': case_name,
            'temporary_path': str(temp_dir),
            'delete_temporary_results': False,  # Keep for debugging
            'save_displacement_field': True,
            'copy_target': True,
        }
        
        # Don't override registration_parameters - let DeepHistReg use defaults
        # This avoids the parameter conflicts that cause pad_value errors
        
        print(f"  Minimal config: {config}")
        
        start_time = time.time()
        deeperhistreg.run_registration(**config)
        elapsed = time.time() - start_time
        
        print(f"  ✅ DeepHistReg completed in {elapsed:.1f} seconds")
        return True, elapsed, None
        
    except Exception as e:
        print(f"  ❌ DeepHistReg failed: {e}")
        
        # Debug output
        print("  Debug - Registration directory contents:")
        if reg_dir.exists():
            for item in reg_dir.rglob("*"):
                if item.is_file():
                    print(f"    {item.relative_to(reg_dir)} ({item.stat().st_size} bytes)")
        
        print("  Debug - Temp directory contents:")
        if temp_dir.exists():
            for item in temp_dir.rglob("*"):
                if item.is_file():
                    print(f"    {item.relative_to(temp_dir)} ({item.stat().st_size} bytes)")
        
        return False, 0, str(e)

def find_displacement_field_simple(reg_dir, temp_dir, case_name='grayscale_reg'):
    """
    Find displacement field with simple search
    """
    search_locations = [
        temp_dir / f'{case_name}_displacement_field.npy',
        temp_dir / 'displacement_field.npy',
        reg_dir / f'{case_name}_displacement_field.npy',
        reg_dir / 'displacement_field.npy',
    ]
    
    # Also search in any subdirectories created by DeepHistReg
    for subdir in [reg_dir, temp_dir]:
        if subdir.exists():
            for npy_file in subdir.rglob("*.npy"):
                if 'displacement' in npy_file.name or 'deformation' in npy_file.name:
                    search_locations.append(npy_file)
    
    for path in search_locations:
        if path.exists():
            print(f"   Found displacement field: {path}")
            try:
                # Test loading
                disp = np.load(str(path))
                print(f"   Displacement shape: {disp.shape}")
                return path
            except Exception as e:
                print(f"   Cannot load {path}: {e}")
    
    return None

def apply_displacement_to_original_color(original_color, displacement_field, target_shape):
    """
    Apply displacement field computed from grayscale to original color H&E
    """
    print("   Applying displacement field to original color image...")
    
    h, w = target_shape[:2]
    
    # Handle different displacement field formats
    if displacement_field.ndim == 3:
        if displacement_field.shape[0] == 2:  # (2, H, W)
            flow = displacement_field.transpose(1, 2, 0)
        elif displacement_field.shape[2] == 2:  # (H, W, 2)
            flow = displacement_field
        else:
            print(f"   Unexpected displacement shape: {displacement_field.shape}")
            return None
    else:
        print(f"   Unexpected displacement dimensions: {displacement_field.shape}")
        return None
    
    # Resize flow if needed
    if flow.shape[:2] != (h, w):
        print(f"   Resizing flow from {flow.shape[:2]} to {(h, w)}")
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate maps
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[:, :, 0]).astype(np.float32)
    map_y = (y + flow[:, :, 1]).astype(np.float32)
    
    # Apply transformation to color image
    warped_color = cv2.remap(
        original_color, map_x, map_y, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped_color

#############################################################################
# MAIN GRAYSCALE REGISTRATION FUNCTION
#############################################################################

def register_qptiff_grayscale(he_qptiff_path: Path, if_qptiff_path: Path, 
                             output_dir: Path, if_channels: list = None) -> Dict:
    """
    Register QPTIFF using grayscale for compatibility, apply to color for output
    """
    print("\n" + "="*70)
    print(" GRAYSCALE QPTIFF REGISTRATION (DEEPHISTREG COMPATIBLE)")
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
        # Step 1: Load IF and create grayscale
        print("\n1. Loading IF and creating optimal grayscale...")
        import tifffile
        
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            # Handle 4D data
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            # Create grayscale from IF channels
            if_gray = create_optimal_if_grayscale(if_data, if_channels)
            print(f"   IF grayscale shape: {if_gray.shape}")
            
            # Enhance for registration
            if_gray_enhanced = enhance_grayscale_for_registration(if_gray)
            
            target_shape = (*if_gray.shape, 1)  # Store target shape
        
        # Step 2: Load H&E and create grayscale
        print("\n2. Loading H&E and creating optimal grayscale...")
        
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_data = tif.pages[0].asarray()
            print(f"   H&E shape: {he_data.shape}")
            
            # Handle planar configuration
            if he_data.ndim == 3 and he_data.shape[0] == 3:
                he_data = np.transpose(he_data, (1, 2, 0))
            
            # Ensure RGB and uint8
            if he_data.ndim == 2:
                he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
            
            if he_data.dtype != np.uint8:
                if he_data.dtype == np.uint16:
                    he_data = (he_data / 256).astype(np.uint8)
                else:
                    he_data = he_data.astype(np.uint8)
            
            # Store original color H&E
            he_original_color = he_data.copy()
            
            # Resize to match IF
            if he_data.shape[:2] != if_gray.shape:
                print(f"   Resizing H&E from {he_data.shape} to match IF {if_gray.shape}")
                he_data = cv2.resize(he_data, (if_gray.shape[1], if_gray.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original_color = cv2.resize(he_original_color, (if_gray.shape[1], if_gray.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            
            # Create grayscale from color H&E
            he_gray = create_optimal_grayscale_from_color(he_data, method='weighted')
            print(f"   H&E grayscale shape: {he_gray.shape}")
            
            # Enhance for registration
            he_gray_enhanced = enhance_grayscale_for_registration(he_gray)
        
        # Step 3: Save grayscale images for DeepHistReg
        print("\n3. Saving grayscale images for registration...")
        
        he_gray_path = reg_dir / "he_grayscale.tiff"
        if_gray_path = reg_dir / "if_grayscale.tiff"
        
        # Save as grayscale TIFF (single channel)
        tifffile.imwrite(he_gray_path, he_gray_enhanced, compression='lzw')
        tifffile.imwrite(if_gray_path, if_gray_enhanced, compression='lzw')
        
        print(f"   Grayscale H&E saved: {he_gray_path}")
        print(f"   Grayscale IF saved: {if_gray_path}")
        
        # Step 4: Run DeepHistReg on grayscale images
        print("\n4. Running DeepHistReg on grayscale images...")
        
        success, elapsed, error = run_deephistreg_grayscale(
            he_gray_path, if_gray_path, output_dir, 'grayscale_reg'
        )
        
        if success:
            print(f"   ✅ Grayscale registration completed")
            results['registration_success'] = True
            results['elapsed_time'] = elapsed
        else:
            print(f"   ❌ Grayscale registration failed: {error}")
            results['registration_success'] = False
            results['registration_error'] = error
        
        # Step 5: Find displacement field and apply to color
        print("\n5. Applying grayscale transformation to color H&E...")
        
        disp_field_path = find_displacement_field_simple(reg_dir, temp_dir, 'grayscale_reg')
        
        if disp_field_path:
            try:
                displacement_field = np.load(str(disp_field_path))
                print(f"   Loaded displacement field: {displacement_field.shape}")
                
                # Apply to original color H&E
                warped_color = apply_displacement_to_original_color(
                    he_original_color, displacement_field, target_shape
                )
                
                if warped_color is not None:
                    # Save final color result
                    final_output_path = output_dir / "registered_HE_color_from_grayscale.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Color result saved: {final_output_path}")
                    results['success'] = True
                    results['registered_path'] = final_output_path
                    results['method'] = 'grayscale_to_color'
                    results['output_shape'] = warped_color.shape
                else:
                    print("   ❌ Failed to apply displacement to color image")
                    results['success'] = False
                    results['error'] = "Displacement application failed"
                
            except Exception as e:
                print(f"   ❌ Error processing displacement field: {e}")
                results['success'] = False
                results['error'] = f"Displacement processing failed: {e}"
        else:
            print("   ❌ No displacement field found")
            results['success'] = False
            results['error'] = "No displacement field found"
        
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
    """Main execution with grayscale compatibility"""
    import argparse
    
    print("\n" + "="*70)
    print(" GRAYSCALE QPTIFF REGISTRATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Grayscale-compatible QPTIFF registration")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default="./output_grayscale", help="Output directory")
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
    
    # Run grayscale-compatible registration
    results = register_qptiff_grayscale(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✅ Registration completed successfully!")
        print(f"   Method: {results.get('method', 'unknown')}")
        if 'elapsed_time' in results:
            print(f"   Time: {results['elapsed_time']:.1f} seconds")
        print(f"   Output: {results['registered_path']}")
        if 'output_shape' in results:
            print(f"   Dimensions: {results['output_shape']}")
    else:
        print(f"❌ Registration failed: {results.get('error', 'Unknown error')}")
        if 'registration_error' in results:
            print(f"   DeepHistReg error: {results['registration_error']}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
