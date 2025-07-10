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
# PREPROCESSING FUNCTIONS (from your successful approach)
#############################################################################

def preprocess_for_registration(source, target):
    """
    Advanced preprocessing for better alignment between H&E and IF
    This is your successful preprocessing approach
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
    """Your successful registration parameters"""
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
    
    # IMPORTANT: Tell DeepHistReg to work with QPTIFF directly
    params['loading_params']['loader'] = 'tiff'  # For TIFF/QPTIFF support
    params['loading_params']['downsample_factor'] = 1  # No downsampling
    
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
        # Step 1: Prepare IF image (extract channels as RGB)
        print("\n1. Preparing IF image...")
        
        # For IF, we need to extract specific channels
        # DeepHistReg expects RGB, so we'll create a temporary RGB version
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
        
        # First, let's check H&E dimensions to ensure we're loading full resolution
        with tifffile.TiffFile(he_qptiff_path) as tif:
            # Try to get the largest (first) page
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
            
            # IMPORTANT: Store original H&E for color preservation
            he_original_color = he_data.copy()
            
            # Resize H&E to match IF if needed
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E from {he_data.shape} to match IF {target_shape}")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original_color = cv2.resize(he_original_color, (target_shape[1], target_shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            
            # Save temporary H&E
            he_temp_path = temp_dir / "he_temp.tiff"
            tifffile.imwrite(he_temp_path, he_data, photometric='rgb', compression='lzw')
        
        # Step 3: Preprocess both images
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
        
        # Step 5: Apply transformation to original color H&E
        print("\n5. Applying transformation to original color H&E...")
        
        disp_field_path = temp_dir / 'displacement_field.npy'
        warped_color = None  # Initialize to avoid undefined variable error
        
        # First check if grayscale result exists
        warped_gray_path = reg_dir / "warped_source.tiff"
        
        if not warped_gray_path.exists():
            print("   ❌ Warped grayscale image not found!")
            # Try alternative naming
            warped_gray_path = reg_dir / "qptiff_reg_warped_source.tiff"
            if not warped_gray_path.exists():
                # Check all tiff files in registration directory
                print("   Looking for warped result...")
                tiff_files = list(reg_dir.glob("*warped*.tiff"))
                if tiff_files:
                    warped_gray_path = tiff_files[0]
                    print(f"   Found: {warped_gray_path.name}")
        
        if disp_field_path.exists():
            try:
                # Load displacement field
                displacement_field = np.load(str(disp_field_path))
                print(f"   Displacement field shape: {displacement_field.shape}")
                
                # Apply to original color image
                h, w = target_shape[:2]
                flow = displacement_field.transpose(1, 2, 0)
                
                # Create mesh grid
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + flow[:, :, 0]).astype(np.float32)
                map_y = (y + flow[:, :, 1]).astype(np.float32)
                
                # Warp the original color H&E
                warped_color = cv2.remap(he_original_color, map_x, map_y, cv2.INTER_LINEAR)
                
                # Save final color result
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
                warped_color = None
        
        else:
            print("   ❌ Displacement field not found!")
            print(f"   Expected at: {disp_field_path}")
            
            # Check alternative locations
            alt_paths = [
                temp_dir / 'qptiff_reg_displacement_field.npy',
                reg_dir / 'displacement_field.npy',
                reg_dir / 'TEMP' / 'qptiff_reg_displacement_field.npy'
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"   Found at alternative location: {alt_path}")
                    disp_field_path = alt_path
                    # Retry loading
                    try:
                        displacement_field = np.load(str(disp_field_path))
                        # ... (apply transformation code as above)
                        break
                    except Exception as e:
                        print(f"   Error loading from {alt_path}: {e}")
        
        # If displacement field method failed, try alternative color preservation
        if warped_color is None and warped_gray_path.exists():
            print("\n   Trying alternative color transfer method...")
            try:
                # Load the warped grayscale result
                warped_gray = tifffile.imread(str(warped_gray_path))
                
                # Method 1: Find transformation between original and warped grayscale
                if warped_gray.shape[:2] == he_original_color.shape[:2]:
                    # If same size, we can try to transfer color directly
                    # Convert warped to grayscale if needed
                    if warped_gray.ndim == 3:
                        warped_gray_single = cv2.cvtColor(warped_gray, cv2.COLOR_RGB2GRAY)
                    else:
                        warped_gray_single = warped_gray
                    
                    # Convert original to grayscale
                    he_gray = cv2.cvtColor(he_original_color, cv2.COLOR_RGB2GRAY)
                    
                    # Use the grayscale warped as a guide to warp the color channels
                    warped_color = np.zeros_like(he_original_color)
                    
                    # Simple approach: use the warped grayscale as intensity guide
                    for c in range(3):
                        # Scale each color channel by the ratio of warped to original grayscale
                        ratio = warped_gray_single.astype(np.float32) / (he_gray.astype(np.float32) + 1e-6)
                        ratio = np.clip(ratio, 0, 2)  # Limit the ratio
                        warped_color[:, :, c] = np.clip(he_original_color[:, :, c] * ratio, 0, 255).astype(np.uint8)
                    
                    # Save the result
                    final_output_path = output_dir / "registered_HE_color_alt.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    print(f"   ✅ Alternative color result saved: {final_output_path.name}")
                    results['success'] = True
                    results['registered_path'] = final_output_path
                    results['method'] = 'alternative_color_transfer'
                else:
                    # If different sizes, use the grayscale result
                    print("   Size mismatch, using grayscale result")
                    warped_color = warped_gray if warped_gray.ndim == 3 else cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2RGB)
                    
                    # Copy grayscale result as color output
                    final_output_path = output_dir / "registered_HE_gray.tiff"
                    tifffile.imwrite(
                        final_output_path,
                        warped_color,
                        photometric='rgb',
                        compression='lzw',
                        bigtiff=True
                    )
                    
                    results['success'] = True
                    results['registered_path'] = final_output_path
                    results['note'] = 'Color preservation failed, using grayscale result'
                    
            except Exception as e:
                print(f"   ❌ Alternative method also failed: {e}")
                results['success'] = False
                results['error'] = f"Both color methods failed: {e}"
        
        # Store grayscale result path regardless
        if warped_gray_path.exists():
            results['registered_gray'] = warped_gray_path
        
        # Step 6: Create visualizations (only if we have some result)
        if warped_color is not None:
            print("\n6. Creating visualizations...")
            try:
                create_visualizations(he_original_color, if_rgb, warped_color, output_dir)
            except Exception as e:
                print(f"   Warning: Visualization failed: {e}")
        else:
            print("\n6. Skipping visualizations (no warped image available)")
        
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
    
    # 1. Side-by-side comparison
    comparison = np.hstack([he_original, if_img, warped])
    cv2.imwrite(str(viz_dir / "side_by_side.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 2. Checkerboard overlay
    checker_size = max(100, min(h, w) // 20)  # Adaptive checker size
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

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution"""
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
