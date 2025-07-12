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
    params['loading_params'] = {
        'loader': 'tiff',
        'downsample_factor': 1,
    }
    
    # CRITICAL: Ensure displacement field is saved
    params['save_displacement_field'] = True
    params['save_deformation_field'] = True  # Alternative name
    params['save_transform_parameters'] = True
    params['output_displacement_field'] = True
    
    return params

#############################################################################
# ENHANCED DISPLACEMENT FIELD HANDLING AND ROBUST TIFF LOADING
#############################################################################

def find_displacement_field_comprehensive(reg_dir, temp_dir, case_name='qptiff_reg'):
    """
    Comprehensive search for displacement field including .mha files
    """
    print("   Comprehensive displacement field search...")
    
    # All possible names and locations (including .mha files)
    possible_locations = [reg_dir, temp_dir, temp_dir / case_name, reg_dir / case_name]
    possible_names = [
        'displacement_field.mha',      # ITK/SimpleITK format - MOST LIKELY
        'displacement_field.npy',
        f'{case_name}_displacement_field.mha',
        f'{case_name}_displacement_field.npy',
        'deformation_field.mha',
        'deformation_field.npy',
        f'{case_name}_deformation_field.mha',
        f'{case_name}_deformation_field.npy',
        'transform_field.mha',
        'displacement.mha',
        'deformation.mha',
        'field.mha',
        'transform_field.npy',
        'displacement.npy',
        'deformation.npy',
        'field.npy',
    ]
    
    # Search in all combinations
    for location in possible_locations:
        if not location.exists():
            continue
        for name in possible_names:
            path = location / name
            if path.exists():
                print(f"   Found displacement field: {path}")
                return path
    
    # If not found by name, search all relevant files by extension
    print("   Searching by file extension...")
    relevant_extensions = ['.mha', '.mhd', '.npy', '.nii', '.nii.gz']
    for location in possible_locations:
        if not location.exists():
            continue
        for ext in relevant_extensions:
            for file in location.glob(f"*{ext}"):
                if any(keyword in file.name.lower() for keyword in ['displacement', 'deformation', 'field', 'transform']):
                    print(f"   Found potential displacement field by name: {file}")
                    return file
    
    print("   No displacement field found")
    return None

def load_displacement_field_from_mha(mha_path):
    """
    Load displacement field from .mha file using SimpleITK
    """
    print(f"   Loading .mha displacement field: {mha_path}")
    
    try:
        import SimpleITK as sitk
        
        # Load the displacement field
        displacement_image = sitk.ReadImage(str(mha_path))
        
        # Convert to numpy array
        displacement_array = sitk.GetArrayFromImage(displacement_image)
        
        print(f"   Original .mha shape: {displacement_array.shape}")
        print(f"   Original .mha spacing: {displacement_image.GetSpacing()}")
        print(f"   Original .mha direction: {displacement_image.GetDirection()}")
        
        # ITK/SimpleITK typically stores displacement fields as (Z, Y, X, Components)
        # or (Y, X, Components) for 2D
        # We need to convert to (Y, X, 2) or (2, Y, X) format for OpenCV
        
        if displacement_array.ndim == 4:
            # 3D case: (Z, Y, X, Components) -> take middle Z slice
            z_middle = displacement_array.shape[0] // 2
            displacement_2d = displacement_array[z_middle, :, :, :2]  # Take X,Y components
            print(f"   Extracted 2D slice from 3D field: {displacement_2d.shape}")
        elif displacement_array.ndim == 3:
            # 2D case: (Y, X, Components)
            displacement_2d = displacement_array[:, :, :2]  # Take X,Y components
            print(f"   Using 2D field: {displacement_2d.shape}")
        else:
            raise ValueError(f"Unexpected displacement field dimensions: {displacement_array.shape}")
        
        # Convert from ITK coordinate system to OpenCV
        # ITK: (X, Y) components in physical coordinates
        # OpenCV: (X, Y) components in pixel coordinates
        
        # Flip Y component if needed (ITK vs OpenCV coordinate system differences)
        displacement_opencv = displacement_2d.copy()
        displacement_opencv[:, :, 1] = -displacement_opencv[:, :, 1]  # Flip Y
        
        print(f"   Final displacement shape for OpenCV: {displacement_opencv.shape}")
        print(f"   Displacement range X: [{displacement_opencv[:,:,0].min():.2f}, {displacement_opencv[:,:,0].max():.2f}]")
        print(f"   Displacement range Y: [{displacement_opencv[:,:,1].min():.2f}, {displacement_opencv[:,:,1].max():.2f}]")
        
        return displacement_opencv
        
    except ImportError:
        print("   ❌ SimpleITK not available. Installing...")
        try:
            import subprocess
            subprocess.check_call(['pip', 'install', 'SimpleITK'])
            print("   ✅ SimpleITK installed, retrying...")
            
            # Retry after installation
            import SimpleITK as sitk
            displacement_image = sitk.ReadImage(str(mha_path))
            displacement_array = sitk.GetArrayFromImage(displacement_image)
            
            if displacement_array.ndim == 4:
                z_middle = displacement_array.shape[0] // 2
                displacement_2d = displacement_array[z_middle, :, :, :2]
            elif displacement_array.ndim == 3:
                displacement_2d = displacement_array[:, :, :2]
            else:
                raise ValueError(f"Unexpected displacement field dimensions: {displacement_array.shape}")
            
            displacement_opencv = displacement_2d.copy()
            displacement_opencv[:, :, 1] = -displacement_opencv[:, :, 1]  # Flip Y
            
            return displacement_opencv
            
        except Exception as install_error:
            print(f"   ❌ Could not install SimpleITK: {install_error}")
            return None
    
    except Exception as e:
        print(f"   ❌ Error loading .mha file: {e}")
        return None



def try_alternative_displacement_extraction(reg_dir, temp_dir, case_name):
    """
    Try to extract displacement information from other DeepHistReg outputs
    """
    print("   Attempting to extract displacement from other outputs...")
    
    # Look for ITK transform files
    possible_extensions = ['.tfm', '.txt', '.h5']
    for ext in possible_extensions:
        for location in [reg_dir, temp_dir, temp_dir / case_name]:
            if not location.exists():
                continue
            for transform_file in location.glob(f"*{ext}"):
                print(f"   Found transform file: {transform_file}")
                # For now, just note it exists - would need ITK to parse
                
    # Look for warped images that we can use directly
    warped_files = []
    for location in [reg_dir, temp_dir, temp_dir / case_name]:
        if not location.exists():
            continue
        warped_files.extend(location.glob("*warped*"))
        warped_files.extend(location.glob("*registered*"))
        warped_files.extend(location.glob("*result*"))
    
    print(f"   Found {len(warped_files)} potential result files")
    for wf in warped_files:
        print(f"     {wf.name} ({wf.stat().st_size} bytes)")
    
    return warped_files


                try:
                    # This would require SimpleITK or similar
                    print(f"   Found displacement field in {ext} format: {disp_file.name}")
                    return disp_file
                except Exception as e:
                    print(f"   Could not load {disp_file.name}: {e}")
    
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

def load_warped_result_robust(warped_path):
    """
    Robust loading of warped result with multiple fallback methods
    """
    print(f"   Attempting to load: {warped_path}")
    
    # Method 1: Try tifffile first
    try:
        print("   Trying tifffile...")
        warped = tifffile.imread(str(warped_path))
        print(f"   ✅ Success with tifffile, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ tifffile failed: {e}")
    
    # Method 2: Try OpenCV
    try:
        print("   Trying OpenCV...")
        warped = cv2.imread(str(warped_path), cv2.IMREAD_COLOR)
        if warped is not None:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            print(f"   ✅ Success with OpenCV, shape: {warped.shape}")
            return warped
        else:
            print("   ❌ OpenCV returned None")
    except Exception as e:
        print(f"   ❌ OpenCV failed: {e}")
    
    # Method 3: Try PIL
    try:
        print("   Trying PIL...")
        from PIL import Image
        pil_img = Image.open(str(warped_path))
        warped = np.array(pil_img)
        print(f"   ✅ Success with PIL, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ PIL failed: {e}")
    
    # Method 4: Try with specific tifffile parameters
    try:
        print("   Trying tifffile with specific parameters...")
        with tifffile.TiffFile(str(warped_path)) as tif:
            print(f"   TIFF info: {len(tif.pages)} pages")
            for i, page in enumerate(tif.pages):
                print(f"   Page {i}: {page.shape}, {page.dtype}, compression: {page.compression}")
            
            # Try loading the first page
            warped = tif.pages[0].asarray()
            print(f"   ✅ Success with specific parameters, shape: {warped.shape}")
            return warped
    except Exception as e:
        print(f"   ❌ Specific tifffile failed: {e}")
    
    # Method 5: Try skimage
    try:
        print("   Trying skimage...")
        from skimage import io
        warped = io.imread(str(warped_path))
        print(f"   ✅ Success with skimage, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ skimage failed: {e}")
    
    # Method 6: Try reading as binary and reconstructing
    try:
        print("   Trying binary reconstruction...")
        with open(warped_path, 'rb') as f:
            # Read first few bytes to check format
            header = f.read(16)
            print(f"   File header: {header}")
            
        # Try using imageio
        import imageio
        warped = imageio.imread(str(warped_path))
        print(f"   ✅ Success with imageio, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ imageio failed: {e}")
    
    # Method 7: Try with different tifffile engines
    try:
        print("   Trying tifffile with different options...")
        # Try with different options
        warped = tifffile.imread(str(warped_path), is_ome=False)
        print(f"   ✅ Success with non-OME mode, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ Non-OME mode failed: {e}")
    
    # Method 8: Try to convert the file first
    try:
        print("   Trying file conversion...")
        # Create a temporary converted file
        temp_path = warped_path.parent / f"temp_converted_{warped_path.stem}.tiff"
        
        # Use PIL to open and re-save in a compatible format
        from PIL import Image
        with Image.open(str(warped_path)) as img:
            img.save(str(temp_path), 'TIFF', compression='lzw')
        
        # Try to load the converted file
        warped = tifffile.imread(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        print(f"   ✅ Success with conversion, shape: {warped.shape}")
        return warped
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
    
    print("   ❌ All loading methods failed")
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
        
        # Debug: Check what files were created
        print(f"\n   Debug: Files created in registration directory:")
        for file in sorted(reg_dir.rglob("*")):
            if file.is_file():
                print(f"     {file.relative_to(reg_dir)} ({file.stat().st_size} bytes)")
        
        print(f"\n   Debug: Files created in temp directory:")
        for file in sorted(temp_dir.rglob("*")):
            if file.is_file():
                print(f"     {file.relative_to(temp_dir)} ({file.stat().st_size} bytes)")
        
        # Look for ANY displacement/deformation field files
        print(f"\n   Debug: Searching for displacement fields...")
        all_npy_files = list(reg_dir.rglob("*.npy")) + list(temp_dir.rglob("*.npy"))
        for npy_file in all_npy_files:
            try:
                data = np.load(str(npy_file))
                print(f"     NPY file: {npy_file.name}, shape: {data.shape}, dtype: {data.dtype}")
            except Exception as e:
                print(f"     NPY file: {npy_file.name} (error loading: {e})")
        
        # Look for transform files
        transform_exts = ['.txt', '.tfm', '.h5', '.mat']
        for ext in transform_exts:
            transform_files = list(reg_dir.rglob(f"*{ext}")) + list(temp_dir.rglob(f"*{ext}"))
            for tf in transform_files:
                print(f"     Transform file: {tf.name} ({tf.stat().st_size} bytes)")
        
        # Step 5: Apply transformation with improved handling
        print("\n5. Applying transformation to original color H&E...")
        
        # Step 5: Apply transformation to original color H&E
        print("\n5. Applying transformation to original color H&E...")
        
        # Try comprehensive displacement field search
        disp_field_path = find_displacement_field_comprehensive(reg_dir, temp_dir, 'qptiff_color_reg')
        
        warped_color = None
        
        if disp_field_path:
            try:
                # Load displacement field based on file extension
                if disp_field_path.suffix.lower() == '.mha':
                    displacement_field = load_displacement_field_from_mha(disp_field_path)
                elif disp_field_path.suffix.lower() == '.npy':
                    displacement_field = np.load(str(disp_field_path))
                    print(f"   Displacement field shape: {displacement_field.shape}")
                else:
                    print(f"   Unsupported displacement field format: {disp_field_path.suffix}")
                    displacement_field = None
                
                if displacement_field is not None:
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
        
        # If displacement field method failed, try alternative extraction
        if warped_color is None:
            warped_files = try_alternative_displacement_extraction(reg_dir, temp_dir, 'qptiff_color_reg')
            
            # Try to load each warped file with robust methods
            for warped_file in warped_files:
                if 'tiff' in warped_file.suffix.lower():
                    print(f"\n   Trying to load: {warped_file.name}")
                    loaded_result = load_warped_result_robust(warped_file)
                    
                    if loaded_result is not None:
                        # Resize to match original IF dimensions if needed
                        if loaded_result.shape[:2] != target_shape[:2]:
                            loaded_result = cv2.resize(loaded_result, (target_shape[1], target_shape[0]))
                        
                        # Save result
                        final_output_path = output_dir / f"registered_HE_from_{warped_file.stem}.tiff"
                        tifffile.imwrite(
                            final_output_path,
                            loaded_result,
                            photometric='rgb',
                            compression='lzw',
                            bigtiff=True
                        )
                        
                        print(f"   ✅ Successfully saved result from {warped_file.name}")
                        warped_color = loaded_result
                        results['success'] = True
                        results['registered_path'] = final_output_path
                        results['method'] = f'direct_from_{warped_file.stem}'
                        results['preprocessing_method'] = preprocessing_method
                        results['elapsed_time'] = elapsed
                        break
        
        if warped_color is None:
            print("   ❌ No usable registration result found")
            print("   Available files for debugging:")
            for file in sorted(reg_dir.rglob("*")):
                if file.is_file():
                    print(f"     {file.name}: {file.stat().st_size} bytes")
            results['success'] = False
            results['error'] = "No displacement field or usable warped result found"
        
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
