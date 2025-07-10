#!/usr/bin/env python3
"""
Complete H&E to IF Registration Pipeline for QPTIFF Images
===========================================================
Designed for vast.ai with NVIDIA H200 GPU

This script combines QPTIFF preprocessing with DeepHistReg registration
for high-resolution whole slide images.

Features:
- Handles large QPTIFF files (H&E: 3 channels, IF: 8 channels)
- Multi-resolution pyramid extraction
- GPU-accelerated registration with DeepHistReg
- Color preservation in final output
- Comprehensive evaluation metrics

Author: Combined from user's pipelines
Date: 2025-01-09
"""

### System Imports ###
import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, Optional, Dict, List

### Scientific Computing ###
import numpy as np
import torch as tc

### Image Processing ###
import cv2
from PIL import Image
import tifffile
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

### DeepHistReg ###
# Handle PyVIPS import issues
try:
    import deeperhistreg
    from deeperhistreg.dhr_input_output.dhr_loaders import pil_loader, tiff_loader
    from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid
    DEEPHISTREG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: DeepHistReg import failed: {e}")
    print("   Attempting workaround...")
    DEEPHISTREG_AVAILABLE = False
    
    # Try to fix PyVIPS issue
    import os
    os.environ['PYVIPS_VIPS'] = '0'  # Disable PyVIPS in DeepHistReg
    
    try:
        import deeperhistreg
        from deeperhistreg.dhr_input_output.dhr_loaders import pil_loader, tiff_loader
        from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid
        DEEPHISTREG_AVAILABLE = True
        print("   ✅ Workaround successful - DeepHistReg loaded without PyVIPS")
    except:
        print("   ❌ Workaround failed - Registration will not be available")

#############################################################################
# CONFIGURATION
#############################################################################

# Default paths (modify as needed)
WORK_DIR = Path("/workspace")  # vast.ai default workspace
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
TEMP_DIR = WORK_DIR / "temp"

# Registration parameters
DEFAULT_RESOLUTION_LEVEL = 1  # 0=full, 1=10x, 2=20x, etc.
DEFAULT_IF_CHANNELS = [0, 1, 5]  # DAPI, CD8, CD163 (adjust as needed)

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

#############################################################################
# UTILITY FUNCTIONS
#############################################################################

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def format_size(bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

def check_gpu():
    """Check GPU availability"""
    if tc.cuda.is_available():
        gpu_name = tc.cuda.get_device_name(0)
        gpu_memory = tc.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("❌ No GPU detected! Registration will be slow.")
        return False

#############################################################################
# QPTIFF LOADING AND PREPROCESSING
#############################################################################

class QPTIFFLoader:
    """Handle QPTIFF loading and preprocessing"""
    
    def __init__(self, resolution_level: int = 1, use_openslide: bool = True):
        self.resolution_level = resolution_level
        self.use_openslide = use_openslide
        
        # Try to import optional libraries
        self.openslide_available = False
        self.pyvips_available = False
        
        try:
            import openslide
            self.openslide = openslide
            self.openslide_available = True
            print("✅ OpenSlide available for WSI reading")
        except ImportError:
            print("⚠️  OpenSlide not available, using tifffile")
            
        try:
            import pyvips
            self.pyvips = pyvips
            self.pyvips_available = True
            print("✅ PyVIPS available for large image handling")
        except ImportError:
            print("⚠️  PyVIPS not available")
    
    def load_with_openslide(self, slide_path: Path, level: int) -> np.ndarray:
        """Load slide using OpenSlide (faster for WSI)"""
        slide = self.openslide.OpenSlide(str(slide_path))
        
        # Get dimensions at specified level
        width, height = slide.level_dimensions[level]
        print(f"  Level {level} dimensions: {width} x {height}")
        
        # Read the whole level
        region = slide.read_region((0, 0), level, (width, height))
        
        # Convert to numpy array and remove alpha channel
        img_array = np.array(region)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        slide.close()
        return img_array
    
    def load_with_pyvips(self, image_path: Path, level: int) -> np.ndarray:
        """Load image using PyVIPS (memory efficient)"""
        image = self.pyvips.Image.new_from_file(str(image_path), 
                                               access='sequential',
                                               page=level)
        
        # Convert to numpy
        mem_img = image.write_to_memory()
        img_array = np.frombuffer(mem_img, dtype=np.uint8).reshape(
            image.height, image.width, image.bands)
        
        return img_array
        
    def load_he_qptiff(self, qptiff_path: Path, output_dir: Path) -> Tuple[np.ndarray, Path]:
        """Load H&E QPTIFF and extract specified resolution level"""
        print_section(f"Loading H&E QPTIFF: {qptiff_path.name}")
        
        try:
            # Try OpenSlide first if available
            if self.use_openslide and self.openslide_available:
                try:
                    print("  Using OpenSlide for loading...")
                    data = self.load_with_openslide(qptiff_path, self.resolution_level)
                    print(f"  Loaded with OpenSlide: shape {data.shape}")
                except Exception as e:
                    print(f"  OpenSlide failed: {e}, falling back to tifffile")
                    data = None
            else:
                data = None
            
            # Fallback to tifffile
            if data is None:
                with tifffile.TiffFile(qptiff_path) as tif:
                    # Check if it's a pyramid
                    if len(tif.pages) > 1:
                        print(f"  Pyramid detected with {len(tif.pages)} levels")
                        
                        # Get the specified resolution level
                        if self.resolution_level >= len(tif.pages):
                            print(f"  Warning: Requested level {self.resolution_level} not available")
                            self.resolution_level = len(tif.pages) - 1
                        
                        page = tif.pages[self.resolution_level]
                        print(f"  Loading resolution level {self.resolution_level}")
                    else:
                        print("  Single resolution image")
                        page = tif.pages[0]
                    
                    # Read the data
                    data = page.asarray()
                    print(f"  Original shape: {data.shape}, dtype: {data.dtype}")
                    
                    # Handle planar configuration
                    if data.ndim == 3 and data.shape[0] == 3:
                        print("  Converting from planar to interleaved format...")
                        data = np.transpose(data, (1, 2, 0))
                
            # Ensure 3 channels
            if data.ndim == 2:
                print("  Converting grayscale to RGB...")
                data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            
            # Save as temporary file for processing
            temp_path = output_dir / f"he_level{self.resolution_level}_temp.tiff"
            print(f"  Saving temporary file: {temp_path.name}")
            tifffile.imwrite(
                temp_path,
                data,
                photometric='rgb',
                compression='lzw',
                bigtiff=True
            )
            
            print(f"  ✅ H&E loaded: shape {data.shape}, {format_size(temp_path.stat().st_size)}")
            return data, temp_path
                
        except Exception as e:
            print(f"  ❌ Error loading H&E: {e}")
            raise
    
    def load_if_qptiff(self, qptiff_path: Path, output_dir: Path, 
                       channels: List[int] = None) -> Tuple[np.ndarray, Path]:
        """Load IF QPTIFF and extract specified channels"""
        print_section(f"Loading IF QPTIFF: {qptiff_path.name}")
        
        if channels is None:
            channels = DEFAULT_IF_CHANNELS
            
        try:
            with tifffile.TiffFile(qptiff_path) as tif:
                # Get the first page/series
                data = tif.asarray()
                print(f"  Original shape: {data.shape}, dtype: {data.dtype}")
                
                # Handle different dimension arrangements
                if data.ndim == 4:  # Z, C, Y, X or C, Z, Y, X
                    if data.shape[0] > data.shape[1]:  # Likely C, Z, Y, X
                        print("  Taking first Z-plane from 4D data...")
                        data = data[:, 0, :, :]
                    else:  # Z, C, Y, X
                        print("  Taking first Z-plane from 4D data...")
                        data = data[0]
                
                # Now should be C, Y, X
                if data.ndim != 3:
                    raise ValueError(f"Unexpected IF data shape: {data.shape}")
                
                # Extract specified channels
                print(f"  Extracting channels: {channels}")
                selected_channels = []
                channel_names = ['DAPI', 'CD8', 'CD163', 'PD1', 'PDL1', 'CD68', 'CK', 'Ki67']
                
                for idx in channels:
                    if idx < data.shape[0]:
                        selected_channels.append(data[idx])
                        if idx < len(channel_names):
                            print(f"    Channel {idx}: {channel_names[idx]}")
                        else:
                            print(f"    Channel {idx}")
                    else:
                        print(f"    Warning: Channel {idx} not available")
                
                # Stack selected channels
                if_stack = np.stack(selected_channels, axis=0)
                print(f"  Stacked shape: {if_stack.shape}")
                
                # Convert to RGB for DeepHistReg (using first 3 channels)
                if len(selected_channels) >= 3:
                    # Normalize each channel to 0-255
                    rgb_channels = []
                    for i in range(3):
                        ch = selected_channels[i]
                        ch_norm = ((ch - ch.min()) / (ch.max() - ch.min()) * 255).astype(np.uint8)
                        rgb_channels.append(ch_norm)
                    
                    if_rgb = np.stack(rgb_channels, axis=-1)
                else:
                    # If less than 3 channels, repeat to make RGB
                    ch = selected_channels[0]
                    ch_norm = ((ch - ch.min()) / (ch.max() - ch.min()) * 255).astype(np.uint8)
                    if_rgb = np.stack([ch_norm] * 3, axis=-1)
                
                # Save as temporary file
                temp_path = output_dir / f"if_rgb_temp.tiff"
                print(f"  Saving temporary RGB file: {temp_path.name}")
                tifffile.imwrite(
                    temp_path,
                    if_rgb,
                    photometric='rgb',
                    compression='lzw'
                )
                
                # Also save the full stack for later use
                stack_path = output_dir / f"if_stack_temp.tiff"
                tifffile.imwrite(
                    stack_path,
                    if_stack,
                    compression='lzw',
                    bigtiff=True
                )
                
                print(f"  ✅ IF loaded: RGB shape {if_rgb.shape}, {format_size(temp_path.stat().st_size)}")
                return if_rgb, temp_path
                
        except Exception as e:
            print(f"  ❌ Error loading IF: {e}")
            raise

#############################################################################
# REGISTRATION FUNCTIONS (from original code)
#############################################################################

def resize_to_match(source, target, preserve_range=True):
    """Resize source image to match target dimensions"""
    target_shape = target.shape[:2]
    if source.shape[:2] != target_shape:
        print(f"Resizing source from {source.shape} to match target {target.shape}")
        source_resized = resize(source, target_shape, preserve_range=preserve_range, anti_aliasing=True)
        if preserve_range:
            source_resized = source_resized.astype(source.dtype)
        return source_resized
    return source

def preprocess_images_advanced(source, target):
    """Advanced preprocessing for better alignment between different modalities"""
    # First resize source to match target dimensions
    source = resize_to_match(source, target)
    
    # Convert to grayscale for processing
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        source_gray = source.astype(np.uint8)
    
    if target.ndim == 3:
        target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        target_gray = target.astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    source_blur = cv2.GaussianBlur(source_gray, (5, 5), 1.0)
    target_blur = cv2.GaussianBlur(target_gray, (5, 5), 1.0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    source_enhanced = clahe.apply(source_blur)
    target_enhanced = clahe.apply(target_blur)
    
    # Edge enhancement for better feature detection
    source_edges = cv2.Canny(source_enhanced, 30, 100)
    target_edges = cv2.Canny(target_enhanced, 30, 100)
    
    # Combine enhanced and edge information
    source_combined = cv2.addWeighted(source_enhanced, 0.7, source_edges, 0.3, 0)
    target_combined = cv2.addWeighted(target_enhanced, 0.7, target_edges, 0.3, 0)
    
    # Convert back to RGB for DeepHistReg
    source_final = cv2.cvtColor(source_combined, cv2.COLOR_GRAY2RGB)
    target_final = cv2.cvtColor(target_combined, cv2.COLOR_GRAY2RGB)
    
    return source_final, target_final

def create_robust_registration_params():
    """Create robust registration parameters optimized for H200 GPU"""
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
    
    # Nonrigid parameters optimized for GPU
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
    
    # Optimization for multimodal registration
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
    
    params['loading_params']['loader'] = 'pil'
    params['loading_params']['downsample_factor'] = 1
    
    return params

def apply_transformation_to_original(original_path, displacement_field_path, 
                                   target_shape, output_path):
    """Apply displacement field to original high-res image"""
    print_section("Applying transformation to original image")
    
    try:
        # Load displacement field
        displacement_field = np.load(str(displacement_field_path))
        print(f"  Displacement field shape: {displacement_field.shape}")
        
        # Load original image
        with tifffile.TiffFile(original_path) as tif:
            original = tif.asarray()
            
        # Handle planar configuration
        if original.ndim == 3 and original.shape[0] == 3:
            original = np.transpose(original, (1, 2, 0))
        
        # Resize if needed
        if original.shape[:2] != target_shape[:2]:
            print(f"  Resizing original from {original.shape} to {target_shape}")
            original = cv2.resize(original, (target_shape[1], target_shape[0]))
        
        # Apply displacement field
        h, w = target_shape[:2]
        flow = displacement_field.transpose(1, 2, 0)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)
        
        warped = cv2.remap(original, map_x, map_y, cv2.INTER_LINEAR)
        
        # Save result
        result_path = output_path / "warped_he_original.tiff"
        tifffile.imwrite(
            result_path,
            warped,
            photometric='rgb',
            compression='lzw',
            bigtiff=True
        )
        
        print(f"  ✅ Transformation applied: {result_path.name}")
        return result_path
        
    except Exception as e:
        print(f"  ❌ Error applying transformation: {e}")
        return None

def perform_registration_pipeline(he_path: Path, if_path: Path, output_dir: Path,
                                he_original_path: Path = None) -> Dict:
    """Complete registration pipeline"""
    print_section("Performing Registration with DeepHistReg")
    
    # Check if DeepHistReg is available
    if not DEEPHISTREG_AVAILABLE:
        print("  ❌ DeepHistReg is not available due to PyVIPS issues")
        print("  Please run: ./fix_pyvips.sh")
        return {'success': False, 'error': 'DeepHistReg not available'}
    
    # Create output directories
    reg_output = output_dir / "registration"
    reg_output.mkdir(exist_ok=True)
    temp_output = reg_output / "TEMP"
    temp_output.mkdir(exist_ok=True)
    
    # Load images for preprocessing
    he_img = cv2.imread(str(he_path))
    if_img = cv2.imread(str(if_path))
    
    he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
    if_img = cv2.cvtColor(if_img, cv2.COLOR_BGR2RGB)
    
    print(f"  H&E shape: {he_img.shape}")
    print(f"  IF shape: {if_img.shape}")
    
    # Preprocess
    print("  Preprocessing images...")
    he_prep, if_prep = preprocess_images_advanced(he_img, if_img)
    
    # Save preprocessed
    prep_he_path = reg_output / "he_preprocessed.tiff"
    prep_if_path = reg_output / "if_preprocessed.tiff"
    cv2.imwrite(str(prep_he_path), cv2.cvtColor(he_prep, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(prep_if_path), cv2.cvtColor(if_prep, cv2.COLOR_RGB2BGR))
    
    # Get registration parameters
    params = create_robust_registration_params()
    
    # Configure DeepHistReg
    config = {
        'source_path': prep_he_path,
        'target_path': prep_if_path,
        'output_path': reg_output,
        'registration_parameters': params,
        'case_name': 'qptiff_registration',
        'save_displacement_field': True,
        'copy_target': True,
        'delete_temporary_results': False,
        'temporary_path': temp_output
    }
    
    print("  Running DeepHistReg...")
    start_time = time.time()
    
    try:
        deeperhistreg.run_registration(**config)
        elapsed = time.time() - start_time
        print(f"  ✅ Registration completed in {elapsed:.1f} seconds")
        
        # Check results
        warped_path = reg_output / "warped_source.tiff"
        disp_field_path = temp_output / 'displacement_field.npy'
        
        results = {
            'success': True,
            'warped_path': warped_path,
            'displacement_field_path': disp_field_path,
            'elapsed_time': elapsed
        }
        
        # Apply to original if provided
        if he_original_path and disp_field_path.exists():
            print("\n  Applying transformation to original resolution H&E...")
            original_result = apply_transformation_to_original(
                he_original_path, 
                disp_field_path,
                if_img.shape,
                reg_output
            )
            if original_result:
                results['warped_original_path'] = original_result
        
        return results
        
    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def evaluate_registration(target_path: Path, warped_path: Path, 
                         output_dir: Path, num_patches: int = 50):
    """Evaluate registration quality"""
    print_section("Evaluating Registration Quality")
    
    # Load images
    target = cv2.imread(str(target_path))
    warped = cv2.imread(str(warped_path))
    
    if target is None or warped is None:
        print("  ❌ Could not load images for evaluation")
        return
    
    # Convert to RGB
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # Ensure same dimensions
    if warped.shape != target.shape:
        warped = cv2.resize(warped, (target.shape[1], target.shape[0]))
    
    # Calculate overall SSIM
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    
    overall_ssim = ssim(target_gray, warped_gray)
    print(f"  Overall SSIM: {overall_ssim:.4f}")
    
    # Patch-based evaluation
    patch_size = 256
    h, w = target.shape[:2]
    
    if h > patch_size and w > patch_size:
        ssim_scores = []
        
        for i in range(num_patches):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            
            target_patch = target_gray[y:y+patch_size, x:x+patch_size]
            warped_patch = warped_gray[y:y+patch_size, x:x+patch_size]
            
            patch_ssim = ssim(target_patch, warped_patch)
            ssim_scores.append(patch_ssim)
        
        print(f"\n  Patch-based statistics ({num_patches} patches):")
        print(f"    Mean SSIM: {np.mean(ssim_scores):.4f}")
        print(f"    Std SSIM: {np.std(ssim_scores):.4f}")
        print(f"    Min SSIM: {np.min(ssim_scores):.4f}")
        print(f"    Max SSIM: {np.max(ssim_scores):.4f}")
    
    # Create visualization
    print("\n  Creating visualizations...")
    
    # Checkerboard overlay
    checker_size = 100
    checkerboard = np.zeros_like(target)
    
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = warped[i:i+checker_size, j:j+checker_size]
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = target[i:i+checker_size, j:j+checker_size]
    
    # Save visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Side by side comparison
    comparison = np.hstack([target, warped])
    cv2.imwrite(str(viz_dir / "side_by_side.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Checkerboard
    cv2.imwrite(str(viz_dir / "checkerboard.jpg"), cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # Overlay
    overlay = cv2.addWeighted(target, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(viz_dir / "overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"  ✅ Visualizations saved to: {viz_dir}")

#############################################################################
# MAIN PIPELINE
#############################################################################

def main():
    """Main execution pipeline"""
    print_header("H&E TO IF QPTIFF REGISTRATION PIPELINE")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Register H&E to IF QPTIFF images")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--resolution-level", type=int, default=1, help="Resolution level (0=full, 1=10x, etc)")
    parser.add_argument("--if-channels", type=int, nargs='+', default=[0, 1, 5], help="IF channels to use")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU check")
    parser.add_argument("--use-openslide", action="store_true", help="Use OpenSlide for loading if available")
    
    args = parser.parse_args()
    
    # Setup paths
    he_qptiff_path = Path(args.he_qptiff)
    if_qptiff_path = Path(args.if_qptiff)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not he_qptiff_path.exists():
        print(f"❌ H&E QPTIFF not found: {he_qptiff_path}")
        return 1
        
    if not if_qptiff_path.exists():
        print(f"❌ IF QPTIFF not found: {if_qptiff_path}")
        return 1
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Check GPU
    if not args.skip_gpu_check:
        if not check_gpu():
            response = input("\nContinue without GPU? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    # Initialize loader
    loader = QPTIFFLoader(resolution_level=args.resolution_level, 
                         use_openslide=args.use_openslide)
    
    try:
        # Load H&E QPTIFF
        he_array, he_temp_path = loader.load_he_qptiff(he_qptiff_path, temp_dir)
        
        # Load IF QPTIFF
        if_array, if_temp_path = loader.load_if_qptiff(
            if_qptiff_path, temp_dir, channels=args.if_channels
        )
        
        # Perform registration
        registration_results = perform_registration_pipeline(
            he_temp_path, 
            if_temp_path, 
            output_dir,
            he_original_path=he_qptiff_path  # For applying to original
        )
        
        if registration_results['success']:
            # Evaluate results
            evaluate_registration(
                if_temp_path,
                registration_results['warped_path'],
                output_dir
            )
            
            # Summary
            print_section("REGISTRATION SUMMARY")
            print(f"✅ Registration completed successfully!")
            print(f"   Time: {registration_results['elapsed_time']:.1f} seconds")
            print(f"   Output: {output_dir}")
            print(f"\nKey outputs:")
            print(f"   - Warped H&E: {registration_results['warped_path'].name}")
            if 'warped_original_path' in registration_results:
                print(f"   - Warped H&E (original res): {registration_results['warped_original_path'].name}")
            print(f"   - Visualizations: {output_dir / 'visualizations'}")
            
        else:
            print(f"\n❌ Registration failed: {registration_results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup temp files if desired
        print("\n" + "="*70)
        response = input("Clean up temporary files? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up")
    
    print(f"\n✅ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
