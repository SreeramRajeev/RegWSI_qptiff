#!/usr/bin/env python3
"""
Enhanced Adaptive QPTIFF Registration Pipeline
==============================================
Automatically adapts preprocessing and registration parameters
based on image characteristics for optimal alignment
"""

### System Imports ###
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

### Scientific Computing ###
import numpy as np
import torch as tc
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.stats import entropy
from skimage import exposure, filters, morphology
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import skimage.transform as skt

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
# IMAGE ANALYSIS FUNCTIONS
#############################################################################

def analyze_image_characteristics(image):
    """
    Analyze image characteristics to determine optimal preprocessing
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Calculate various metrics
    metrics = {}
    
    # Brightness and contrast
    metrics['mean_intensity'] = np.mean(gray)
    metrics['std_intensity'] = np.std(gray)
    metrics['contrast'] = metrics['std_intensity'] / (metrics['mean_intensity'] + 1e-6)
    
    # Histogram analysis
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist_norm = hist / hist.sum()
    metrics['entropy'] = entropy(hist_norm)
    
    # Edge content
    edges = cv2.Canny(gray, 50, 150)
    metrics['edge_density'] = np.mean(edges > 0)
    
    # Texture analysis
    metrics['gradient_magnitude'] = np.mean(filters.sobel(gray))
    
    # Tissue content (non-background pixels)
    threshold = filters.threshold_otsu(gray)
    tissue_mask = gray > threshold
    metrics['tissue_fraction'] = np.mean(tissue_mask)
    
    # Color distribution for RGB images
    if image.ndim == 3:
        metrics['color_variance'] = np.mean([np.std(image[:,:,i]) for i in range(3)])
        # Check for stain separation quality
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        metrics['lab_variance'] = np.std(lab[:,:,1:])  # a* and b* channels
    
    return metrics

def determine_preprocessing_strategy(source_metrics, target_metrics):
    """
    Determine optimal preprocessing based on image metrics
    """
    strategy = {
        'clahe_clip_limit': 2.0,
        'clahe_grid_size': 8,
        'blur_kernel': 3,
        'blur_sigma': 1.0,
        'edge_weight': 0.3,
        'use_color_normalization': False,
        'use_histogram_matching': False,
        'enhancement_strength': 'medium'
    }
    
    # Adjust based on contrast
    avg_contrast = (source_metrics['contrast'] + target_metrics['contrast']) / 2
    if avg_contrast < 0.2:
        # Low contrast - need stronger enhancement
        strategy['clahe_clip_limit'] = 4.0
        strategy['enhancement_strength'] = 'strong'
    elif avg_contrast > 0.5:
        # High contrast - gentler enhancement
        strategy['clahe_clip_limit'] = 1.5
        strategy['enhancement_strength'] = 'light'
    
    # Adjust based on tissue content
    avg_tissue = (source_metrics['tissue_fraction'] + target_metrics['tissue_fraction']) / 2
    if avg_tissue < 0.3:
        # Sparse tissue - need careful handling
        strategy['blur_kernel'] = 5
        strategy['edge_weight'] = 0.5
    
    # Check if images have very different characteristics
    intensity_diff = abs(source_metrics['mean_intensity'] - target_metrics['mean_intensity'])
    if intensity_diff > 50:
        strategy['use_histogram_matching'] = True
    
    # For images with good color information
    if 'lab_variance' in source_metrics and source_metrics['lab_variance'] > 20:
        strategy['use_color_normalization'] = True
    
    # Adjust grid size based on image entropy
    avg_entropy = (source_metrics['entropy'] + target_metrics['entropy']) / 2
    if avg_entropy > 6.5:
        strategy['clahe_grid_size'] = 16  # Finer grid for complex images
    elif avg_entropy < 5.0:
        strategy['clahe_grid_size'] = 4   # Coarser grid for simple images
    
    return strategy

#############################################################################
# ADAPTIVE PREPROCESSING
#############################################################################

def adaptive_preprocess(source, target, strategy):
    """
    Apply adaptive preprocessing based on determined strategy
    """
    print(f"  Applying {strategy['enhancement_strength']} enhancement...")
    
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
    
    # Apply histogram matching if needed
    if strategy['use_histogram_matching']:
        print("  Applying histogram matching...")
        source = exposure.match_histograms(source, target, channel_axis=-1)
    
    # Apply color normalization if beneficial
    if strategy['use_color_normalization']:
        print("  Applying color normalization...")
        source = normalize_staining(source)
        target = normalize_staining(target)
    
    # Adaptive noise reduction
    if strategy['blur_kernel'] > 0:
        source = cv2.GaussianBlur(source, 
                                 (strategy['blur_kernel'], strategy['blur_kernel']), 
                                 strategy['blur_sigma'])
        target = cv2.GaussianBlur(target, 
                                 (strategy['blur_kernel'], strategy['blur_kernel']), 
                                 strategy['blur_sigma'])
    
    # Adaptive CLAHE
    source_enhanced = np.zeros_like(source)
    target_enhanced = np.zeros_like(target)
    
    clahe = cv2.createCLAHE(clipLimit=strategy['clahe_clip_limit'], 
                           tileGridSize=(strategy['clahe_grid_size'], strategy['clahe_grid_size']))
    
    for channel in range(3):
        source_enhanced[:, :, channel] = clahe.apply(source[:, :, channel])
        target_enhanced[:, :, channel] = clahe.apply(target[:, :, channel])
    
    # Adaptive edge enhancement
    if strategy['edge_weight'] > 0:
        source_edges = compute_multi_scale_edges(source_enhanced)
        target_edges = compute_multi_scale_edges(target_enhanced)
        
        # Combine with original
        source_final = np.zeros_like(source_enhanced)
        target_final = np.zeros_like(target_enhanced)
        
        for channel in range(3):
            source_final[:, :, channel] = cv2.addWeighted(
                source_enhanced[:, :, channel], 1 - strategy['edge_weight'],
                source_edges, strategy['edge_weight'], 0
            )
            target_final[:, :, channel] = cv2.addWeighted(
                target_enhanced[:, :, channel], 1 - strategy['edge_weight'],
                target_edges, strategy['edge_weight'], 0
            )
    else:
        source_final = source_enhanced
        target_final = target_enhanced
    
    return source_final, target_final

def normalize_staining(image):
    """
    Normalize staining using Reinhard color normalization
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Compute mean and std for each channel
    l_mean, l_std = lab[:,:,0].mean(), lab[:,:,0].std()
    a_mean, a_std = lab[:,:,1].mean(), lab[:,:,1].std()
    b_mean, b_std = lab[:,:,2].mean(), lab[:,:,2].std()
    
    # Target statistics (typical H&E values)
    target_l_mean, target_l_std = 140.0, 35.0
    target_a_mean, target_a_std = 128.0, 15.0
    target_b_mean, target_b_std = 128.0, 15.0
    
    # Normalize each channel
    lab[:,:,0] = (lab[:,:,0] - l_mean) * (target_l_std / l_std) + target_l_mean
    lab[:,:,1] = (lab[:,:,1] - a_mean) * (target_a_std / a_std) + target_a_mean
    lab[:,:,2] = (lab[:,:,2] - b_mean) * (target_b_std / b_std) + target_b_mean
    
    # Clip values
    lab[:,:,0] = np.clip(lab[:,:,0], 0, 255)
    lab[:,:,1] = np.clip(lab[:,:,1], 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
    
    # Convert back to RGB
    normalized = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return normalized

def compute_multi_scale_edges(image):
    """
    Compute edges at multiple scales for robustness
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Multi-scale edge detection
    scales = [1, 2, 4]
    edges = np.zeros_like(gray, dtype=np.float32)
    
    for scale in scales:
        # Gaussian blur for scale
        blurred = gaussian_filter(gray, sigma=scale)
        
        # Compute gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Accumulate (with scale normalization)
        edges += magnitude / scale
    
    # Normalize
    edges = edges / len(scales)
    edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
    
    return edges

#############################################################################
# ADAPTIVE REGISTRATION PARAMETERS
#############################################################################

def create_adaptive_registration_params(image_metrics, preprocessing_strategy, attempt=1):
    """
    Create registration parameters adapted to image characteristics
    """
    params = default_initial_nonrigid()
    
    # Determine registration strategy based on metrics
    if image_metrics['edge_density'] > 0.1:
        # Good edge content - feature-based should work well
        alignment_type = 'feature_based'
        detector = 'superpoint' if attempt == 1 else 'sift'
    else:
        # Poor edges - try intensity-based or different feature detector
        if attempt == 1:
            alignment_type = 'feature_based'
            detector = 'orb'  # More general features
        else:
            alignment_type = 'intensity_based'
            detector = None
    
    if alignment_type == 'feature_based':
        params['initial_alignment_params'] = {
            'type': 'feature_based',
            'detector': detector,
            'matcher': 'superglue' if detector == 'superpoint' else 'flann',
            'ransac_threshold': 5.0 if image_metrics['tissue_fraction'] > 0.5 else 10.0,
            'max_features': 20000 if image_metrics['entropy'] > 6 else 10000,
            'match_ratio': 0.7 if attempt == 1 else 0.8,
            'use_mutual_best': True if attempt > 1 else False,
            'nms_radius': 4,
            'keypoint_threshold': 0.001 if image_metrics['edge_density'] < 0.05 else 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        }
    else:
        params['initial_alignment_params'] = {
            'type': 'intensity_based',
            'metric': 'mattes_mutual_information',
            'optimizer': 'regular_step_gradient_descent',
            'iterations': 1000,
            'learning_rate': 1.0,
            'min_step': 0.0001,
            'relaxation_factor': 0.5,
        }
    
    # Adaptive nonrigid parameters
    if preprocessing_strategy['enhancement_strength'] == 'strong':
        # Difficult case - more iterations, stronger regularization
        iterations = [300, 200, 150, 100, 50]
        smoothing_sigma = 4.0
        update_field_sigma = 3.0
    elif preprocessing_strategy['enhancement_strength'] == 'light':
        # Easy case - fewer iterations
        iterations = [150, 100, 50, 25]
        smoothing_sigma = 2.0
        update_field_sigma = 1.5
    else:
        # Medium case
        iterations = [200, 150, 100, 50]
        smoothing_sigma = 3.0
        update_field_sigma = 2.0
    
    params['nonrigid_params'] = {
        'type': 'demons' if attempt <= 2 else 'bspline',
        'iterations': iterations,
        'smoothing_sigma': smoothing_sigma,
        'update_field_sigma': update_field_sigma,
        'max_step_length': 2.0 if image_metrics['tissue_fraction'] > 0.7 else 5.0,
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric' if attempt == 1 else 'fixed',
    }
    
    # Multi-resolution based on image size and content
    if image_metrics['entropy'] > 6.5:
        # Complex image - more levels
        params['multiresolution_params'] = {
            'levels': 6,
            'shrink_factors': [32, 16, 8, 4, 2, 1],
            'smoothing_sigmas': [16.0, 8.0, 4.0, 2.0, 1.0, 0.5],
        }
    else:
        params['multiresolution_params'] = {
            'levels': 5,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [8.0, 4.0, 2.0, 1.0, 0.5],
        }
    
    # Optimization parameters
    metric = 'mattes_mutual_information' if image_metrics['entropy'] > 5 else 'correlation'
    
    params['optimization_params'] = {
        'metric': metric,
        'number_of_bins': 64 if image_metrics['entropy'] > 6 else 32,
        'optimizer': 'gradient_descent',
        'learning_rate': 1.0 if attempt == 1 else 2.0,
        'min_step': 0.0001,
        'iterations': 1000 if attempt > 1 else 500,
        'relaxation_factor': 0.7,
        'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.2 if image_metrics['tissue_fraction'] > 0.5 else 0.1,
    }
    
    # Common parameters
    params['loading_params']['loader'] = 'tiff'
    params['loading_params']['downsample_factor'] = 1
    params['save_displacement_field'] = True
    
    return params

#############################################################################
# QUALITY ASSESSMENT
#############################################################################

def assess_registration_quality(source, target, warped, displacement_field=None):
    """
    Comprehensive registration quality assessment
    """
    metrics = {}
    
    # Ensure same size
    h, w = target.shape[:2]
    if source.shape[:2] != (h, w):
        source = cv2.resize(source, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # Convert to grayscale for metrics
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    else:
        source_gray = source
        target_gray = target
        warped_gray = warped
    
    # SSIM
    metrics['ssim'] = ssim(target_gray, warped_gray)
    metrics['ssim_improvement'] = metrics['ssim'] - ssim(target_gray, source_gray)
    
    # Normalized Cross Correlation
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    warped_norm = (warped_gray - np.mean(warped_gray)) / (np.std(warped_gray) + 1e-6)
    metrics['ncc'] = np.mean(target_norm * warped_norm)
    
    # Mutual Information
    metrics['mi'] = mutual_information(target_gray, warped_gray)
    
    # Edge alignment
    target_edges = cv2.Canny(target_gray, 50, 150)
    warped_edges = cv2.Canny(warped_gray, 50, 150)
    metrics['edge_overlap'] = dice_coefficient(target_edges > 0, warped_edges > 0)
    
    # Displacement field smoothness (if available)
    if displacement_field is not None:
        dx = displacement_field[:, :, 0]
        dy = displacement_field[:, :, 1]
        
        # Gradient of displacement field
        dx_grad = np.gradient(dx)
        dy_grad = np.gradient(dy)
        
        # Smoothness metric (lower is smoother)
        metrics['displacement_smoothness'] = np.mean(np.abs(dx_grad[0]) + np.abs(dx_grad[1]) + 
                                                    np.abs(dy_grad[0]) + np.abs(dy_grad[1]))
        
        # Maximum displacement
        metrics['max_displacement'] = np.max(np.sqrt(dx**2 + dy**2))
    
    # Overall quality score (weighted combination)
    metrics['quality_score'] = (
        0.4 * metrics['ssim'] + 
        0.3 * metrics['ncc'] + 
        0.2 * metrics['edge_overlap'] + 
        0.1 * (metrics['mi'] / 2.0)  # Normalized MI
    )
    
    return metrics

def mutual_information(im1, im2, bins=256):
    """Calculate mutual information between two images"""
    hist_2d, _, _ = np.histogram2d(im1.ravel(), im2.ravel(), bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def dice_coefficient(mask1, mask2):
    """Calculate Dice coefficient between two binary masks"""
    intersection = np.sum(mask1 & mask2)
    return 2.0 * intersection / (np.sum(mask1) + np.sum(mask2) + 1e-6)

#############################################################################
# FALLBACK REGISTRATION METHODS
#############################################################################

def try_opencv_registration(source, target):
    """
    Fallback registration using OpenCV methods
    """
    print("   Trying OpenCV feature-based registration...")
    
    # Convert to grayscale
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    else:
        source_gray = source
        target_gray = target
    
    # Try multiple feature detectors
    detectors = [
        ('SIFT', cv2.SIFT_create(nfeatures=10000)),
        ('ORB', cv2.ORB_create(nfeatures=10000)),
        ('AKAZE', cv2.AKAZE_create())
    ]
    
    best_transform = None
    best_score = -1
    
    for name, detector in detectors:
        try:
            # Detect features
            kp1, des1 = detector.detectAndCompute(source_gray, None)
            kp2, des2 = detector.detectAndCompute(target_gray, None)
            
            if len(kp1) < 10 or len(kp2) < 10:
                continue
            
            # Match features
            if name == 'ORB':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
            else:
                matcher = cv2.FlannBasedMatcher()
                matches = matcher.knnMatch(des1, des2, k=2)
                # Lowe's ratio test
                matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
            
            if len(matches) < 10:
                continue
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Warp and evaluate
                h, w = target.shape[:2]
                warped = cv2.warpPerspective(source, M, (w, h))
                
                # Quick quality check
                if warped.ndim == 3:
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
                else:
                    warped_gray = warped
                
                score = ssim(target_gray, warped_gray)
                
                if score > best_score:
                    best_score = score
                    best_transform = M
                    
                print(f"     {name}: {len(matches)} matches, SSIM: {score:.3f}")
                
        except Exception as e:
            print(f"     {name} failed: {e}")
            continue
    
    return best_transform, best_score

def try_phase_correlation(source, target):
    """
    Try phase correlation for initial alignment
    """
    print("   Trying phase correlation alignment...")
    
    # Convert to grayscale float
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        source_gray = source.astype(np.float32)
        target_gray = target.astype(np.float32)
    
    # Ensure same size
    if source_gray.shape != target_gray.shape:
        source_gray = cv2.resize(source_gray, (target_gray.shape[1], target_gray.shape[0]))
    
    try:
        # Phase correlation
        shift, error = cv2.phaseCorrelate(source_gray, target_gray)
        
        # Create translation matrix
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        
        print(f"     Translation: ({shift[0]:.1f}, {shift[1]:.1f}), Error: {error:.3f}")
        
        return M, 1.0 - error
        
    except Exception as e:
        print(f"     Phase correlation failed: {e}")
        return None, -1

#############################################################################
# ENHANCED REGISTRATION PIPELINE
#############################################################################

def register_qptiff_adaptive(he_qptiff_path: Path, if_qptiff_path: Path, 
                           output_dir: Path, if_channels: list = None) -> Dict:
    """
    Enhanced adaptive QPTIFF registration for optimal alignment
    """
    print("\n" + "="*70)
    print(" ADAPTIVE QPTIFF REGISTRATION PIPELINE")
    print("="*70)
    
    if if_channels is None:
        if_channels = DEFAULT_IF_CHANNELS
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "registration_adaptive"
    reg_dir.mkdir(exist_ok=True)
    temp_dir = reg_dir / "TEMP"
    temp_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Step 1: Load and prepare images
        print("\n1. Loading and preparing images...")
        
        # Load IF image
        with tifffile.TiffFile(if_qptiff_path) as tif:
            if_data = tif.asarray()
            print(f"   IF shape: {if_data.shape}")
            
            # Handle different formats
            if if_data.ndim == 4:
                if_data = if_data[0] if if_data.shape[0] < if_data.shape[1] else if_data[:, 0, :, :]
            
            # Extract channels and create RGB
            if if_data.ndim == 3 and if_data.shape[0] <= 16:
                selected = []
                for ch_idx in if_channels[:3]:
                    if ch_idx < if_data.shape[0]:
                        ch = if_data[ch_idx]
                        # Robust percentile normalization
                        p1, p99 = np.percentile(ch[ch > 0], [1, 99]) if np.any(ch > 0) else (0, 1)
                        ch_norm = np.clip((ch - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)
                        selected.append(ch_norm)
                
                # Ensure 3 channels
                while len(selected) < 3:
                    selected.append(np.zeros_like(selected[0]))
                
                if_rgb = np.stack(selected[:3], axis=-1)
            else:
                # Already RGB or grayscale
                if_rgb = if_data.astype(np.uint8)
                if if_rgb.ndim == 2:
                    if_rgb = cv2.cvtColor(if_rgb, cv2.COLOR_GRAY2RGB)
            
            target_shape = if_rgb.shape
        
        # Load H&E image
        with tifffile.TiffFile(he_qptiff_path) as tif:
            he_data = tif.pages[0].asarray()
            print(f"   H&E shape: {he_data.shape}")
            
            # Handle different formats
            if he_data.ndim == 3 and he_data.shape[0] == 3:
                he_data = np.transpose(he_data, (1, 2, 0))
            
            if he_data.ndim == 2:
                he_data = cv2.cvtColor(he_data, cv2.COLOR_GRAY2RGB)
            
            # Convert to uint8
            if he_data.dtype != np.uint8:
                if he_data.dtype == np.uint16:
                    he_data = (he_data / 256).astype(np.uint8)
                else:
                    p1, p99 = np.percentile(he_data, [1, 99])
                    he_data = np.clip((he_data - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            
            # Store original
            he_original = he_data.copy()
            
            # Resize to match IF
            if he_data.shape[:2] != target_shape[:2]:
                print(f"   Resizing H&E to match IF...")
                he_data = cv2.resize(he_data, (target_shape[1], target_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                he_original = cv2.resize(he_original, (target_shape[1], target_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Analyze image characteristics
        print("\n2. Analyzing image characteristics...")
        he_metrics = analyze_image_characteristics(he_data)
        if_metrics = analyze_image_characteristics(if_rgb)
        
        print(f"   H&E - Contrast: {he_metrics['contrast']:.2f}, "
              f"Entropy: {he_metrics['entropy']:.2f}, "
              f"Edge density: {he_metrics['edge_density']:.3f}")
        print(f"   IF  - Contrast: {if_metrics['contrast']:.2f}, "
              f"Entropy: {if_metrics['entropy']:.2f}, "
              f"Edge density: {if_metrics['edge_density']:.3f}")
        
        # Step 3: Determine preprocessing strategy
        print("\n3. Determining optimal preprocessing strategy...")
        avg_metrics = {k: (he_metrics[k] + if_metrics.get(k, 0)) / 2 
                      for k in he_metrics if k in if_metrics}
        preprocessing_strategy = determine_preprocessing_strategy(he_metrics, if_metrics)
        
        # Step 4: Registration attempts with different strategies
        print("\n4. Starting adaptive registration...")
        
        best_result = None
        best_quality = -1
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            
            # Adaptive preprocessing
            he_prep, if_prep = adaptive_preprocess(he_data, if_rgb, preprocessing_strategy)
            
            # Save preprocessed versions
            he_prep_path = temp_dir / f"he_preprocessed_attempt{attempt}.tiff"
            if_prep_path = temp_dir / f"if_preprocessed_attempt{attempt}.tiff"
            tifffile.imwrite(he_prep_path, he_prep, photometric='rgb', compression='lzw', bigtiff=True)
            tifffile.imwrite(if_prep_path, if_prep, photometric='rgb', compression='lzw', bigtiff=True)
            
            # Create adaptive registration parameters
            reg_params = create_adaptive_registration_params(avg_metrics, 
                                                           preprocessing_strategy, 
                                                           attempt)
            
            # Try DeepHistReg
            try:
                case_name = f'adaptive_reg_attempt{attempt}'
                
                config = {
                    'source_path': str(he_prep_path),
                    'target_path': str(if_prep_path),
                    'output_path': str(temp_dir),
                    'registration_parameters': reg_params,
                    'case_name': case_name,
                    'save_displacement_field': True,
                    'copy_target': True,
                    'delete_temporary_results': False,
                    'temporary_path': str(temp_dir / case_name)
                }
                
                print(f"   Running DeepHistReg with {reg_params['initial_alignment_params']['type']} alignment...")
                start_time = time.time()
                deeperhistreg.run_registration(**config)
                elapsed = time.time() - start_time
                print(f"   Registration completed in {elapsed:.1f} seconds")
                
                # Find and apply displacement field
                disp_field_path = find_displacement_field(temp_dir, case_name)
                
                if disp_field_path:
                    displacement_field = load_displacement_field_robust(disp_field_path)
                    warped = apply_displacement_field_safe(he_original, displacement_field)
                    
                    # Assess quality
                    quality_metrics = assess_registration_quality(he_original, if_rgb, 
                                                                warped, displacement_field)
                    
                    print(f"   Quality - SSIM: {quality_metrics['ssim']:.3f}, "
                          f"NCC: {quality_metrics['ncc']:.3f}, "
                          f"Score: {quality_metrics['quality_score']:.3f}")
                    
                    if quality_metrics['quality_score'] > best_quality:
                        best_quality = quality_metrics['quality_score']
                        best_result = {
                            'warped': warped,
                            'displacement_field': displacement_field,
                            'quality_metrics': quality_metrics,
                            'attempt': attempt,
                            'elapsed_time': elapsed
                        }
                    
                    # Good enough?
                    if quality_metrics['quality_score'] > 0.8:
                        print("   ✓ Excellent alignment achieved!")
                        break
                
            except Exception as e:
                print(f"   DeepHistReg failed: {e}")
            
            # Try fallback methods if DeepHistReg failed or quality is poor
            if best_quality < 0.5:
                print("   Trying fallback registration methods...")
                
                # Try OpenCV registration
                transform, score = try_opencv_registration(he_prep, if_prep)
                if transform is not None and score > best_quality:
                    h, w = if_rgb.shape[:2]
                    warped = cv2.warpPerspective(he_original, transform, (w, h))
                    
                    quality_metrics = assess_registration_quality(he_original, if_rgb, warped)
                    if quality_metrics['quality_score'] > best_quality:
                        best_quality = quality_metrics['quality_score']
                        best_result = {
                            'warped': warped,
                            'transform': transform,
                            'quality_metrics': quality_metrics,
                            'method': 'opencv',
                            'attempt': attempt
                        }
                
                # Try phase correlation
                transform, score = try_phase_correlation(he_prep, if_prep)
                if transform is not None:
                    h, w = if_rgb.shape[:2]
                    warped = cv2.warpAffine(he_original, transform, (w, h))
                    
                    quality_metrics = assess_registration_quality(he_original, if_rgb, warped)
                    if quality_metrics['quality_score'] > best_quality * 0.8:  # Accept if reasonably close
                        best_quality = quality_metrics['quality_score']
                        best_result = {
                            'warped': warped,
                            'transform': transform,
                            'quality_metrics': quality_metrics,
                            'method': 'phase_correlation',
                            'attempt': attempt
                        }
            
            # Adjust strategy for next attempt
            if attempt < max_attempts and best_quality < 0.7:
                print("   Adjusting strategy for next attempt...")
                if preprocessing_strategy['enhancement_strength'] == 'medium':
                    preprocessing_strategy['enhancement_strength'] = 'strong'
                    preprocessing_strategy['clahe_clip_limit'] += 1.0
                preprocessing_strategy['use_histogram_matching'] = True
                preprocessing_strategy['edge_weight'] = min(0.7, preprocessing_strategy['edge_weight'] + 0.2)
        
        # Step 5: Save best result
        if best_result is not None:
            print(f"\n5. Saving best registration result (attempt {best_result.get('attempt', 'N/A')})...")
            
            final_output_path = output_dir / "registered_HE_adaptive.tiff"
            tifffile.imwrite(
                final_output_path,
                best_result['warped'],
                photometric='rgb',
                compression='lzw',
                bigtiff=True
            )
            
            # Save quality report
            quality_report = output_dir / "registration_quality_report.txt"
            with open(quality_report, 'w') as f:
                f.write("Registration Quality Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Best attempt: {best_result.get('attempt', 'N/A')}\n")
                f.write(f"Method: {best_result.get('method', 'deephistreg')}\n")
                f.write(f"Time: {best_result.get('elapsed_time', 0):.1f} seconds\n\n")
                f.write("Quality Metrics:\n")
                for key, value in best_result['quality_metrics'].items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            results['success'] = True
            results['registered_path'] = final_output_path
            results['quality_metrics'] = best_result['quality_metrics']
            results['best_attempt'] = best_result.get('attempt', 'N/A')
            
            # Create visualizations
            print("\n6. Creating quality visualizations...")
            create_enhanced_visualizations(he_original, if_rgb, best_result['warped'], 
                                         output_dir, best_result['quality_metrics'])
            
        else:
            print("\n✗ Registration failed - no acceptable alignment found")
            results['success'] = False
            results['error'] = "No acceptable registration achieved"
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results

#############################################################################
# UTILITIES
#############################################################################

def save_tiff_safe(filepath, image, **kwargs):
    """Save TIFF with automatic BigTIFF detection"""
    size_estimate = image.nbytes
    if size_estimate > 3 * 1024**3:
        kwargs['bigtiff'] = True
    tifffile.imwrite(filepath, image, **kwargs)

def find_displacement_field(temp_dir, case_name):
    """Find displacement field in output directory"""
    search_patterns = [
        f'{case_name}_displacement_field.*',
        'displacement_field.*',
        '*displacement*.*',
    ]
    
    search_dirs = [
        temp_dir / case_name,
        temp_dir / case_name / 'Results_Final',
        temp_dir,
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for pattern in search_patterns:
                matches = list(search_dir.glob(pattern))
                for match in matches:
                    if match.is_file() and match.stat().st_size > 1000:
                        return match
    
    return None

def load_displacement_field_robust(filepath):
    """Load displacement field from various formats"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        return np.load(str(filepath))
    
    elif filepath.suffix == '.mha':
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK required for .mha files")
        
        displacement_image = sitk.ReadImage(str(filepath))
        displacement_array = sitk.GetArrayFromImage(displacement_image)
        
        if displacement_array.ndim == 4:
            displacement_array = displacement_array[0]
        
        if displacement_array.shape[-1] == 2:
            return displacement_array
        elif displacement_array.shape[0] == 2:
            return displacement_array.transpose(1, 2, 0)
    
    raise ValueError(f"Unsupported format: {filepath.suffix}")

def apply_displacement_field_safe(image, displacement_field):
    """Apply displacement field with size checking"""
    h, w = image.shape[:2]
    
    # Resize displacement field if needed
    if displacement_field.shape[:2] != (h, w):
        displacement_field = cv2.resize(displacement_field, (w, h), 
                                      interpolation=cv2.INTER_LINEAR)
    
    # Check OpenCV size limits
    max_size = 32767
    if h <= max_size and w <= max_size:
        # Direct application
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + displacement_field[:, :, 0]).astype(np.float32)
        map_y = (y + displacement_field[:, :, 1]).astype(np.float32)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT)
    else:
        # Use scipy for large images
        warped = np.zeros_like(image)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        for c in range(image.shape[2] if image.ndim == 3 else 1):
            channel = image[:, :, c] if image.ndim == 3 else image
            sample_x = np.clip(x_coords + displacement_field[:, :, 0], 0, w - 1)
            sample_y = np.clip(y_coords + displacement_field[:, :, 1], 0, h - 1)
            
            if image.ndim == 3:
                warped[:, :, c] = map_coordinates(channel, [sample_y, sample_x], 
                                                order=1, mode='reflect')
            else:
                warped = map_coordinates(channel, [sample_y, sample_x], 
                                       order=1, mode='reflect')
        
        return warped.astype(image.dtype)

def create_enhanced_visualizations(original, target, warped, output_dir, metrics):
    """Create comprehensive visualization outputs"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Ensure same size
    h, w = target.shape[:2]
    if original.shape[:2] != (h, w):
        original = cv2.resize(original, (w, h))
    if warped.shape[:2] != (h, w):
        warped = cv2.resize(warped, (w, h))
    
    # Create different visualizations
    
    # 1. Side-by-side comparison
    comparison = np.hstack([original, target, warped])
    cv2.imwrite(str(viz_dir / "comparison.jpg"), 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 2. Animated checkerboard
    checker_sizes = [50, 100, 200]
    for size in checker_sizes:
        checkerboard = create_checkerboard(target, warped, size)
        cv2.imwrite(str(viz_dir / f"checkerboard_{size}.jpg"), 
                   cv2.cvtColor(checkerboard, cv2.COLOR_RGB2BGR))
    
    # 3. Difference images
    diff_before = cv2.absdiff(target, original)
    diff_after = cv2.absdiff(target, warped)
    diff_comparison = np.hstack([diff_before, diff_after])
    cv2.imwrite(str(viz_dir / "difference_comparison.jpg"), 
                cv2.cvtColor(diff_comparison, cv2.COLOR_RGB2BGR))
    
    # 4. Overlay with different blend modes
    overlays = {
        'blend_50': cv2.addWeighted(target, 0.5, warped, 0.5, 0),
        'blend_target_dominant': cv2.addWeighted(target, 0.7, warped, 0.3, 0),
        'blend_warped_dominant': cv2.addWeighted(target, 0.3, warped, 0.7, 0),
    }
    
    for name, overlay in overlays.items():
        cv2.imwrite(str(viz_dir / f"{name}.jpg"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 5. Quality metrics visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot images
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original H&E')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target)
    axes[0, 1].set_title('Target IF')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(warped)
    axes[1, 0].set_title('Registered H&E')
    axes[1, 0].axis('off')
    
    # Plot metrics
    axes[1, 1].text(0.1, 0.9, f"Quality Score: {metrics['quality_score']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.7, f"SSIM: {metrics['ssim']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f"NCC: {metrics['ncc']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"Edge Overlap: {metrics['edge_overlap']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.1, f"MI: {metrics['mi']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(viz_dir / "quality_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Visualizations saved to: {viz_dir}")

def create_checkerboard(img1, img2, square_size):
    """Create checkerboard pattern mixing two images"""
    h, w = img1.shape[:2]
    checkerboard = np.zeros_like(img1)
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if (i//square_size + j//square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = img1[i:i+square_size, j:j+square_size]
            else:
                checkerboard[i:i+square_size, j:j+square_size] = img2[i:i+square_size, j:j+square_size]
    
    return checkerboard

#############################################################################
# MAIN EXECUTION
#############################################################################

def main():
    """Main execution with enhanced pipeline"""
    import argparse
    
    print("\n" + "="*70)
    print(" ENHANCED ADAPTIVE QPTIFF REGISTRATION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Adaptive QPTIFF registration for optimal alignment")
    parser.add_argument("--he-qptiff", type=str, required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if-qptiff", type=str, required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output-dir", type=str, default="./output_adaptive", help="Output directory")
    parser.add_argument("--if-channels", type=int, nargs='+', default=[0, 1, 5], 
                       help="IF channels to use (default: 0=DAPI, 1=CD8, 5=CD163)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    he_path = Path(args.he_qptiff)
    if_path = Path(args.if_qptiff)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not he_path.exists():
        print(f"✗ H&E file not found: {he_path}")
        return 1
    
    if not if_path.exists():
        print(f"✗ IF file not found: {if_path}")
        return 1
    
    print(f"\nInput files:")
    print(f"  H&E: {he_path.name} ({he_path.stat().st_size/1e9:.2f} GB)")
    print(f"  IF:  {if_path.name} ({if_path.stat().st_size/1e9:.2f} GB)")
    print(f"  Output: {output_dir}")
    
    # Check GPU
    if tc.cuda.is_available():
        print(f"\n✓ GPU available: {tc.cuda.get_device_name(0)}")
        # Set memory growth
        tc.cuda.empty_cache()
    else:
        print("\n⚠  No GPU detected - registration will be slower")
    
    # Run adaptive registration
    results = register_qptiff_adaptive(he_path, if_path, output_dir, args.if_channels)
    
    # Summary
    print("\n" + "="*70)
    print(" REGISTRATION SUMMARY")
    print("="*70)
    
    if results.get('success'):
        print(f"✓ Registration completed successfully!")
        print(f"   Best attempt: {results.get('best_attempt', 'N/A')}")
        print(f"   Output: {results['registered_path']}")
        
        quality = results.get('quality_metrics', {})
        print(f"\nQuality Metrics:")
        print(f"   Overall Score: {quality.get('quality_score', 0):.3f}")
        print(f"   SSIM: {quality.get('ssim', 0):.3f}")
        print(f"   NCC: {quality.get('ncc', 0):.3f}")
        print(f"   Edge Overlap: {quality.get('edge_overlap', 0):.3f}")
        
        if quality.get('quality_score', 0) > 0.8:
            print("\n✓ Excellent alignment achieved!")
        elif quality.get('quality_score', 0) > 0.6:
            print("\n✓ Good alignment achieved")
        else:
            print("\n⚠  Moderate alignment - manual review recommended")
    else:
        print(f"✗ Registration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
