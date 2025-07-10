#!/usr/bin/env python3
"""
WSI Registration Script using DeeperHistReg
Registers two whole slide images (WSI) in qptiff format.

Usage:
    python registration.py \
      --he-qptiff /path/to/he.qptiff \
      --if-qptiff /path/to/if.qptiff \
      --output-dir ./output
"""

import argparse
import pathlib
import sys
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt

### DeeperHistReg Imports ###
import deeperhistreg
from deeperhistreg.dhr_input_output.dhr_loaders import tiff_loader
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register two WSI qptiff images using DeeperHistReg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--he-qptiff", 
        required=True, 
        type=str,
        help="Path to the H&E stained WSI qptiff file (source image)"
    )
    
    parser.add_argument(
        "--if-qptiff", 
        required=True, 
        type=str,
        help="Path to the IF stained WSI qptiff file (target image)"
    )
    
    parser.add_argument(
        "--output-dir", 
        required=True, 
        type=str,
        help="Output directory for registration results"
    )
    
    parser.add_argument(
        "--registration-level", 
        default=0, 
        type=int,
        help="Pyramid level for registration (0 = highest resolution)"
    )
    
    parser.add_argument(
        "--save-displacement-field", 
        action="store_true",
        help="Save the displacement field for further analysis"
    )
    
    parser.add_argument(
        "--copy-target", 
        action="store_true",
        help="Copy the target image to output directory"
    )
    
    parser.add_argument(
        "--keep-temp", 
        action="store_true",
        help="Keep temporary files after registration"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualization plots before and after registration"
    )
    
    parser.add_argument(
        "--patch-size", 
        default=1024, 
        type=int,
        help="Size of patches for visualization"
    )
    
    parser.add_argument(
        "--patch-offset-x", 
        default=1000, 
        type=int,
        help="X offset for patch extraction"
    )
    
    parser.add_argument(
        "--patch-offset-y", 
        default=1000, 
        type=int,
        help="Y offset for patch extraction"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments."""
    he_path = pathlib.Path(args.he_qptiff)
    if_path = pathlib.Path(args.if_qptiff)
    
    if not he_path.exists():
        raise FileNotFoundError(f"H&E qptiff file not found: {he_path}")
    
    if not if_path.exists():
        raise FileNotFoundError(f"IF qptiff file not found: {if_path}")
    
    if not he_path.suffix.lower() in ['.qptiff', '.tiff', '.tif']:
        raise ValueError(f"H&E file must be a qptiff/tiff file: {he_path}")
    
    if not if_path.suffix.lower() in ['.qptiff', '.tiff', '.tif']:
        raise ValueError(f"IF file must be a qptiff/tiff file: {if_path}")
    
    return he_path, if_path


def load_and_visualize_images(he_path, if_path, args, output_dir):
    """Load images and create visualization plots."""
    print("Loading images...")
    
    # Load images
    he_loader = tiff_loader.TIFFLoader(he_path)
    if_loader = tiff_loader.TIFFLoader(if_path)
    
    he_image = he_loader.load_level(level=args.registration_level)
    if_image = if_loader.load_level(level=args.registration_level)
    
    print(f"H&E image shape: {he_image.shape}")
    print(f"IF image shape: {if_image.shape}")
    
    if args.visualize:
        print("Creating pre-registration visualizations...")
        
        # Create output directory for visualizations
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Full images
        plt.figure(figsize=(15, 10), dpi=150)
        plt.subplot(2, 2, 1)
        plt.imshow(he_image)
        plt.title("H&E Image (Source)")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(if_image)
        plt.title("IF Image (Target)")
        plt.axis('off')
        
        # Patches
        try:
            he_patch = he_loader.load_region(
                level=args.registration_level, 
                offset=(args.patch_offset_x, args.patch_offset_y), 
                shape=(args.patch_size, args.patch_size)
            )
            if_patch = if_loader.load_region(
                level=args.registration_level, 
                offset=(args.patch_offset_x, args.patch_offset_y), 
                shape=(args.patch_size, args.patch_size)
            )
            
            plt.subplot(2, 2, 3)
            plt.imshow(he_patch)
            plt.title(f"H&E Patch ({args.patch_size}x{args.patch_size})")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(if_patch)
            plt.title(f"IF Patch ({args.patch_size}x{args.patch_size})")
            plt.axis('off')
            
        except Exception as e:
            print(f"Warning: Could not extract patches: {e}")
        
        plt.tight_layout()
        plt.savefig(viz_dir / "pre_registration.png", bbox_inches='tight')
        plt.close()
        
        print(f"Pre-registration visualization saved to: {viz_dir / 'pre_registration.png'}")
    
    return he_loader, if_loader


def run_registration(he_path, if_path, output_dir, args):
    """Run the registration process."""
    print("Setting up registration configuration...")
    
    # Define registration parameters
    registration_params = default_initial_nonrigid()
    registration_params['loading_params']['loader'] = 'tiff'  # For qptiff/tiff formats
    
    # Create case name
    case_name = f"{he_path.stem}_{if_path.stem}"
    
    # Set up temporary path
    temporary_path = output_dir / f"{case_name}_TEMP"
    
    # Create configuration
    config = {
        'source_path': he_path,
        'target_path': if_path,
        'output_path': output_dir,
        'registration_parameters': registration_params,
        'case_name': case_name,
        'save_displacement_field': args.save_displacement_field,
        'copy_target': args.copy_target,
        'delete_temporary_results': not args.keep_temp,
        'temporary_path': temporary_path
    }
    
    print("Starting registration...")
    print(f"Source: {he_path}")
    print(f"Target: {if_path}")
    print(f"Output: {output_dir}")
    print(f"Case name: {case_name}")
    
    # Run registration
    try:
        deeperhistreg.run_registration(**config)
        print("Registration completed successfully!")
        return True
    except Exception as e:
        print(f"Registration failed: {e}")
        return False


def visualize_results(he_path, if_path, output_dir, args):
    """Create post-registration visualizations."""
    if not args.visualize:
        return
    
    print("Creating post-registration visualizations...")
    
    # Look for registered output
    registered_source_path = None
    for ext in ['.tiff', '.tif']:
        potential_path = output_dir / f"{he_path.stem}_registered{ext}"
        if potential_path.exists():
            registered_source_path = potential_path
            break
    
    if registered_source_path is None:
        print("Warning: Could not find registered source image for visualization")
        return
    
    try:
        # Load registered source and original target
        registered_loader = tiff_loader.TIFFLoader(registered_source_path)
        target_loader = tiff_loader.TIFFLoader(if_path)
        
        registered_image = registered_loader.load_level(level=0)
        target_image = target_loader.load_level(level=0)
        
        # Create visualization
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(15, 10), dpi=150)
        
        plt.subplot(2, 2, 1)
        plt.imshow(registered_image)
        plt.title("Registered H&E Image")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(target_image)
        plt.title("Target IF Image")
        plt.axis('off')
        
        # Patches
        try:
            registered_patch = registered_loader.load_region(
                level=0, 
                offset=(args.patch_offset_x, args.patch_offset_y), 
                shape=(args.patch_size, args.patch_size)
            )
            target_patch = target_loader.load_region(
                level=0, 
                offset=(args.patch_offset_x, args.patch_offset_y), 
                shape=(args.patch_size, args.patch_size)
            )
            
            plt.subplot(2, 2, 3)
            plt.imshow(registered_patch)
            plt.title(f"Registered H&E Patch ({args.patch_size}x{args.patch_size})")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(target_patch)
            plt.title(f"Target IF Patch ({args.patch_size}x{args.patch_size})")
            plt.axis('off')
            
        except Exception as e:
            print(f"Warning: Could not extract post-registration patches: {e}")
        
        plt.tight_layout()
        plt.savefig(viz_dir / "post_registration.png", bbox_inches='tight')
        plt.close()
        
        print(f"Post-registration visualization saved to: {viz_dir / 'post_registration.png'}")
        
    except Exception as e:
        print(f"Warning: Could not create post-registration visualization: {e}")


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Validate inputs
        he_path, if_path = validate_inputs(args)
        
        # Create output directory
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"WSI Registration Pipeline")
        print(f"========================")
        print(f"H&E Image: {he_path}")
        print(f"IF Image: {if_path}")
        print(f"Output Directory: {output_dir}")
        print()
        
        # Load and visualize images (if requested)
        he_loader, if_loader = load_and_visualize_images(he_path, if_path, args, output_dir)
        
        # Run registration
        success = run_registration(he_path, if_path, output_dir, args)
        
        if success:
            # Create post-registration visualizations
            visualize_results(he_path, if_path, output_dir, args)
            
            print("\nRegistration pipeline completed successfully!")
            print(f"Results saved to: {output_dir}")
            
            # List output files
            print("\nOutput files:")
            for file in sorted(output_dir.glob("*")):
                if file.is_file():
                    print(f"  - {file.name}")
        else:
            print("\nRegistration pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
