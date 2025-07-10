```python
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
import os
from typing import Union

import numpy as np
import torch as tc
import matplotlib.pyplot as plt

# GPU Optimization for H200
ios.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
tc.cuda.empty_cache()  # Clear GPU cache

### DeeperHistReg Imports ###
import deeperhistreg
from deeperhistreg.dhr_input_output.dhr_loaders import tiff_loader
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Register two WSI qptiff images using DeeperHistReg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--he-qptiff", required=True, type=str,
                        help="Path to the H&E stained WSI qptiff file (source image)")
    parser.add_argument("--if-qptiff", required=True, type=str,
                        help="Path to the IF stained WSI qptiff file (target image)")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Output directory for registration results")
    parser.add_argument("--registration-level", default=0, type=int,
                        help="Pyramid level for registration (0 = highest resolution)")
    parser.add_argument("--save-displacement-field", action="store_true",
                        help="Save the displacement field for further analysis")
    parser.add_argument("--copy-target", action="store_true",
                        help="Copy the target image to output directory")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files after registration")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots before and after registration")
    parser.add_argument("--patch-size", default=1024, type=int,
                        help="Size of patches for visualization")
    parser.add_argument("--patch-offset-x", default=1000, type=int,
                        help="X offset for patch extraction")
    parser.add_argument("--patch-offset-y", default=1000, type=int,
                        help="Y offset for patch extraction")
    parser.add_argument("--gpu-memory-fraction", default=0.9, type=float,
                        help="Fraction of GPU memory to use (0.1-1.0)")
    parser.add_argument("--batch-size", default=None, type=int,
                        help="Batch size for processing (auto-determined if not set)")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="Number of data loading workers")
    return parser.parse_args()


def setup_gpu_environment():
    print("Setting up GPU environment...")
    if tc.cuda.is_available():
        gpu_name = tc.cuda.get_device_name(0)
        gpu_memory = tc.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {tc.version.cuda}")
        print(f"PyTorch Version: {tc.__version__}")
        tc.cuda.set_per_process_memory_fraction(0.9)
        tc.backends.cudnn.benchmark = True
        tc.backends.cudnn.deterministic = False
        return True
    else:
        print("Warning: CUDA not available, using CPU")
        return False


def validate_inputs(args):
    he_path = pathlib.Path(args.he_qptiff)
    if_path = pathlib.Path(args.if_qptiff)
    if not he_path.exists():
        raise FileNotFoundError(f"H&E qptiff file not found: {he_path}")
    if not if_path.exists():
        raise FileNotFoundError(f"IF qptiff file not found: {if_path}")
    if he_path.suffix.lower() not in ['.qptiff', '.tiff', '.tif']:
        raise ValueError(f"H&E file must be a qptiff/tiff file: {he_path}")
    if if_path.suffix.lower() not in ['.qptiff', '.tiff', '.tif']:
        raise ValueError(f"IF file must be a qptiff/tiff file: {if_path}")
    return he_path, if_path


def load_and_analyze_images(he_path, if_path, args, output_dir):
    print("Loading images...")
    he_loader = tiff_loader.TIFFLoader(he_path)
    if_loader = tiff_loader.TIFFLoader(if_path)
    he_image = he_loader.load_level(level=args.registration_level)
    if_image = if_loader.load_level(level=args.registration_level)
    print(f"H&E image shape: {he_image.shape}")
    print(f"IF image shape: {if_image.shape}")
    he_stats = {'mean': np.mean(he_image), 'std': np.std(he_image),
                'min': np.min(he_image), 'max': np.max(he_image)}
    if_stats = {'mean': np.mean(if_image), 'std': np.std(if_image),
                'min': np.min(if_image), 'max': np.max(if_image)}
    print(f"H&E stats - Mean: {he_stats['mean']:.2f}, Std: {he_stats['std']:.2f}, Range: [{he_stats['min']}-{he_stats['max']}]" )
    print(f"IF stats - Mean: {if_stats['mean']:.2f}, Std: {if_stats['std']:.2f}, Range: [{if_stats['min']}-{if_stats['max']}]" )
    if args.visualize:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        plt.switch_backend('Agg')
        plt.figure(figsize=(15, 10), dpi=150)
        plt.subplot(2, 2, 1); plt.imshow(he_image); plt.title("H&E Image (Source)"); plt.axis('off')
        plt.subplot(2, 2, 2); plt.imshow(if_image); plt.title("IF Image (Target)"); plt.axis('off')
        try:
            he_patch = he_loader.load_region(level=args.registration_level,
                                            offset=(args.patch_offset_x, args.patch_offset_y),
                                            shape=(args.patch_size, args.patch_size))
            if_patch = if_loader.load_region(level=args.registration_level,
                                            offset=(args.patch_offset_x, args.patch_offset_y),
                                            shape=(args.patch_size, args.patch_size))
            plt.subplot(2, 2, 3); plt.imshow(he_patch); plt.title(f"H&E Patch ({args.patch_size}x{args.patch_size})"); plt.axis('off')
            plt.subplot(2, 2, 4); plt.imshow(if_patch); plt.title(f"IF Patch ({args.patch_size}x{args.patch_size})"); plt.axis('off')
            print(f"H&E patch shape: {he_patch.shape}")
            print(f"IF patch shape: {if_patch.shape}")
        except Exception as e:
            print(f"Warning: Could not extract patches: {e}")
        plt.tight_layout(); plt.savefig(viz_dir / "pre_registration.png", bbox_inches='tight'); plt.close()
        print(f"Pre-registration visualization saved to: {viz_dir / 'pre_registration.png'}")
    return he_loader, if_loader


def run_registration(he_path, if_path, output_dir, args):
    print("Setting up registration configuration...")
    params = default_initial_nonrigid()
    params['loading_params']['loader'] = 'tiff'
    if tc.cuda.is_available():
        params['device'] = 'cuda'; params['mixed_precision'] = True
        params['batch_size'] = args.batch_size if args.batch_size else (
            8 if tc.cuda.get_device_properties(0).total_memory/1024**3 >= 80 else 4)
        params['num_workers'] = args.num_workers
        print(f"Using GPU with batch size: {params['batch_size']}")
    case_name = f"{he_path.stem}_{if_path.stem}"
    temp_path = output_dir / f"{case_name}_TEMP"
    config = {'source_path': he_path, 'target_path': if_path,
              'output_path': output_dir, 'registration_parameters': params,
              'case_name': case_name, 'save_displacement_field': args.save_displacement_field,
              'copy_target': args.copy_target, 'delete_temporary_results': not args.keep_temp,
              'temporary_path': temp_path}
    print(f"Source: {he_path}\nTarget: {if_path}\nOutput: {output_dir}\nCase name: {case_name}")
    try:
        if tc.cuda.is_available(): print(f"Initial GPU memory: {tc.cuda.memory_allocated()/1024**3:.2f} GB")
        deeperhistreg.run_registration(**config)
        if tc.cuda.is_available(): tc.cuda.empty_cache(); print(f"Final GPU memory: {tc.cuda.memory_allocated()/1024**3:.2f} GB")
        print("Registration completed successfully!")
        return True
    except Exception as e:
        print(f"Registration failed: {e}")
        if tc.cuda.is_available(): tc.cuda.empty_cache()
        return False


def visualize_results(he_path, if_path, output_dir, args):
    if not args.visualize: return
    print("Creating post-registration visualizations...")
    reg_path = None
    for ext in ['.tiff', '.tif']:
        p = output_dir / f"{he_path.stem}_registered{ext}"
        if p.exists(): reg_path = p; break
    if not reg_path:
        print("Warning: Registered image not found")
        return
    try:
        reg_loader = tiff_loader.TIFFLoader(reg_path)
        tgt_loader = tiff_loader.TIFFLoader(if_path)
        reg_img = reg_loader.load_level(level=0)
        tgt_img = tgt_loader.load_level(level=0)
        print(f"Registered shape: {reg_img.shape}, Target shape: {tgt_img.shape}")
        reg_stats = {'mean': np.mean(reg_img), 'std': np.std(reg_img), 'min': np.min(reg_img), 'max': np.max(reg_img)}
        tgt_stats = {'mean': np.mean(tgt_img), 'std': np.std(tgt_img), 'min': np.min(tgt_img), 'max': np.max(tgt_img)}
        print(f"Registered stats - Mean: {reg_stats['mean']:.2f}, Std: {reg_stats['std']:.2f}, Range: [{reg_stats['min']}-{reg_stats['max']}]" )
        print(f"Target stats - Mean: {tgt_stats['mean']:.2f}, Std: {tgt_stats['std']:.2f}, Range: [{tgt_stats['min']}-{tgt_stats['max']}]" )
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        plt.switch_backend('Agg')
        plt.figure(figsize=(15, 10), dpi=150)
        plt.subplot(2, 2, 1); plt.imshow(reg_img); plt.title("Registered H&E Image"); plt.axis('off')
        plt.subplot(2, 2, 2); plt.imshow(tgt_img); plt.title("Target IF Image"); plt.axis('off')
        try:
            reg_patch = reg_loader.load_region(level=0,
                                              offset=(args.patch_offset_x, args.patch_offset_y),
                                              shape=(args.patch_size, args.patch_size))
            tgt_patch = tgt_loader.load_region(level=0,
                                              offset=(args.patch_offset_x, args.patch_offset_y),
                                              shape=(args.patch_size, args.patch_size))
            plt.subplot(2, 2, 3); plt.imshow(reg_patch); plt.title(f"Registered Patch ({args.patch_size})"); plt.axis('off')
            plt.subplot(2, 2, 4); plt.imshow(tgt_patch); plt.title(f"Target Patch ({args.patch_size})"); plt.axis('off')
        except Exception as e:
            print(f"Warning: Could not extract post-registration patches: {e}")
        plt.tight_layout(); plt.savefig(viz_dir / "post_registration.png", bbox_inches='tight'); plt.close()
        print(f"Post-registration visualization saved to: {viz_dir / 'post_registration.png'}")
    except Exception as e:
        print(f"Warning: Could not create post-registration visualization: {e}")


def main():
    args = parse_arguments()
    try:
        gpu_available = setup_gpu_environment()
        if gpu_available:
            tc.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
            print(f"Set GPU memory fraction to: {args.gpu_memory_fraction}")
        he_path, if_path = validate_inputs(args)
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWSI Registration Pipeline (H200 Optimized)"); print(f"==========================================")
        print(f"H&E Image: {he_path}"); print(f"IF Image: {if_path}"); print(f"Output Directory: {output_dir}")
        he_loader, if_loader = load_and_analyze_images(he_path, if_path, args, output_dir)
        success = run_registration(he_path, if_path, output_dir, args)
        if success:
            visualize_results(he_path, if_path, output_dir, args)
            print("\nRegistration pipeline completed successfully!"); print(f"Results saved to: {output_dir}")
            print("\nOutput files:");
            for file in sorted(output_dir.glob("*")):
                if file.is_file(): print(f"  - {file.name}")
        else:
            print("\nRegistration pipeline failed!"); sys.exit(1)
        if gpu_available: tc.cuda.empty_cache(); print(f"Final GPU memory usage: {tc.cuda.memory_allocated()/1024**3:.2f} GB")
    except Exception as e:
        print(f"Error: {e}")
        if tc.cuda.is_available(): tc.cuda.empty_cache()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

