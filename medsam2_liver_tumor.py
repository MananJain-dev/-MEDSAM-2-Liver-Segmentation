"""
MedSAM2 Liver Tumor Segmentation for LiTS Dataset
Corrected Version: Fixes SAM2 initialization and Liver Detection Logic.

Usage:
python medsam2_liver_tumor.py --ct_path "Training_Batch2/volume-30.nii" --output_dir output_folder --checkpoint_path "checkpoints/MedSAM2_latest.pt"
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
from skimage import measure, morphology, filters
from scipy import ndimage
import shutil

# Color settings for visualization
LIVER_COLOR = np.array([0, 255, 0])  # Green for liver
TUMOR_COLOR = np.array([255, 0, 0])  # Red for tumor

def download_checkpoint(checkpoint_dir="checkpoints"):
    """Download MedSAM2 checkpoint if not present"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "MedSAM2_latest.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint already exists at {checkpoint_path}")
        return checkpoint_path
    
    print("Downloading MedSAM2 checkpoint...")
    url = "https://github.com/bowang-lab/MedSAM2/releases/download/v1.0/MedSAM2_latest.pt"
    
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
            def reporthook(blocknum, blocksize, totalsize):
                t.total = totalsize
                t.update(blocknum * blocksize - t.n)
            
            urllib.request.urlretrieve(url, checkpoint_path, reporthook=reporthook)
        print(f"✓ Checkpoint downloaded successfully to {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to download checkpoint: {e}")
        print("Please download manually from: https://github.com/bowang-lab/MedSAM2")
        sys.exit(1)
    
    return checkpoint_path

def install_medsam2():
    """Clone and install MedSAM2 repository"""
    if os.path.exists("MedSAM2"):
        print("✓ MedSAM2 repository already exists")
        sys.path.insert(0, os.path.abspath("MedSAM2"))
        return
    
    print("Cloning MedSAM2 repository...")
    os.system("git clone https://github.com/bowang-lab/MedSAM2.git")
    
    if not os.path.exists("MedSAM2"):
        print("✗ Failed to clone repository")
        sys.exit(1)
    
    print("Installing MedSAM2 dependencies...")
    os.system("pip install -e MedSAM2")
    sys.path.insert(0, os.path.abspath("MedSAM2"))
    print("✓ MedSAM2 installed successfully")

def get_largest_cc(segmentation):
    """Get largest connected component"""
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def process_ct_scan(ct_path, output_dir, checkpoint_path):
    """Process single CT scan with MedSAM2 (Fixed Initialization)"""
    
    print(f"\n{'='*60}")
    print(f"Processing CT scan: {ct_path}")
    print(f"{'='*60}\n")
    
    # Create output directories
    slices_dir = os.path.join(output_dir, "slices")
    results_dir = os.path.join(output_dir, "results")
    overlay_dir = os.path.join(output_dir, "overlays")
    
    for d in [slices_dir, results_dir, overlay_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 1. Load CT scan
    print("1. Loading CT scan...")
    nii_image = sitk.ReadImage(ct_path)
    nii_data = sitk.GetArrayFromImage(nii_image)
    print(f"   ✓ CT shape: {nii_data.shape}")
    
    # 2. Apply windowing for liver
    print("2. Applying liver windowing...")
    lower_bound, upper_bound = -100, 400
    nii_data_windowed = np.clip(nii_data, lower_bound, upper_bound)
    nii_data_windowed = (nii_data_windowed - np.min(nii_data_windowed)) / \
                        (np.max(nii_data_windowed) - np.min(nii_data_windowed)) * 255.0
    nii_data_windowed = np.uint8(nii_data_windowed)
    
    key_slice_idx = nii_data.shape[0] // 2
    
    # 3. Save slices
    print("3. Saving individual slices...")
    # SAM2 requires filenames to be sorted (e.g. 00000.jpg)
    for i in tqdm(range(nii_data.shape[0]), desc="   Saving slices"):
        slice_img = Image.fromarray(nii_data_windowed[i])
        slice_img.save(os.path.join(slices_dir, f"{i:05d}.jpg"))
    print(f"   ✓ Saved {nii_data.shape[0]} slices to {slices_dir}")
    
    # 4. Automatic liver detection (IMPROVED LOGIC)
    print("\n4. Detecting liver region automatically...")
    key_slice = nii_data_windowed[key_slice_idx]
    h, w = key_slice.shape
    
    # --- FIX: Anatomy-Aware Detection ---
    # 1. Create a "Search Mask"
    # Liver is on patient's right -> Image LEFT side
    search_mask = np.zeros_like(key_slice, dtype=bool)
    search_mask[int(h*0.2):int(h*0.8), 0:int(w*0.5)] = True
    
    # 2. Exclude the Spine (Center of image)
    center_x = w // 2
    spine_width = int(w * 0.15)
    search_mask[:, center_x - spine_width : center_x + spine_width] = False

    # 3. Thresholding for soft tissue
    tissue_mask = (key_slice > 50) & (key_slice < 220)
    
    # Combine
    candidates = search_mask & tissue_mask
    
    # 4. Clean up noise
    candidates = morphology.remove_small_objects(candidates, min_size=1000)
    candidates = morphology.binary_closing(candidates, morphology.disk(10))
    
    # 5. Select the Liver (Largest object in search zone)
    labeled_regions = measure.label(candidates)
    regions = measure.regionprops(labeled_regions, intensity_image=key_slice)
    
    if len(regions) > 0:
        liver_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = liver_region.bbox
        padding = 30 
        bbox_xyxy = np.array([
            max(0, minc - padding),
            max(0, minr - padding),
            min(w, maxc + padding),
            min(h, maxr + padding)
        ])
        print(f"   ✓ Liver detected! BBox: {bbox_xyxy}")
        liver_candidates_viz = candidates
    else:
        print("   ⚠ Detection uncertain. Using fallback box.")
        bbox_xyxy = np.array([0, int(h*0.2), int(w*0.5), int(h*0.8)])
        liver_candidates_viz = np.zeros_like(key_slice)

    # Visualize detection
    plt.figure(figsize=(10, 5))
    plt.imshow(key_slice, cmap='gray')
    rect = plt.Rectangle((bbox_xyxy[0], bbox_xyxy[1]), 
                         bbox_xyxy[2]-bbox_xyxy[0], 
                         bbox_xyxy[3]-bbox_xyxy[1],
                         fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title('Detected Liver Region')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'liver_detection.png'), bbox_inches='tight')
    plt.close()
    
    # 5. Initialize MedSAM2 (CORRECTED)
    print("\n5. Initializing MedSAM2 model...")
    try:
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        
        # 1. Locate Config
        config_dir = os.path.abspath("MedSAM2/sam2/configs")
        # Use the specific file present in your system
        config_name = "sam2.1_hiera_t512.yaml"
        
        # Fallback scan if file names differ slightly
        if not os.path.exists(os.path.join(config_dir, config_name)):
            found = [f for f in os.listdir(config_dir) if 'hiera_t' in f and f.endswith('.yaml')]
            if found: config_name = found[0]
            
        print(f"   ✓ Using config: {config_name}")

        # 2. Load Config with Hydra
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            
        # 3. OVERRIDE TARGET to VideoPredictor
        # This tells Hydra to build the 'SAM2VideoPredictor' class instead of the base model
        cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
        
        # 4. Instantiate the full predictor directly
        predictor = instantiate(cfg.model, _recursive_=True)
        
        # 5. Load Weights
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        predictor.load_state_dict(sd)
        predictor = predictor.cuda().eval()
        
        print("   ✓ Model instantiated and loaded successfully")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Run inference
    print("\n6. Running MedSAM2 inference...")
    segs_3D = np.zeros(nii_data.shape, dtype=np.uint8)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialize state from folder
        inference_state = predictor.init_state(video_path=slices_dir)
        
        # Add prompt
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_slice_idx,
            obj_id=1,
            box=bbox_xyxy,
        )
        
        # Forward pass
        print("   Propagating forward...")
        for out_frame_idx, _, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state),
            desc="   Forward",
            total=nii_data.shape[0]-key_slice_idx
        ):
            segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        
        # Backward pass
        predictor.reset_state(inference_state)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_slice_idx,
            obj_id=1,
            box=bbox_xyxy,
        )
        
        print("   Propagating backward...")
        for out_frame_idx, _, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state, reverse=True),
            desc="   Backward",
            total=key_slice_idx
        ):
            segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            
    print("   ✓ Inference complete")
    
    # 7. Post-process & Save
    print("\n7. Saving results...")
    if np.max(segs_3D) > 0:
        segs_3D = get_largest_cc(segs_3D).astype(np.uint8)

    liver_mask = segs_3D.copy()
    tumor_mask = np.zeros_like(segs_3D)
    
    # Separate Tumor (Heuristic: darker regions in liver)
    for i in range(segs_3D.shape[0]):
        if np.sum(segs_3D[i]) > 0:
            liver_region = nii_data_windowed[i] * segs_3D[i]
            if np.sum(liver_region) > 0:
                thresh = np.percentile(liver_region[liver_region > 0], 25)
                tumor_mask[i] = (liver_region < thresh) & (liver_region > 0)
    liver_mask = liver_mask & (~tumor_mask)

    # Save NIfTI files
    for name, mask in [("liver", liver_mask), ("tumor", tumor_mask), ("combined", liver_mask + tumor_mask*2)]:
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(nii_image)
        sitk.WriteImage(img, os.path.join(results_dir, f"{name}_mask.nii.gz"))
    
    # Save Overlay Images
    for i in tqdm(range(segs_3D.shape[0]), desc="   Saving overlays"):
        img_rgb = np.stack([nii_data_windowed[i]] * 3, axis=-1).astype(np.float32)
        
        # Overlay Liver (Green)
        liver_px = liver_mask[i] > 0
        img_rgb[liver_px] = img_rgb[liver_px] * 0.5 + LIVER_COLOR * 0.5
        
        # Overlay Tumor (Red)
        tumor_px = tumor_mask[i] > 0
        img_rgb[tumor_px] = img_rgb[tumor_px] * 0.3 + TUMOR_COLOR * 0.7
        
        Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8)).save(os.path.join(overlay_dir, f"overlay_{i:04d}.png"))
        
    # Save Summary
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    slices_to_show = [key_slice_idx-10, key_slice_idx, key_slice_idx+10]
    
    for idx, slice_num in enumerate(slices_to_show):
        if 0 <= slice_num < segs_3D.shape[0]:
            # Original
            axes[0, idx].imshow(nii_data_windowed[slice_num], cmap='gray')
            axes[0, idx].set_title(f'Original - Slice {slice_num}')
            axes[0, idx].axis('off')
            
            # Overlay
            img_path = os.path.join(overlay_dir, f"overlay_{slice_num:04d}.png")
            if os.path.exists(img_path):
                axes[1, idx].imshow(Image.open(img_path))
            axes[1, idx].set_title(f'Segmentation - Slice {slice_num}')
            axes[1, idx].axis('off')
            
    plt.suptitle('MedSAM2 Liver Tumor Segmentation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
        
    print(f"\nResults saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MedSAM2 Liver Tumor Segmentation')
    parser.add_argument('--ct_path', type=str, required=True, help='Path to CT scan (.nii or .nii.gz)')
    parser.add_argument('--output_dir', type=str, default='medsam2_output', help='Output directory for results')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to MedSAM2 checkpoint')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MedSAM2 Liver Tumor Segmentation Setup")
    print("="*60)
    
    install_medsam2()
    
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print(f"\n✗ Checkpoint not found: {args.checkpoint_path}")
            sys.exit(1)
        checkpoint_path = args.checkpoint_path
        print(f"✓ Using checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = download_checkpoint()
    
    if not os.path.exists(args.ct_path):
        print(f"\n✗ CT scan file not found: {args.ct_path}")
        sys.exit(1)
    
    process_ct_scan(args.ct_path, args.output_dir, checkpoint_path)

if __name__ == "__main__":
    main()