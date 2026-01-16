"""
MedSAM2 Liver & Multi-Focal Tumor Segmentation
Status: "Multi-Object Mode"
- Detects multiple separate tumors simultaneously
- Assigns unique ID to each lesion for independent tracking
- Aggregates all detected lesions into one final Tumor Mask

Usage:
python medsam2_liver_tumor.py --ct_path "path/to/volume.nii" --output_dir output_folder
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
from skimage import measure, morphology
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
import matplotlib.patches as patches

# Color settings
LIVER_COLOR = np.array([0, 255, 0])
TUMOR_COLOR = np.array([255, 0, 0])

def download_checkpoint(checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "MedSAM2_latest.pt")
    if os.path.exists(checkpoint_path): return checkpoint_path
    
    print("Downloading MedSAM2 checkpoint...")
    url = "https://github.com/bowang-lab/MedSAM2/releases/download/v1.0/MedSAM2_latest.pt"
    try:
        urllib.request.urlretrieve(url, checkpoint_path)
    except:
        print("✗ Download failed. Please download MedSAM2_latest.pt manually.")
        sys.exit(1)
    return checkpoint_path

def install_medsam2():
    if os.path.exists("MedSAM2"):
        sys.path.insert(0, os.path.abspath("MedSAM2"))
        return
    print("Cloning MedSAM2...")
    os.system("git clone https://github.com/bowang-lab/MedSAM2.git")
    os.system("pip install -e MedSAM2")
    sys.path.insert(0, os.path.abspath("MedSAM2"))

def process_ct_scan(ct_path, output_dir, checkpoint_path):
    print(f"\n{'='*60}\nProcessing: {ct_path}\n{'='*60}")
    
    for d in ["slices", "results", "overlays"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    slices_dir = os.path.join(output_dir, "slices")
    
    # 1. Load & Window
    print("1. Loading & Preprocessing...")
    nii_image = sitk.ReadImage(ct_path)
    nii_data = sitk.GetArrayFromImage(nii_image)
    
    img_windowed = np.clip(nii_data, -100, 400)
    img_windowed = (img_windowed - img_windowed.min()) / (img_windowed.max() - img_windowed.min()) * 255.0
    img_windowed = np.uint8(img_windowed)
    
    print("2. Saving input slices...")
    for i in tqdm(range(len(img_windowed))):
        Image.fromarray(img_windowed[i]).save(os.path.join(slices_dir, f"{i:05d}.jpg"))
    
    # 3. Detect Prompts (Multi-Object Logic)
    key_slice_idx = len(img_windowed) // 2
    key_slice = img_windowed[key_slice_idx]
    h, w = key_slice.shape
    
    print("\n3. Generating Multi-Object Prompts...")
    
    # --- A. LIVER PROMPT ---
    search_mask = np.zeros_like(key_slice, dtype=bool)
    search_mask[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.5)] = True 
    search_mask[:, w//2 - int(w*0.15) : w//2 + int(w*0.15)] = False 
    
    tissue_mask = (key_slice > 50) & (key_slice < 220)
    candidates = search_mask & tissue_mask
    candidates = morphology.remove_small_objects(candidates, 1000)
    candidates = morphology.binary_closing(candidates, morphology.disk(15))
    
    liver_bbox = None
    regions = measure.regionprops(measure.label(candidates))
    if regions:
        liver_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = liver_region.bbox
        pad = 20
        liver_bbox = np.array([max(0,minc-pad), max(0,minr-pad), min(w,maxc+pad), min(h,maxr+pad)])
        print(f"   ✓ Prompt 1 (Liver): {liver_bbox}")
    else:
        print("   ⚠ Liver detection failed. Using fallback.")
        liver_bbox = np.array([int(w*0.1), int(h*0.2), int(w*0.5), int(h*0.8)])

    # --- B. MULTI-TUMOR PROMPTS ---
    tumor_bboxes = [] # List to store multiple tumors
    
    if liver_bbox is not None:
        x1, y1, x2, y2 = liver_bbox
        liver_crop = key_slice[y1:y2, x1:x2]
        
        if liver_crop.size > 0:
            valid_tissue = liver_crop[liver_crop > 20] # Ignore air
            
            if valid_tissue.size > 0:
                thresh = np.percentile(valid_tissue, 25) 
                tumor_candidates = (liver_crop < thresh) & (liver_crop > 20)
                tumor_candidates = morphology.remove_small_objects(tumor_candidates, min_size=20)
                tumor_candidates = morphology.binary_closing(tumor_candidates, morphology.disk(2))
                
                # MULTI-OBJECT LOGIC: Find ALL distinct blobs
                labeled_tumors = measure.label(tumor_candidates)
                tum_regions = measure.regionprops(labeled_tumors)
                
                # Sort by size (largest first) and pick top 5
                tum_regions.sort(key=lambda x: x.area, reverse=True)
                
                for idx, region in enumerate(tum_regions[:5]): # Limit to top 5 to avoid noise
                    tr, tc, tr2, tc2 = region.bbox
                    t_bbox = np.array([x1 + tc, y1 + tr, x1 + tc2, y1 + tr2])
                    tumor_bboxes.append(t_bbox)
                    print(f"   ✓ Prompt {idx+2} (Tumor {idx+1}): {t_bbox} (Area: {region.area})")
                
                if not tumor_bboxes:
                    print("   ℹ No tumor found (Tissue too uniform)")
            else:
                 print("   ℹ Crop contained only air")

    # Save Debug Image
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(key_slice, cmap='gray'); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(candidates, cmap='gray'); plt.title("Liver Mask")
    plt.subplot(1,3,3); plt.imshow(key_slice, cmap='gray')
    
    if tumor_bboxes:
        for idx, bbox in enumerate(tumor_bboxes):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1]-5, f"T{idx+1}", color='red', fontsize=8)
        plt.title(f"DETECTED {len(tumor_bboxes)} TUMORS")
    else:
        plt.title("No Tumors Found")
    plt.savefig(os.path.join(output_dir, "debug_tumor_prompt.png"))

    # 4. Initialize MedSAM2
    print("\n4. Initializing MedSAM2...")
    try:
        config_dir = os.path.abspath("MedSAM2/sam2/configs")
        config_name = "sam2.1_hiera_t512.yaml"
        if not os.path.exists(os.path.join(config_dir, config_name)):
            found = [f for f in os.listdir(config_dir) if 'hiera_t' in f and f.endswith('.yaml')]
            if found: config_name = found[0]
            
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
            predictor = instantiate(cfg.model, _recursive_=True)
            
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        predictor.load_state_dict(sd)
        predictor = predictor.cuda().eval()
        print("   ✓ Model Ready")
    except Exception as e:
        print(f"Error: {e}"); sys.exit(1)

    # 5. Run Inference (Multi-Object Tracking)
    print("\n5. Tracking Objects in 3D...")
    segs_liver = np.zeros_like(img_windowed, dtype=np.uint8)
    segs_tumor_combined = np.zeros_like(img_windowed, dtype=np.uint8)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=slices_dir)
        
        # Add Liver Prompt (ID 1)
        predictor.add_new_points_or_box(state, frame_idx=key_slice_idx, obj_id=1, box=liver_bbox)
        
        # Add MULTIPLE Tumor Prompts (IDs 2, 3, 4...)
        for idx, bbox in enumerate(tumor_bboxes):
            obj_id = idx + 2 # IDs start at 2
            predictor.add_new_points_or_box(state, frame_idx=key_slice_idx, obj_id=obj_id, box=bbox)
        
        # Propagate Forward
        print("   Propagating Forward...")
        for idx, _, out_logits in tqdm(predictor.propagate_in_video(state), total=len(img_windowed)-key_slice_idx):
             # Extract Liver (Object 1 is index 0)
             segs_liver[idx] = (out_logits[0] > 0.0).cpu().numpy()[0]
             
             # Extract All Tumors (Indices 1 onwards)
             if len(tumor_bboxes) > 0 and out_logits.shape[0] > 1:
                 # Combine all tumor masks into one layer using MAX (Union)
                 tumor_masks = (out_logits[1:] > 0.0).cpu().numpy()
                 # Collapse all tumor channels (axis 0) into one 2D mask
                 segs_tumor_combined[idx] = np.max(tumor_masks, axis=0)[0]

        # Reset & Propagate Backward
        predictor.reset_state(state)
        predictor.add_new_points_or_box(state, frame_idx=key_slice_idx, obj_id=1, box=liver_bbox)
        for idx, bbox in enumerate(tumor_bboxes):
            predictor.add_new_points_or_box(state, frame_idx=key_slice_idx, obj_id=idx+2, box=bbox)

        print("   Propagating Backward...")
        for idx, _, out_logits in tqdm(predictor.propagate_in_video(state, reverse=True), total=key_slice_idx):
             segs_liver[idx] = (out_logits[0] > 0.0).cpu().numpy()[0]
             if len(tumor_bboxes) > 0 and out_logits.shape[0] > 1:
                 tumor_masks = (out_logits[1:] > 0.0).cpu().numpy()
                 segs_tumor_combined[idx] = np.max(tumor_masks, axis=0)[0]

    # 6. Save Results
    print("\n6. Saving Data...")
    
    def get_largest_cc(segmentation):
        labels = measure.label(segmentation)
        if labels.max() == 0: return segmentation
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    if segs_liver.max() > 0: segs_liver = get_largest_cc(segs_liver).astype(np.uint8)
    
    # Clean tumor mask (keep only parts inside liver)
    if segs_tumor_combined.max() > 0: 
        segs_tumor_combined = segs_tumor_combined & segs_liver 

    # Save NIfTIs
    res_dir = os.path.join(output_dir, "results")
    for name, mask in [("liver", segs_liver), ("tumor", segs_tumor_combined)]:
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(nii_image)
        sitk.WriteImage(img, os.path.join(res_dir, f"{name}_mask.nii.gz"))

    # Save Overlays
    print("   Generating overlays...")
    ov_dir = os.path.join(output_dir, "overlays")
    for i in tqdm(range(len(img_windowed))):
        rgb = np.stack([img_windowed[i]]*3, -1).astype(np.float32)
        if segs_liver[i].sum() > 0: rgb[segs_liver[i]>0] = rgb[segs_liver[i]>0]*0.6 + LIVER_COLOR*0.4
        if segs_tumor_combined[i].sum() > 0: rgb[segs_tumor_combined[i]>0] = rgb[segs_tumor_combined[i]>0]*0.4 + TUMOR_COLOR*0.6
        Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(os.path.join(ov_dir, f"overlay_{i:04d}.png"))
        
    print(f"\nDone! Results in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='medsam2_output')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints")
    args = parser.parse_args()
    
    install_medsam2()
    ckpt = download_checkpoint(args.checkpoint_path if os.path.isdir(args.checkpoint_path) else "checkpoints")
    process_ct_scan(args.ct_path, args.output_dir, ckpt)