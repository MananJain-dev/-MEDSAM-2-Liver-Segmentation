"""
MedSAM 2 Implementation for 3D Liver Tumor Segmentation on LITS Dataset
Treats 3D CT scans as video sequences and uses memory attention for spatial continuity
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


# ============================================================================
# LITS DATASET LOADER
# ============================================================================

class LITSDataset(Dataset):
    """LITS Dataset with flexible directory handling"""
    
    def __init__(self, data_path, mode='train', transform=None):
        self.data_path = Path(data_path)
        self.mode = mode
        self.transform = transform
        self.img_dir = self._find_data_directory()
        
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {self.img_dir}")
        
        all_files = os.listdir(self.img_dir)
        self.volume_files = sorted([
            f for f in all_files 
            if (f.startswith('volume') or f.startswith('ct_train')) 
            and f.endswith('.nii') 
            and not f.startswith('segmentation')
        ])
    
    def _find_data_directory(self):
        """Find data directory with flexible path matching"""
        candidates = [
            self.data_path / 'Training_Batch2' if self.mode == 'train' 
            else self.data_path / 'Testing_Batch2',
            self.data_path / 'media' / 'nas' / '01_Datasets' / 'CT' / 'LITS' 
            / ('Training Batch 2' if self.mode == 'train' else 'Testing Batch 2'),
            self.data_path / 'training' if self.mode == 'train' 
            else self.data_path / 'testing',
            self.data_path,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                files = os.listdir(candidate)
                if any(f.endswith('.nii') for f in files):
                    return candidate
        return candidates[0]
    
    def __len__(self):
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        vol_file = self.volume_files[idx]
        vol_path = self.img_dir / vol_file
        
        vol_sitk = sitk.ReadImage(str(vol_path))
        volume = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)
        
        seg = None
        if self.mode == 'train':
            seg_patterns = [
                vol_file.replace('volume', 'segmentation'),
                f"segmentation-{vol_file.split('-')[-1]}",
            ]
            
            for pattern in seg_patterns:
                seg_path = self.img_dir / pattern
                if seg_path.exists():
                    try:
                        seg_sitk = sitk.ReadImage(str(seg_path))
                        seg_array = sitk.GetArrayFromImage(seg_sitk)
                        seg = {
                            'liver': (seg_array == 1).astype(np.uint8),
                            'tumor': (seg_array == 2).astype(np.uint8)
                        }
                        break
                    except:
                        pass
        
        volume = self._normalize_volume(volume)
        
        return {
            'volume': torch.from_numpy(volume),
            'segmentation': seg,
            'filename': vol_file,
            'shape': volume.shape
        }
    
    @staticmethod
    def _normalize_volume(volume):
        """Normalize CT to HU range"""
        volume = np.clip(volume, -200, 250)
        volume = (volume - 25) / 225
        return volume


# ============================================================================
# MEMORY FUSION MODULE
# ============================================================================

class MemoryFusionModule(nn.Module):
    """Ultra-lightweight memory fusion for spatial continuity"""
    
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, current_features, memory_features):
        """
        Fuse current and previous slice features
        
        Args:
            current_features: (B, C, H, W) current slice features
            memory_features: (B, C, H, W) previous slice features
        
        Returns:
            fused features: (B, C, H, W)
        """
        if memory_features is None:
            return current_features
        
        fused = (self.alpha * current_features + 
                (1 - self.alpha) * memory_features)
        
        return fused


# ============================================================================
# MEDSAM 2 MODEL
# ============================================================================

class SimpleMedSAM2(nn.Module):
    """Lightweight MedSAM 2 for 3D medical image segmentation"""
    
    def __init__(self, feature_dim=128, use_memory=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_memory = use_memory
        
        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Prompt encoder
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feature_dim)
        )
        
        # Memory fusion
        if use_memory:
            self.memory_fusion = MemoryFusionModule(feature_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )
    
    def forward(self, image_slice, bbox, prev_features=None):
        """
        Forward pass
        
        Args:
            image_slice: (B, 1, H, W) CT slice
            bbox: (B, 4) bounding box
            prev_features: (B, C, H, W) previous features
        
        Returns:
            mask: (B, 2, H, W) segmentation
            features: (B, C, H, W) current features
        """
        features = self.encoder(image_slice)
        
        bbox_encoded = self.prompt_encoder(bbox)
        bbox_encoded = bbox_encoded.unsqueeze(-1).unsqueeze(-1)
        bbox_encoded = bbox_encoded.expand(
            -1, -1, features.shape[2], features.shape[3]
        )
        
        features = features + bbox_encoded
        
        if self.use_memory and prev_features is not None:
            features = self.memory_fusion(features, prev_features)
        
        mask = self.decoder(features)
        
        return mask, features


# ============================================================================
# 3D SEGMENTATION PIPELINE
# ============================================================================

class MedSAM23DSegmentor:
    """3D segmentation using MedSAM 2"""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = SimpleMedSAM2(
            feature_dim=128, 
            use_memory=True
        ).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
        
        self.model.eval()
    
    def get_bounding_box(self, volume, margin=10):
        """Extract bounding box from volume"""
        nonzero = np.where(np.abs(volume) > 0.1)
        
        if len(nonzero[0]) == 0:
            return np.array([0, 0, volume.shape[1], volume.shape[2]])
        
        y_min = max(0, nonzero[1].min() - margin)
        y_max = min(volume.shape[1], nonzero[1].max() + margin)
        x_min = max(0, nonzero[2].min() - margin)
        x_max = min(volume.shape[2], nonzero[2].max() + margin)
        
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    @torch.no_grad()
    def segment_3d_fast(self, volume, bbox=None, sample_rate=5):
        """
        Fast 3D segmentation with sampling and interpolation
        
        Args:
            volume: (D, H, W) 3D CT volume
            bbox: Bounding box
            sample_rate: Process every nth slice
        
        Returns:
            segmentation: (D, H, W, 2) result
        """
        D, H, W = volume.shape
        
        if bbox is None:
            bbox = self.get_bounding_box(volume)
        
        bbox_norm = np.array([
            bbox[0]/W, bbox[1]/H, 
            bbox[2]/W, bbox[3]/H
        ], dtype=np.float32)
        
        segmentation = np.zeros((D, H, W, 2), dtype=np.float32)
        
        sampled_indices = list(range(0, D, sample_rate))
        if D - 1 not in sampled_indices:
            sampled_indices.append(D - 1)
        
        print(f"Processing {len(sampled_indices)} key slices...")
        
        segmentation_sampled = {}
        prev_features = None
        
        for idx, i in enumerate(sampled_indices):
            try:
                slice_2d = volume[i]
                input_size = 512
                slice_resized = cv2.resize(
                    slice_2d, 
                    (input_size, input_size)
                )
                
                slice_tensor = torch.from_numpy(
                    slice_resized
                ).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                bbox_tensor = torch.from_numpy(
                    bbox_norm
                ).float().unsqueeze(0).to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                use_prev = prev_features if idx > 0 else None
                mask, features = self.model(
                    slice_tensor, 
                    bbox_tensor, 
                    use_prev
                )
                
                mask_np = torch.softmax(mask, dim=1)
                mask_np = mask_np.cpu().numpy()[0]
                mask_np = mask_np.transpose(1, 2, 0)
                mask_np = cv2.resize(mask_np, (W, H))
                
                if mask_np.ndim == 2:
                    mask_np = np.stack([mask_np, mask_np], axis=-1)
                elif mask_np.shape[-1] != 2:
                    mask_np = mask_np[..., :2]
                
                segmentation_sampled[i] = mask_np
                prev_features = features.detach().cpu()
                prev_features = prev_features.to(self.device)
                
                print(f"  Slice {idx+1}/{len(sampled_indices)} "
                      f"at index {i}")
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at slice {i}, interpolating")
                    if segmentation_sampled:
                        last_seg = list(
                            segmentation_sampled.values()
                        )[-1]
                        segmentation_sampled[i] = last_seg
                    torch.cuda.empty_cache()
                else:
                    raise
        
        print("Interpolating between key slices...")
        sampled_indices_list = sorted(segmentation_sampled.keys())
        
        for j in range(len(sampled_indices_list) - 1):
            idx1 = sampled_indices_list[j]
            idx2 = sampled_indices_list[j+1]
            seg1 = segmentation_sampled[idx1]
            seg2 = segmentation_sampled[idx2]
            
            for k in range(idx1, idx2 + 1):
                if idx2 != idx1:
                    alpha = (k - idx1) / (idx2 - idx1)
                else:
                    alpha = 0
                segmentation[k] = (1 - alpha) * seg1 + alpha * seg2
        
        return segmentation
    
    def save_segmentation(self, segmentation, output_path):
        """Save segmentation masks as NIfTI"""
        liver_mask = segmentation[..., 0]
        tumor_mask = segmentation[..., 1]
        
        liver_mask = (liver_mask > 0.5).astype(np.uint8)
        tumor_mask = (tumor_mask > 0.5).astype(np.uint8)
        
        combined = np.zeros_like(liver_mask)
        combined[liver_mask == 1] = 1
        combined[tumor_mask == 1] = 2
        
        img = sitk.GetImageFromArray(combined)
        sitk.WriteImage(img, output_path)
        print(f"Saved segmentation to {output_path}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_3d_segmentation(volume, segmentation, slices=None):
    """Visualize 3D segmentation results"""
    if slices is None:
        D = volume.shape[0]
        slices = [D // 4, D // 2, 3 * D // 4]
    
    fig, axes = plt.subplots(len(slices), 3, figsize=(12, 4*len(slices)))
    
    for idx, s in enumerate(slices):
        axes[idx, 0].imshow(volume[s], cmap='gray')
        axes[idx, 0].set_title(f'CT Slice {s}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(volume[s], cmap='gray')
        axes[idx, 1].imshow(
            segmentation[s, :, :, 0], 
            cmap='Blues', 
            alpha=0.5
        )
        axes[idx, 1].set_title(f'Liver (Slice {s})')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(volume[s], cmap='gray')
        axes[idx, 2].imshow(
            segmentation[s, :, :, 1], 
            cmap='Reds', 
            alpha=0.5
        )
        axes[idx, 2].set_title(f'Tumor (Slice {s})')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == '__main__':
    config = {
        'data_path': r'D:\mini project\Training_Batch2',
        'output_dir': './segmentation_results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sample_idx': 0,
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("=" * 70)
    print("MedSAM 2: 3D Liver Tumor Segmentation")
    print("=" * 70)
    
    # Step 1: Load dataset
    print("\n[1] Loading LITS Dataset...")
    dataset = LITSDataset(config['data_path'], mode='train')
    print(f"✓ Found {len(dataset)} volumes")
    
    # Step 2: Load sample
    print(f"\n[2] Loading sample volume {config['sample_idx']}...")
    sample = dataset[config['sample_idx']]
    volume = sample['volume'].numpy()
    print(f"✓ Volume shape: {volume.shape}")
    
    # Step 3: Initialize model
    print(f"\n[3] Initializing MedSAM 2 model...")
    device = config['device']
    try:
        segmentor = MedSAM23DSegmentor(device=device)
        print(f"✓ Model loaded on {device}")
    except RuntimeError as e:
        if 'cuda' in str(e).lower():
            print(f"  Switching to CPU...")
            device = 'cpu'
            segmentor = MedSAM23DSegmentor(device=device)
            print(f"✓ Model loaded on CPU")
    
    # Step 4: Run segmentation
    print(f"\n[4] Running 3D segmentation...")
    print(f"    Using lightweight memory fusion")
    
    segmentation = segmentor.segment_3d_fast(
        volume, 
        sample_rate=5
    )
    print(f"✓ Segmentation complete: {segmentation.shape}")
    
    # Step 5: Save results
    print(f"\n[5] Saving results...")
    output_path = os.path.join(
        config['output_dir'], 
        f"{sample['filename'].replace('.nii', '_seg.nii')}"
    )
    segmentor.save_segmentation(segmentation, output_path)
    
    # Step 6: Visualize
    print(f"\n[6] Visualizing results...")
    visualize_3d_segmentation(volume, segmentation)
    
    print("\n" + "=" * 70)
    print("✓ Pipeline complete!")
    print("=" * 70)