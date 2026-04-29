import os
import argparse
import numpy as np
from pathlib import Path
import torch

def load_clip_embeddings(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'CLIP embeddings not found at {path}. Run clip_utils.py first.')
    return np.load(path)

def compute_metrics(pred_labels, true_labels, num_classes=13):
    # Calculate OA
    oa = np.mean(pred_labels == true_labels)
    
    # Calculate mIoU
    ious = []
    for c in range(num_classes):
        intersection = np.sum((pred_labels == c) & (true_labels == c))
        union = np.sum((pred_labels == c) | (true_labels == c))
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)
    
    miou = np.nanmean(ious)
    return oa, miou, ious

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='features/s3dis_area5')
    parser.add_argument('--clip_emb', type=str, default='data/s3dis_processed/label_to_clip_embeddings.npy')
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f'Features directory {features_dir} not found. Run extract_features.py first.')
        return
        
    clip_embeddings = load_clip_embeddings(args.clip_emb)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_embeddings_torch = torch.from_numpy(clip_embeddings).float().to(device)
    
    # TODO: Wait for Leonardo's translation_head
    print('MOCKUP MODE: src/translation_head.py is missing.')
    print('Generating random predictions to test evaluation logic.')
    
    all_preds = []
    all_labels = []
    
    for npz_file in features_dir.glob('*.npz'):
        data = np.load(npz_file)
        features = data['features'] # (N, 256)
        labels = data['labels'] # (N,)
        
        num_points = features.shape[0]
        
        # MOCKUP: Simulate MLP projecting to CLIP space (N, 512)
        # Instead of actually projecting the random 256 to 512, we just generate random 512
        pred_clip = torch.randn(num_points, 512).float().to(device)
        pred_clip = torch.nn.functional.normalize(pred_clip, p=2, dim=-1)
        
        # Compute cosine similarity with the 13 class embeddings
        # pred_clip: (N, 512), clip_embeddings_torch: (13, 512)
        sims = pred_clip @ clip_embeddings_torch.T # (N, 13)
        
        # Get predicted class (highest similarity)
        pred_classes = torch.argmax(sims, dim=-1).cpu().numpy()
        
        all_preds.append(pred_classes)
        all_labels.append(labels)
        print(f'Evaluated {npz_file.name}')
        
    if not all_preds:
        print('No features found to evaluate.')
        return
        
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    oa, miou, ious = compute_metrics(all_preds, all_labels)
    
    print('\n--- EVALUATION RESULTS ---')
    print(f'Overall Accuracy (OA): {oa:.4f}')
    print(f'Mean IoU (mIoU):       {miou:.4f}')
    print('Per-class IoU:')
    for c, iou in enumerate(ious):
        val = f'{iou:.4f}' if not np.isnan(iou) else 'N/A'
        print(f'  Class {c:2d}: {val}')

if __name__ == '__main__':
    main()
