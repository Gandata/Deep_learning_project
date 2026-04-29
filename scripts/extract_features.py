import os
import argparse
import numpy as np
from pathlib import Path
import sys

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import S3DISDataset

def main():
    parser = argparse.ArgumentParser(description='Extract features for S3DIS Area 5')
    parser.add_argument('--data_dir', type=str, default='data/s3dis_processed', help='Path to raw data directory')
    parser.add_argument('--out_dir', type=str, default='features/s3dis_area5', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading Area 5 from {args.data_dir}...')
    try:
        dataset = S3DISDataset(root=args.data_dir, areas=[5])
    except RuntimeError as e:
        print(e)
        print("Make sure Adrian's data preprocessing is done.")
        return

    # TODO: Wait for Leonardo to provide src.encoder.py
    print('MOCKUP MODE: src/encoder.py is missing.')
    print('Generating random feature arrays (D=256) to unblock evaluation pipeline.')
    
    for i in range(len(dataset)):
        sample = dataset[i]
        room_name = sample['room'].replace('/', '_') # e.g. Area_5_conferenceRoom_1
        out_path = out_dir / f'{room_name}.npz'
        
        if out_path.exists():
            print(f'Skipping {room_name}, already exists.')
            continue
            
        num_points = sample['coord'].shape[0]
        
        # MOCKUP: Generate random (N, 256) features
        features = np.random.randn(num_points, 256).astype(np.float32)
        
        # Save as npz
        np.savez_compressed(
            out_path,
            features=features,
            labels=sample['label'],
            coord=sample['coord']
        )
        print(f'Saved mock features for {room_name} -> {out_path}')

if __name__ == '__main__':
    main()
