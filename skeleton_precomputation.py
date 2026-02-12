import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from stage1_model import MMWavePoseTransformer
from stage1_dataloader import MMFi_mmWave_Dataset 

def precompute_skeletons(csv_path, data_root, model_path, output_dir="./data/precomputed_skeletons"):
    device = torch.device("cuda")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and freeze Stage 1 Model
    model = MMWavePoseTransformer(num_joints=17).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load the cleaned Action Segments CSV
    df = pd.read_csv(csv_path)
    
    # Prepare a dummy dataset for aggregation
    dummy_ds = MMFi_mmWave_Dataset(data_root, subjects=[], actions=[])
    
    print(f"Pre-computing skeletons for {len(df)} segments...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mmwave_dir = os.path.join(data_root, row['Environment'], row['Subject'], row['Action'], 'mmwave')
        
        # Unique filename: S01_A04_start1_end11.npy
        out_name = f"{row['Subject']}_{row['Action']}_start{row['Start']}_end{row['End']}.npy"
        out_path = os.path.join(output_dir, out_name)
        
        if os.path.exists(out_path):
            continue

        segment_skeletons = []
        
        with torch.no_grad():
            # MMFi frame indices are typically 1-based in naming (frame001.bin)
            # but 0-based in our dataloader logic.
            for f_idx in range(row['Start'] - 1, row['End']):
                # Reuse the 5-frame aggregation logic from your Stage 1 dataloader
                points = dummy_ds._get_aggregated_points(mmwave_dir, f_idx)
                
                # Check for empty aggregated frames
                if points.shape[0] == 0:
                    points = np.zeros((1, 5))
                
                # Move to GPU and predict
                points_tensor = torch.FloatTensor(points).to(device).unsqueeze(0)
                skeleton = model(points_tensor) # Output: [1, 17, 3]
                
                segment_skeletons.append(skeleton.squeeze(0).cpu().numpy())
        
        # Save the sequence of skeletons for this specific repetition
        np.save(out_path, np.array(segment_skeletons))

if __name__ == "__main__":
    

    CSV_FILE = "./data/MMFi_action_segments_rmA1_2_3_6_len10to30.csv"
    DATA_ROOT = "./data/content/MMFi"
    MODEL_WEIGHTS = "best_mmwave_model.pth"
    
    precompute_skeletons(CSV_FILE, DATA_ROOT, MODEL_WEIGHTS)