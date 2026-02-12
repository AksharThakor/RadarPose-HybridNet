import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from stage1_dataloader import MMFi_mmWave_Dataset, collate_mmwave
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np

# Joint connectivity for the 17-joint skeleton
CONNECTIVITY = [
    (10, 9), (9, 8), (8, 7), (7, 0),       # Torso
    (8, 11), (11, 12), (12, 13),           # Left Arm
    (8, 14), (14, 15), (15, 16),           # Right Arm
    (0, 4), (4, 5), (5, 6),                # Left Leg
    (0, 1), (1, 2), (2, 3)                 # Right Leg
]

def visualize_sample(points, skeleton):
    fig = plt.figure(figsize=(15, 7))
    
    # Plot 1 Aggregated mmWave Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    # points: [N, 5] -> (x, y, z, d, I)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 4]
    
    img = ax1.scatter(x, z, y, c=intensity, cmap='viridis', s=10) # Swapping y and z for typical 3D view
    ax1.set_title(f"Aggregated mmWave Point Cloud\n({len(points)} points)")
    ax1.set_xlabel('X (Lateral)')
    ax1.set_ylabel('Z (Depth)')
    ax1.set_zlabel('Y (Vertical)')
    fig.colorbar(img, ax=ax1, label='Intensity')
    
    # Setting limits based on coordinate analysis
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([2.5, 4.0])
    ax1.set_zlim([-1, 1])

    # Plot 2 Ground Truth Skeleton
    ax2 = fig.add_subplot(122, projection='3d')
    # skeleton: [17, 3]
    js_x, js_y, js_z = skeleton[:, 0], skeleton[:, 1], skeleton[:, 2]
    
    # Plot joints
    ax2.scatter(js_x, js_z, js_y, c='red', s=40)
    
    # Plot bones
    for joint_a, joint_b in CONNECTIVITY:
        ax2.plot([js_x[joint_a], js_x[joint_b]], 
                 [js_z[joint_a], js_z[joint_b]], 
                 [js_y[joint_a], js_y[joint_b]], c='blue')
        
    ax2.set_title("Ground Truth 17-Joint Skeleton")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([2.5, 4.0])
    ax2.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.show()

# Verification Logic
if __name__ == "__main__":
    
    DATA_ROOT = "./data/content/MMFi" # Path to MMFi dataset folder
    # Change following to select different subjects/actions for verification
    subjects = ['S01']
    actions = ['A03']
    
    dataset = MMFi_mmWave_Dataset(DATA_ROOT, subjects, actions)
    print(f"Total samples in verification set: {len(dataset)}")
    
    # Test a single fetch
    sample_points, sample_gt = dataset[150] # check sample index 
    print(sample_points)
    print(f"Sample Points Shape: {sample_points.shape}")
    print(f"Sample GT Shape: {sample_gt.shape}")
    

    visualize_sample(sample_points.numpy(), sample_gt.numpy())
    
    # Test DataLoader with collate function
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_mmwave)
    batch_p, batch_gt = next(iter(dataloader))
    print(f"Batch Padded Points Shape: {batch_p.shape}") 
    print(f"Batch GT Shape: {batch_gt.shape}") 


def visualize_fixed(points, skeleton, filename="fixed_pose.png"):
    fig = plt.figure(figsize=(15, 8))
    
    # Paper coordinates: X=Lateral, Y=Vertical, Z=Depth
    # To view this naturally in Matplotlib (where Z is up):
    # Map X -> X_plot, Z -> Y_plot, Y -> -Z_plot (to flip it right-side up)
    
    # mmWave Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    if len(points) > 0:
        px, py, pz, pd, pi = points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]
        ax1.scatter(px, pz, -py, c=pi, cmap='viridis', s=15, alpha=0.6)
    
    ax1.set_title(f"mmWave Point Cloud\nPoints detected: {len(points)}")
    ax1.set_xlim([-1, 1])     # Lateral
    ax1.set_ylim([3.0, 3.8])  # Depth
    ax1.set_zlim([-1, 1])     # Vertical
    ax1.set_xlabel('X (Side)')
    ax1.set_ylabel('Z (Depth)')
    ax1.set_zlabel('-Y (Up)')

    # Ground Truth Skeleton
    ax2 = fig.add_subplot(122, projection='3d')
    sx, sy, sz = skeleton[:, 0], skeleton[:, 1], skeleton[:, 2]
    
    ax2.scatter(sx, sz, -sy, c='red', s=50)
    
    for b_start, b_end in CONNECTIVITY:
        ax2.plot([sx[b_start], sx[b_end]], 
                 [sz[b_start], sz[b_end]], 
                 [-sy[b_start], -sy[b_end]], c='blue', linewidth=2)
        
    ax2.set_title("Aligned Ground Truth Skeleton")
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([3.0, 3.8])
    ax2.set_zlim([-1, 1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('-Y')

    plt.savefig(filename)
    plt.close()
    display(Image(filename))

visualize_fixed(sample_points.numpy(), sample_gt.numpy())