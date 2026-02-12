import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stage1_dataloader import MMFi_mmWave_Dataset
from stage1_model import MMWavePoseTransformer

# Configuration
ACTION_PATH = "./data/content/MMFi/E01/S04/A02"   # CHANGE ONLY THIS
MODEL_PATH = "./best_mmwave_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skeleton connectivity (17 joints)
CONNECTIVITY = [
    (10, 9), (9, 8), (8, 7), (7, 0),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
    (0, 4), (4, 5), (5, 6),
    (0, 1), (1, 2), (2, 3)
]


# Load one mmWave .bin file (float64 Nx5)
def load_bin(path):
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=np.float64)

    data = np.fromfile(path, dtype=np.float64)
    data = data[: (data.size // 5) * 5]
    return data.reshape(-1, 5)

# Load all mmWave frames in a sequence and compute global anchor
def load_mmwave_sequence(action_path):
    mmwave_path = os.path.join(action_path, "mmwave")

    frame_files = sorted(
        [f for f in os.listdir(mmwave_path) if f.endswith(".bin")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    frames = [load_bin(os.path.join(mmwave_path, f)) for f in frame_files]

    # Global anchor (median of all xyz)
    all_xyz = np.concatenate([f[:, :3] for f in frames if f.shape[0] > 0], axis=0)
    anchor = np.median(all_xyz, axis=0)

    return frames, anchor


def generate_predictions(action_path):

    parts = os.path.normpath(action_path).split(os.sep)
    action = parts[-1]
    subject = parts[-2]
    environment = parts[-3]
    data_root = action_path.split("MMFi")[0] + "MMFi"

    print(f"Detected → Environment: {environment}, Subject: {subject}, Action: {action}")

    model = MMWavePoseTransformer(num_joints=17).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = MMFi_mmWave_Dataset(
        data_root,
        subjects=[subject],
        actions=[action]
    )

    preds, gts = [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            points, gt = dataset[i]
            points = points.unsqueeze(0).to(DEVICE)

            pred = model(points).squeeze(0).cpu().numpy()
            gt = gt.numpy()

            # Root alignment
            pred -= pred[0]
            gt -= gt[0]

            # Flip vertical axis (fix upside-down)
            pred[:, 1] *= -1
            gt[:, 1] *= -1

            preds.append(pred)
            gts.append(gt)

    return np.array(preds), np.array(gts)

# Animation
def animate(mm_frames, anchor, preds, gts, save_path="mmwave_pose_triplet.gif"):

    num_frames = min(len(mm_frames), len(preds))

    fig = plt.figure(figsize=(18, 6))

    ax_gt   = fig.add_subplot(131, projection='3d')
    ax_mm   = fig.add_subplot(132, projection='3d')
    ax_pred = fig.add_subplot(133, projection='3d')

    ax_gt.set_title("Ground Truth Skeleton")
    ax_mm.set_title("mmWave Radar Point Cloud")
    ax_pred.set_title("Model Prediction")

    for ax in [ax_gt, ax_mm, ax_pred]:
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("X (Lateral)")
        ax.set_ylabel("Y (Vertical)")
        ax.set_zlabel("Z (Depth)")
        ax.view_init(elev=15, azim=45)

    gt_lines   = [ax_gt.plot([], [], [], 'b-')[0] for _ in CONNECTIVITY]
    pred_lines = [ax_pred.plot([], [], [], 'r-')[0] for _ in CONNECTIVITY]
    mm_scatter = ax_mm.scatter([], [], [], s=8, c=[], cmap='jet')

    def update(frame):

        gt_frame = gts[frame].copy()
        pred_frame = preds[frame].copy()

        gt_frame[:, [1, 2]] = gt_frame[:, [2, 1]]
        pred_frame[:, [1, 2]] = pred_frame[:, [2, 1]]

        gt_frame[:, 0] *= -1
        pred_frame[:, 0] *= -1

        for i, (j1, j2) in enumerate(CONNECTIVITY):
            gt_lines[i].set_data(
                [gt_frame[j1, 0], gt_frame[j2, 0]],
                [gt_frame[j1, 1], gt_frame[j2, 1]]
            )
            gt_lines[i].set_3d_properties(
                [gt_frame[j1, 2], gt_frame[j2, 2]]
            )

        for i, (j1, j2) in enumerate(CONNECTIVITY):
            pred_lines[i].set_data(
                [pred_frame[j1, 0], pred_frame[j2, 0]],
                [pred_frame[j1, 1], pred_frame[j2, 1]]
            )
            pred_lines[i].set_3d_properties(
                [pred_frame[j1, 2], pred_frame[j2, 2]]
            )


        pts = mm_frames[frame]

        if pts.shape[0] > 0:
            xyz = pts[:, :3] - anchor

 
            mask = (
                (xyz[:, 0] >= -1.5) & (xyz[:, 0] <= 1.5) &
                (xyz[:, 1] >= -1.5) & (xyz[:, 1] <= 1.5) &
                (xyz[:, 2] >= -1.5) & (xyz[:, 2] <= 1.5)
            )

            xyz = xyz[mask]
            intensity = pts[mask, 3]

            mm_scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            mm_scatter.set_array(intensity)
        else:
            mm_scatter._offsets3d = ([], [], [])

        fig.suptitle(f"Frame {frame+1}/{num_frames}", fontsize=14)

        return gt_lines + pred_lines + [mm_scatter]

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    ani.save(save_path, writer=writer)
    plt.close()
    print("Saved:", save_path)


if __name__ == "__main__":

    print("Loading mmWave frames...")
    mm_frames, anchor = load_mmwave_sequence(ACTION_PATH)

    print("Generating skeleton predictions...")
    preds, gts = generate_predictions(ACTION_PATH)

    print("Rendering final visualization...")
    animate(mm_frames, anchor, preds, gts, save_path="mmwave_pose_triplet.gif")
