"""
This script provides a naive baseline for FIFA Skeletal Tracking Challenge.

Author: Tianjian Jiang
Date: Nov 10, 2025
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from lib.camera_tracker import CameraTracker, CameraTrackerOptions
from lib.postprocess import smoothen


OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]

# Indices of foot keypoints in our 15-keypoint format:
# 11=RAnkle, 12=LAnkle, 13=RBigToe, 14=LBigToe
FOOT_INDICES = [11, 12, 13, 14]


def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray

    returns:
        intersection: (3,) intersection point
    """
    # solve the x and y where z = 0
    t = -o[2] / d[2]
    return o + t * d


def ray_from_xy(xy, K, R, t, k1=0.0, k2=0.0):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for radial distortion using coefficients k1 and k2.

    Args:
        xy: (2,) array_like containing pixel coordinates [x, y] in the image.
        K: (3, 3) ndarray representing the camera intrinsic matrix.
        R: (3, 3) ndarray representing the camera rotation matrix.
        t: (3,) ndarray representing the camera translation vector.
        k1: float, the first radial distortion coefficient (default 0).
        k2: float, the second radial distortion coefficient (default 0).

    Returns:
        origin: (3,) ndarray representing the camera center in world coordinates.
        direction: (3,) unit ndarray representing the direction of the ray in world coordinates.
    """
    # Convert the pixel coordinate to homogeneous coordinates.
    p = np.array([xy[0], xy[1], 1.0])

    # Compute the normalized coordinate (distorted) in the camera coordinate system.
    p_norm = np.linalg.inv(K) @ p  # p_norm = [x_d, y_d, 1]
    x_d, y_d = p_norm[0], p_norm[1]

    # Compute the radial distance (squared) in the normalized plane.
    r2 = x_d**2 + y_d**2
    # Compute the distortion factor.
    factor = 1 + k1 * r2 + k2 * (r2**2)

    # Correct the distorted normalized coordinates.
    x_undist = x_d / factor
    y_undist = y_d / factor

    # Construct the undistorted direction in camera coordinates (z = 1).
    d_cam = np.array([x_undist, y_undist, 1.0])

    # Transform the direction to world coordinates.
    direction = R.T @ d_cam
    direction = direction / np.linalg.norm(direction)

    # The camera center in world coordinates is given by -R^T t.
    origin = -R.T @ t
    return origin, direction


def project_points_th(obj_pts, R, C, K, k):
    """Projects 3D points onto 2D image plane using camera intrinsics and distortion.

    args:
        obj_pts: (N, 3) - 3D points in world space
        R: (3, 3) - Rotation matrix
        C: (3,) - Camera center
        K: (3, 3) - Camera intrinsic matrix
        k: (5,) - Distortion coefficients

    returns:
        img_pts: (N, 2) - Projected 2D points
    """

    # Transform world points to camera coordinates
    pts_c = (R @ ((obj_pts - C).unsqueeze(-1))).squeeze(-1)

    # Normalize to get image plane coordinates
    img_pts = pts_c[:, :2] / pts_c[:, 2:]

    # Compute radial distortion
    r2 = (img_pts**2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(r2, 0, 0.5 / min(max(torch.abs(k).max().item(), 1.0), 1.0))
    p = torch.arange(1, k.shape[-1] + 1, device=k.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k * r2.pow(p)).sum(-1, keepdim=True))

    # Apply intrinsics K
    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)  # Homogeneous coords
    img_pts = (K @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]  # Convert back to 2D

    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, R, C, K, k, group_ids, n_groups, iterations=10):
    """
    Optimize per-(person, frame) translation to minimize reprojection error across all joints.

    args:
        pts_3d: (P, 3)    - 3D joint positions (P = total valid joint samples across all groups)
        pts_2d: (P, 2)    - Corresponding 2D observations
        R: (P, 3, 3)      - Rotation matrix per sample (fixed)
        C: (P, 3)         - Camera center per sample (fixed)
        K: (P, 3, 3)      - Camera intrinsic matrix per sample (fixed)
        k: (P, 2)         - Distortion coefficients per sample (fixed)
        group_ids: (P,)   - Which (person, frame) group each sample belongs to [0, n_groups)
        n_groups: int     - Number of (person, frame) groups (= number of valid person/frame pairs)
        iterations: int   - Number of optimization steps

    returns:
        t: (n_groups, 3) - Optimized translation per (person, frame) group
    """
    t = torch.nn.Parameter(torch.zeros(n_groups, 3, dtype=pts_3d.dtype, device=pts_3d.device).requires_grad_(True))
    offset = torch.tensor([3, 3, 0.2], dtype=pts_3d.dtype, device=pts_3d.device)

    assert not torch.isnan(pts_3d).any()
    assert not torch.isnan(pts_2d).any()

    def closure():
        optimizer.zero_grad()
        # broadcast each group's translation to all of its joint samples
        t_per_sample = t[group_ids]  # (P, 3)
        projected_pts = project_points_th(pts_3d + t_per_sample, R, C, K, k)
        loss = torch.nn.functional.mse_loss(projected_pts, pts_2d)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([t], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            t.copy_(torch.clamp(t, -offset, offset))

    return t.detach()


def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes):
    """Fine-tune per-(person, frame) translation by minimizing reprojection error across all valid joints."""
    NUM_PERSONS, NUM_FRAMES = predictions.shape[:2]

    R_arr = np.array([k[0] for k in Rt])   # (NUM_FRAMES, 3, 3)
    t_arr = np.array([k[1] for k in Rt])   # (NUM_FRAMES, 3)
    C_arr = (-t_arr[:, None] @ R_arr).squeeze(1)  # (NUM_FRAMES, 3)

    # valid: (NUM_PERSONS, NUM_FRAMES) — pairs where a bounding box exists
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)

    # skels_2d is (NUM_FRAMES, NUM_PERSONS, 15, 2) — transpose to (NUM_PERSONS, NUM_FRAMES, 15, 2)
    skels_2d_pf = skels_2d.transpose(1, 0, 2, 3)

    # For each valid (person, frame) pair, gather all joints that have non-NaN 2D observations
    kps_3d_all = predictions[valid]   # (M, 15, 3)
    kps_2d_all = skels_2d_pf[valid]   # (M, 15, 2)

    valid_persons, valid_frames = np.where(valid)  # (M,)
    M = len(valid_persons)

    # valid_joints[m, j] = True if joint j is usable for pair m
    valid_joints = (
        ~np.isnan(kps_2d_all).any(axis=-1) &
        ~np.isnan(kps_3d_all).any(axis=-1)
    )  # (M, 15)

    pair_ids, joint_ids = np.where(valid_joints)  # (P,) each — index into the M valid pairs

    pts_3d = kps_3d_all[pair_ids, joint_ids]          # (P, 3)
    pts_2d = kps_2d_all[pair_ids, joint_ids]           # (P, 2)
    frame_ids = valid_frames[pair_ids]                 # (P,) — frame index for each sample

    cam_R = R_arr[frame_ids]                           # (P, 3, 3)
    cam_C = C_arr[frame_ids]                           # (P, 3)
    cam_K = cameras["K"][frame_ids]                    # (P, 3, 3)
    cam_k = cameras["k"][frame_ids, :2]                # (P, 2)

    traj_3d = minimize_reprojection_error(
        pts_3d=torch.tensor(pts_3d, dtype=torch.float32).to("cuda"),
        pts_2d=torch.tensor(pts_2d, dtype=torch.float32).to("cuda"),
        R=torch.tensor(cam_R, dtype=torch.float32).to("cuda"),
        C=torch.tensor(cam_C, dtype=torch.float32).to("cuda"),
        K=torch.tensor(cam_K, dtype=torch.float32).to("cuda"),
        k=torch.tensor(cam_k, dtype=torch.float32).to("cuda"),
        group_ids=torch.tensor(pair_ids, dtype=torch.long).to("cuda"),
        n_groups=M,
    )
    return traj_3d, valid


def process_sequence(
    boxes: np.ndarray,
    cameras: dict,
    skels_3d: np.ndarray,
    skels_2d: np.ndarray,
    video_path: Path | str,
    tracker_options: CameraTrackerOptions,
) -> np.ndarray:
    """a naive baseline that uses the bounding boxes to estimate the camera pose
    1. estimate the camera pose using the bounding boxes
    2. periodically refine the camera pose using lane lines
    3. project the 3D skeletons to the 2D image plane and optimize the translation to minimize reprojection error
    """
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    predictions = np.zeros((NUM_PERSONS, NUM_FRAMES, 15, 3))
    predictions.fill(np.nan)
    pitch_points = np.loadtxt("data/pitch_points.txt")

    video = cv2.VideoCapture(video_path)
    camera_tracker = CameraTracker(
        pitch_points=pitch_points,
        fps=50.0,
        options=tracker_options,
    )
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],
        t=cameras["t"][0],
    )

    Rt = []
    for frame_idx in (pbar := tqdm(range(NUM_FRAMES), desc=f"{video_path.stem}")):
        success, img = video.read()
        if not success:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            break

        state = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )
        yaw, pitch, roll = state.get_ypr()
        pbar.set_postfix_str(f"yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
        Rt.append((state.R.copy(), state.t.copy()))

        for person in range(NUM_PERSONS):
            # decide which foot is in contact with the ground by checking which has lower y
            box = boxes[frame_idx, person]
            if np.isnan(box).any():
                continue

            skel_2d = skels_2d[frame_idx, person]

            # Prefer ankle/toe keypoints for ground contact over the generic lowest pixel
            foot_kps = skel_2d[FOOT_INDICES]
            valid_foot = ~np.isnan(foot_kps).any(axis=1)
            if valid_foot.any():
                candidates = np.array(FOOT_INDICES)[valid_foot]
                IDX = candidates[np.argmax(skel_2d[candidates, 1])]
            else:
                IDX = np.argmax(skel_2d[:, 1])
            x, y = skel_2d[IDX]
            K = cameras["K"][frame_idx]
            k = cameras["k"][frame_idx]
            R, t = Rt[-1]
            o, d = ray_from_xy((x, y), K, R, t, k[0], k[1])
            intersection = intersection_over_plane(o, d)

            # convert from camera space to world space
            skel_3d = skels_3d[frame_idx, person]
            skel_3d = skel_3d @ R
            skel_3d = skel_3d - skel_3d[IDX] + intersection
            predictions[person, frame_idx] = skel_3d

    # fine-tune the translation to minimize reprojection error
    traj_3d, valid = fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes)
    predictions[valid] = predictions[valid] + traj_3d.cpu().numpy()[:, None, :]
    for person in range(NUM_PERSONS):
        predictions[person] = smoothen(predictions[person])
    
    # update the camera parameters
    cameras["R"] = np.array([k[0] for k in Rt], dtype=np.float32)
    cameras["t"] = np.array([k[1] for k in Rt], dtype=np.float32)
    return predictions.astype(np.float32)


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def main(
    sequences: list[str],
    output: Path | str,
    max_refine_interval: int,
    export_camera: bool,
    visualize: bool,
):
    debug_stages = ["projection", "flow", "mask"] if visualize else []
    if export_camera:
        camera_dir = Path("outputs/calibration/")
        camera_dir.mkdir(parents=True, exist_ok=True)
    else:
        camera_dir = None

    root = Path("data/")
    solutions = {}
    for sequence in sequences:
        camera = dict(np.load(root / "cameras" / f"{sequence}.npz"))
        skel2d = np.load(root / "skel_2d" / f"{sequence}.npy")
        skel3d = np.load(root / "skel_3d" / f"{sequence}.npy")
        boxes = np.load(root / "boxes" / f"{sequence}.npy")
        video_path = root / "videos" / f"{sequence}.mp4"

        NUM_FRAMES = boxes.shape[0]
        solutions[sequence] = process_sequence(
            cameras=camera,
            boxes=boxes,
            skels_2d=skel2d[:, :, OPENPOSE_TO_OURS],
            skels_3d=skel3d[:, :, OPENPOSE_TO_OURS],
            video_path=video_path,
            tracker_options=CameraTrackerOptions(
                refine_interval=np.clip(NUM_FRAMES // 500, a_min=1, a_max=max_refine_interval),
                debug_stages=tuple(debug_stages),
            ),
        )

        if export_camera:
            camera_path = camera_dir / f"{sequence}.npz"
            np.savez(camera_path, **camera)

    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **solutions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", "-s", type=str, default="data/sequences_full.txt", help="Path to the sequences file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default="output/submission_full.npz", help="Path to the output npz file"
    )
    parser.add_argument("--refine_interval", "-r", type=int, default=1, help="Interval to refine the camera pose")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the tracking results")
    parser.add_argument("--export_camera", "-c", action="store_true", help="Export the camera parameters")
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    main(sequences, args.output, args.refine_interval, args.export_camera, args.visualize)
