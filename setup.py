"""
Setup script for the FIFA Skeletal Tracking Starter Kit.

Steps:
  1. Install pip dependencies
  2. Download dataset from HuggingFace
  3. Extract video frames into data/images/<sequence>/
"""

import subprocess
import sys
from pathlib import Path


HF_REPO = "tijiang13/FIFA-Skeletal-Tracking-Light-2026"
DATA_DIR = Path("data")

# Folders to download from the HuggingFace repo
HF_FOLDERS = ["cameras", "boxes", "skel_2d", "skel_3d", "videos"]

PIP_PACKAGES = [
    "numpy",
    "torch",
    "opencv-python",
    "tqdm",
    "scipy",
    "Pillow",
    "huggingface_hub",
]


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def install_dependencies() -> None:
    print("\n=== Installing pip dependencies ===")
    run([sys.executable, "-m", "pip", "install", "--upgrade"] + PIP_PACKAGES)


def download_dataset() -> None:
    print("\n=== Downloading dataset from HuggingFace ===")
    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=[f"{folder}/*" for folder in HF_FOLDERS],
        ignore_patterns=["*.git*"],
    )
    print(f"Dataset downloaded to {DATA_DIR.resolve()}")


def extract_frames() -> None:
    print("\n=== Extracting video frames ===")
    video_dir = DATA_DIR / "videos"
    images_dir = DATA_DIR / "images"

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print("No .mp4 files found in data/videos/ — skipping frame extraction.")
        return

    for video_path in videos:
        sequence = video_path.stem
        output_folder = images_dir / sequence
        if output_folder.exists() and any(output_folder.iterdir()):
            print(f"  Skipping {sequence} (frames already exist)")
            continue
        print(f"  Extracting frames for {sequence} ...")
        run([
            sys.executable, "video2image.py",
            "--video_path", str(video_path),
            "--output_folder", str(output_folder),
        ])


def main() -> None:
    install_dependencies()
    download_dataset()
    extract_frames()
    print("\n=== Setup complete! ===")
    print("Run the baseline with:")
    print("  python main.py -s data/sequences_full.txt -o outputs/submission_full.npz -c")


if __name__ == "__main__":
    main()
