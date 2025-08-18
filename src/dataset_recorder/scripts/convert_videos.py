#!/usr/bin/env python3
import subprocess
import argparse
import sys
from pathlib import Path

def create_videos(root_dir: Path, framerate: int, crf: int, preset: str):
    """
    Iterates over subdirectories in a root directory and uses ffmpeg to create
    videos from sequences of PNG images for both 'left' and 'right' cameras.

    Args:
        root_dir (Path): The path to the root directory containing the scene subdirectories.
        framerate (int): The framerate for the output video.
        crf (int): The Constant Rate Factor for x264 encoding (lower is higher quality).
        preset (str): The encoding speed vs. compression preset for x264.
    """
    root_path = Path(root_dir)
    print(f"Starting video creation process in root directory: {root_path}")
    print(f"Using settings: Framerate={framerate}, CRF={crf}, Preset='{preset}'")
    print("-" * 50)

    # Check if the root directory exists
    if not root_path.is_dir():
        print(f"Error: Root directory not found at '{root_path}'")
        return

    # Iterate through each item in the root directory
    for item_path in sorted(root_path.iterdir()):
        # Process only if it's a directory
        if not item_path.is_dir():
            continue

        print(f"Processing directory: {item_path.name}")
        input_dir = item_path / "rgb"

        # Check if the 'rgb' subdirectory exists before processing cameras
        if not input_dir.is_dir():
            print(f"  -> Skipping '{item_path.name}': 'rgb' subdirectory not found.")
            print("-" * 20)
            continue
        
        # Process each camera view
        for view in ["left", "right"]:
            input_pattern = input_dir / f"%06d_{view}.png"
            output_file = item_path / f"{view}.mp4"
            if output_file.exists():
                print(f"  -> Skipping '{view}' camera video, already exists.")
                continue

            print(f"  -> Processing '{view}' camera video...")

            # A simple check to see if there are any matching files to process
            # This avoids ffmpeg errors on empty directories
            first_image = input_dir / f"000000_{view}.png"
            if not first_image.exists():
                print(f"     -> Skipping: No images found for '{view}' view (e.g., {first_image.name}).")
                continue

            # --- Optimized FFmpeg Command ---
            # -hide_banner: Suppresses printing the ffmpeg banner.
            # -y: Overwrites the output file without asking.
            # -framerate: Sets the input framerate. Placed before -i for correctness.
            # -i: Specifies the input file pattern.
            # -c:v libx264: Selects the H.264 video codec.
            # -preset: Trades encoding speed for compression efficiency. 'medium' is a good default.
            # -crf: Constant Rate Factor. Controls quality/file size. 18-28 is a sane range.
            # -pix_fmt yuv420p: Crucial for compatibility with most video players and web browsers.
            command = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-framerate", str(framerate),
                "-i", str(input_pattern),  # Convert Path to string for subprocess
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                str(output_file), # Convert Path to string for subprocess
            ]

            try:
                # Execute the command
                # We use capture_output=True and text=True to show ffmpeg's output only on error
                result = subprocess.run(
                    command, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8'
                )
                print(f"     -> ✔ Success! Video saved to: {output_file}")

            except FileNotFoundError:
                print("Error: 'ffmpeg' command not found.")
                print("Please ensure FFmpeg is installed and in your system's PATH.")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                # This block will run if ffmpeg returns a non-zero exit code (i.e., an error)
                print(f"     -> ❌ Error processing '{view}' view in '{item_path.name}'.")
                print("     -> FFmpeg command failed.")
                print("     -> FFmpeg stderr output:")
                # Indent the error output for better readability
                for line in e.stderr.splitlines():
                    print(f"       {line}")
        
        print("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch create videos from image sequences for left/right cameras in subdirectories.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "root_dir",
        help="The root directory containing subdirectories of image frames."
    )
    parser.add_argument(
        "-r", "--framerate",
        type=int,
        default=10,
        help="Framerate of the output video (default: 10)."
    )
    parser.add_argument(
        "-c", "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor for libx264 (quality). Lower is better quality. (default: 23)."
    )
    parser.add_argument(
        "-p", "--preset",
        type=str,
        default="medium",
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
        help="Encoding speed/compression preset for libx264 (default: medium)."
    )

    args = parser.parse_args()

    # root_dir = Path(args.root_dir)
    # root_dirs = list(root_dir.iterdir())
    # for n, exp_dir in enumerate(root_dirs):
    #     print(f"[{n+1}/{len(root_dirs)}] Processing: {exp_dir}")
    #     create_videos(exp_dir, args.framerate, args.crf, args.preset)

    create_videos(args.root_dir, args.framerate, args.crf, args.preset)
