import shutil
import argparse
from pathlib import Path


def main(root_data_path: Path):
    """
    Iterates over subdirectories in a given root path and removes specified
    directories ('point_cloud', 'rgb', 'depth') from within them.
    """
    # Define the root directory containing the subdirectories to process.
    # Please change this to your target directory.
    root_data_path = Path(root_data_path)
    if not root_data_path.exists():
        print(f"Directory {root_data_path} not found")
        return

    print(f"This will permanently delete processed directories in: {root_data_path}")

    if input("Are you sure? (y/n): ").lower() != "y":
        print("Operation cancelled.")
        return

    # Define the names of the directories to be removed from each subdirectory.
    dirs_to_remove = ["point_cloud", "rgb", "depth"]

    if not root_data_path.is_dir():
        print(f"Error: The specified directory does not exist: {root_data_path}")
        return

    print(f"Scanning subdirectories in: {root_data_path}")

    # Iterate over all items in the root directory.
    for subdir in root_data_path.iterdir():
        if subdir.is_dir():
            print(f"Processing subdirectory: {subdir.name}")
            for dir_name in dirs_to_remove:
                target_dir = subdir / dir_name
                if target_dir.is_dir():
                    print(f"  - Removing: {target_dir}")
                    shutil.rmtree(target_dir)

    print("\nCleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the root directory")
    args = parser.parse_args()
    root_data_path = Path(args.path)
    main(root_data_path)
