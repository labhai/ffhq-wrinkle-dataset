import os
import shutil
import sys
from tqdm import tqdm

def copy_matching_files(ffhq_images, manual_masks, face_images):
    os.makedirs(face_images, exist_ok=True)
    files_in_manual_masks = set(os.listdir(manual_masks))
    matching_files = []
    for root, _, files in os.walk(ffhq_images):
        for file in files:
            if file in files_in_manual_masks:
                file_name_without_ext = os.path.splitext(file)[0]
                try:
                    file_number = int(file_name_without_ext)
                    if 0 <= file_number < 1000:
                        continue 
                except ValueError:
                    pass
                matching_files.append((root, file))
    with tqdm(total=len(matching_files), desc="Copying files") as pbar:
        for root, file in matching_files:
            file_path = os.path.join(root, file)
            dest_path = os.path.join(face_images, file)
            shutil.copy2(file_path, dest_path)
            pbar.update(1)
    print("All matching files have been copied.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python png_parsing.py <ffhq_images> <manual_masks> <face_images>")
        sys.exit(1)
    ffhq_images = sys.argv[1]
    manual_masks = sys.argv[2]
    face_images = sys.argv[3]
    copy_matching_files(ffhq_images, manual_masks, face_images)
