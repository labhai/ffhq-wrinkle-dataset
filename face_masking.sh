#!/bin/bash

declare -A files=(
    ["https://drive.google.com/uc?id=1Oje4gsfgsScBfLMwP-OobgoFK-DHOGqc"]="face_parsed_labels.zip"
)

base_folder="./data" # edit to your desired path
etcs_folder="$base_folder/etcs"

if [ ! -d "$base_folder" ]; then
    mkdir -p "$base_folder"
fi

if [ ! -d "$etcs_folder" ]; then
    mkdir -p "$etcs_folder"
fi

echo "Starting PNG parsing..."
if python ./png_parsing.py "$base_folder/images1024x1024" "$base_folder/manual_wrinkle_masks" "$base_folder/face_images"; then
  echo "PNG parsing finished"
else
  echo "PNG parsing failed"
  exit 1
fi

echo "Download files..."
for url in "${!files[@]}"; do
    zip_file="${files[$url]}"
    gdown -O "$etcs_folder/$zip_file" "$url"
done

for zip_file in "${files[@]}"; do
    extract_folder="${zip_file%.zip}"
    extract_path="$etcs_folder/$zip_file"
    unzip "$extract_path" -d "$etcs_folder/$extract_folder"
    rm "$extract_path"
    echo "Extracted and removed $zip_file into $etcs_folder/$extract_folder"
done

echo "Starting face masking..."
if python ./face_masking.py "$etcs_folder/face_parsed_labels" "$base_folder/face_images" "$base_folder/masked_face_images"; then
  echo "Face masking finished"
else
  echo "Face masking failed"
  exit 1
fi
