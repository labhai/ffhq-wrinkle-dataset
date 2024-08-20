#!/bin/bash

declare -A files=(
    ["https://drive.google.com/uc?id=1vzbx847hq1hgKjMtMmxLzmzLrYb4t0QX"]="manual_wrinkle_masks.zip"
    ["https://drive.google.com/uc?id=1dLl7_ChsJNrZSQIbNGj21sm45hmH0ZxC"]="weak_wrinkle_masks.zip"
)

base_folder="./data" # edit to your desired path

if [ ! -d "$base_folder" ]; then
    mkdir -p "$base_folder"
fi

for url in "${!files[@]}"; do
    zip_file="${files[$url]}"
    gdown -O "$base_folder/$zip_file" "$url"
done

for zip_file in "${files[@]}"; do
    extract_path="$base_folder/$zip_file"
    unzip "$extract_path" -d "$base_folder"
    rm "$extract_path"
    echo "Extracted and removed $zip_file in $extract_path"
done
