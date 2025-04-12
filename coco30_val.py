import os
import random
import shutil

# Original source folder (contains images and labels)
src_train = r"D:/NCT/NCT-2/S2/Capston/DataSets/COCO/val2017"

# Destination folder for the subset
dst_train = r"D:/NCT/NCT-2/S2/Capston/DataSets/COCO/COCO_small/val2017"

# Create the destination folder if it doesn't exist
os.makedirs(dst_train, exist_ok=True)

# Subset ratio (30% of the data)
subset_ratio = 0.3

# Get list of all image files (extensions .jpg or .png)
image_files = [f for f in os.listdir(src_train) if f.lower().endswith(('.jpg', '.png'))]

# Calculate number of images to select
subset_count = int(len(image_files) * subset_ratio)
subset_files = random.sample(image_files, subset_count)
print(f"Selecting {subset_count} images out of {len(image_files)}")

# Copy images and corresponding label files
for file_name in subset_files:
    # Copy the image file
    src_image_path = os.path.join(src_train, file_name)
    dst_image_path = os.path.join(dst_train, file_name)
    shutil.copy2(src_image_path, dst_image_path)

    # Determine the corresponding label file (same name with .txt extension)
    base_name = os.path.splitext(file_name)[0]
    label_file = base_name + ".txt"
    src_label_path = os.path.join(src_train, label_file)
    dst_label_path = os.path.join(dst_train, label_file)

    # If label file exists, copy it; otherwise print a warning
    if os.path.exists(src_label_path):
        shutil.copy2(src_label_path, dst_label_path)
    else:
        print(f"Warning: Label file not found: {label_file}")

print("Subset (30%) created successfully!")
