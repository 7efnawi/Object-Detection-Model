import json
import os
from tqdm import tqdm

ROOT = r"D:\NCT\NCT-2\S2\Capston\DataSets\COCO" 
SPLIT = "val2017" 
JSON_FILE = os.path.join(ROOT, "annotations", f"instances_{SPLIT}.json")
IMAGES_DIR = os.path.join(ROOT, "images", SPLIT)
LABELS_DIR = os.path.join(ROOT, "labels", SPLIT)  # Directory for labels to be created

os.makedirs(LABELS_DIR, exist_ok=True)

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO format bbox: [x_min, y_min, width, height]
    to YOLO format: [x_center, y_center, w, h] (in percentages)
    """
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    # Convert to percentages:
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

def main():
    # Read the JSON file
    with open(JSON_FILE, 'r') as f:
        coco_data = json.load(f)

    # COCO categories: Create a mapping from category_id to index (0-based)
    categories = coco_data["categories"]
    cat2idx = {}
    for i, c in enumerate(categories):
        cat_id = c["id"]      # COCO category ID
        cat2idx[cat_id] = i   # Map to sequential indices (0 to 79, for example, if there are 80 categories)

    # Images
    images = {img["id"]: img for img in coco_data["images"]}

    # Annotations
    annotations = coco_data["annotations"]

    # Create a label text file for each image
    for ann in tqdm(annotations, desc=f"Converting {SPLIT}"):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox = ann["bbox"]  # COCO format: x_min, y_min, width, height

        # Skip certain cases if present (e.g., segmented objects or unwanted categories)
        if ann.get("iscrowd", 0) == 1:
            continue

        # Image information
        img_info = images[img_id]
        img_width = img_info["width"]
        img_height = img_info["height"]
        file_name = img_info["file_name"]

        # Convert bbox to YOLO format
        x_center, y_center, w, h = convert_bbox_coco_to_yolo(bbox, img_width, img_height)

        # Class index (0-based)
        class_idx = cat2idx[cat_id]

        # Label file name
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_file)

        # Open the label file and append a new line (or create it if it doesn't exist)
        with open(label_path, 'a') as lf:
            lf.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    main()
