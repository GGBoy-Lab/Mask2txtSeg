🧾 README - YOLO Segmentation Label Converter
This script converts RGB-colored segmentation mask images into YOLOv8-compatible .txt label files. It extracts contours from masks and writes them in normalized polygon format required by YOLO for instance segmentation tasks.

📁 Project Overview
Purpose: Convert RGB segmentation masks to YOLO label format.
Input:
RGB-colored segmentation mask images (.png)
Corresponding image files
Output:
.txt label files with normalized polygon coordinates
Supports: Multiple object classes via custom color-to-class mapping

⚙️ Requirements
Install the following dependencies:
pip install opencv-python numpy pathlib

🗂️ Directory Structure
Ensure your dataset follows this structure:

project/
├── data/
│   └── VOC/
│       ├── JPEGImages/          # Original image files (.jpg, .png)
│       └── SegmentationClass/   # RGB segmentation mask images (.png)
├── mask2txt_custom.py           # This conversion script
└── labels/                      # Output directory for label files (will be created automatically)

🧪 How It Works
Reads RGB segmentation mask as a NumPy array
Maps pixel values to class IDs using the defined color map
Extracts external contours for each class
Normalizes contour points based on image dimensions
Saves results in YOLO segmentation label format:
