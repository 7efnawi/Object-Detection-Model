# 📚 YOLOv5 Object Detection API Documentation

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*QOGcQM9G4dFAYJq-RK0YYg.png" alt="Object Detection Banner" width="800"/>
  <p><i>Comprehensive Technical Documentation | Version 1.0.0</i></p>
</div>

## 📋 Table of Contents

- [Introduction](#-introduction)
- [System Architecture](#-system-architecture)
- [Installation Guide](#-installation-guide)
- [Dataset Information](#-dataset-information)
- [API Reference](#-api-reference)
- [Development Guide](#-development-guide)
- [Deployment Guide](#-deployment-guide)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [Frequently Asked Questions](#-frequently-asked-questions)
- [Contributing](#-contributing)

<br>

## 🔍 Introduction

The YOLOv5 Object Detection API is a high-performance REST API for real-time object detection in images. Built on top of the state-of-the-art YOLOv5 model trained on the COCO dataset, this service provides accurate object detection with minimal latency.

### Key Features

- **Real-time Detection**: Process images and return results with minimal latency
- **High Accuracy**: Based on YOLOv5 with mAP@0.5 of ~0.54+
- **Scalable Architecture**: Designed to handle multiple concurrent requests
- **Background Model Loading**: Responsive API even during model initialization
- **Easy Deployment**: Optimized for Railway platform with Docker support
- **Comprehensive Error Handling**: Robust error reporting and fault tolerance

<br>

## 🏗 System Architecture

The system follows a modular architecture with the following components:

### High-Level Architecture

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│                │    │                │    │                │
│  HTTP Request  │───▶│  Flask Server  │───▶│  YOLOv5 Model  │
│                │    │                │    │                │
└────────────────┘    └────────────────┘    └────────────────┘
                              │                      │
                              ▼                      ▼
                      ┌────────────────┐    ┌────────────────┐
                      │                │    │                │
                      │  JSON Response │◀───│  Detection     │
                      │                │    │  Processing    │
                      └────────────────┘    └────────────────┘
```

### Components

1. **Flask Web Server**

   - Handles HTTP requests and responses
   - Manages concurrent connections
   - Provides health checks for the deployment platform

2. **Background Model Loader**

   - Uses threading to load the YOLOv5 model asynchronously
   - Enables the API to respond to requests during model initialization
   - Monitors model loading status and reports errors

3. **YOLOv5 Detection Engine**

   - Processes images using the pre-trained model
   - Converts model outputs to normalized coordinates
   - Maps class IDs to human-readable labels

4. **Response Handler**
   - Formats detection results as JSON
   - Provides standardized error responses
   - Includes metadata about the detection process

<br>

## 🛠️ Installation Guide

### Local Development Setup

#### Prerequisites

- Python 3.9 or newer
- Git
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional but recommended for faster inference)
- Access to the COCO dataset (located at `D:\NCT\NCT-2\S2\Capston\DataSets\COCO`)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/7efnawi/OD-Model.git
cd OD-Model
```

#### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Configure Dataset Path

Ensure the COCO dataset path is correctly referenced in your configuration files. The full path to the dataset is:

```
D:\NCT\NCT-2\S2\Capston\DataSets\COCO
```

If you're using a different location, update the dataset path in the `MY_coco30_yolov5.yaml` file.

#### Step 5: Run the Application

```bash
python app.py
```

The server will start on http://localhost:8000

### Docker Setup

#### Prerequisites

- Docker installed on your system
- Access to the COCO dataset (for training/retraining)

#### Step 1: Build the Docker Image

```bash
docker build -t yolov5-api .
```

#### Step 2: Run the Docker Container

```bash
docker run -p 8000:8000 yolov5-api
```

The server will be accessible at http://localhost:8000

<br>

## 📊 Dataset Information

### COCO Dataset Overview

This project uses the Common Objects in Context (COCO) dataset, which is a large-scale object detection, segmentation, and captioning dataset. The complete dataset used for training is located at:

```
D:\NCT\NCT-2\S2\Capston\DataSets\COCO
```

### Dataset Structure

The COCO dataset is structured as follows:

```
COCO/
├── annotations/           # JSON annotation files
│   ├── instances_train2017.json
│   └── instances_val2017.json
│
├── train2017/             # Training images (118K images)
│   └── [image files]
│
├── val2017/               # Validation images (5K images)
│   └── [image files]
│
└── labels/                # YOLO format labels
    ├── train2017/
    └── val2017/
```

### Dataset Usage

For this project, we use:

- Approximately 60% of the COCO dataset (balanced across classes)
- Images are processed at 640x640 resolution
- All 80 standard COCO classes are supported:
  - Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat...
  - Full list of classes is available in the `MY_coco30_yolov5.yaml` file

### Custom Dataset Configuration

The model was trained using a custom configuration specified in `MY_coco30_yolov5.yaml`, which defines:

- Dataset paths (relative to the project root)
- Number of classes (80)
- Names of each class
- Train/validation split

If you want to retrain the model on your own data, you'll need to modify this YAML file to point to your dataset location.

<br>

## 🔌 API Reference

### API Endpoints

#### 1. Main Endpoint

Provides information about the API and model status.

- **URL**: `/`
- **Method**: GET
- **Response Example**:

```json
{
  "message": "YOLOv5 Object Detection API",
  "model_status": "loaded",
  "error": null
}
```

- **Possible model_status values**:
  - `"loading"`: Model is currently being loaded
  - `"loaded"`: Model is ready for inference
  - `"failed"`: Model failed to load (error will contain details)

#### 2. Health Check

Used by deployment platforms to verify the service is running.

- **URL**: `/health`
- **Method**: GET
- **Response Example**:

```json
{
  "status": "ok"
}
```

#### 3. Object Detection

The main endpoint for detecting objects in images.

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:

  - `file`: The image file to analyze (required)

- **Success Response Example**:

```json
{
  "status": "success",
  "message": "Image processed successfully",
  "detections": [
    {
      "class": 0,
      "confidence": 0.85,
      "name": "person",
      "xmin": 120.5,
      "ymin": 220.3,
      "xmax": 250.8,
      "ymax": 380.1
    },
    {
      "class": 2,
      "confidence": 0.76,
      "name": "car",
      "xmin": 450.2,
      "ymin": 320.1,
      "xmax": 550.3,
      "ymax": 420.8
    }
  ],
  "count": 2
}
```

- **Error Response Examples**:

```json
{
  "error": "No file part"
}
```

```json
{
  "error": "Model is still loading"
}
```

```json
{
  "error": "Model failed to load: [error details]"
}
```

### API Usage Examples

#### Using cURL

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

#### Using Python with Requests

```python
import requests

url = "http://localhost:8000/predict"
image_path = "path/to/image.jpg"

with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())
```

#### Using JavaScript/Fetch API

```javascript
// Using FormData and fetch
const imageInput = document.getElementById("imageInput");
const formData = new FormData();
formData.append("file", imageInput.files[0]);

fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    console.log(data);
    // Process detection results
  })
  .catch((error) => console.error("Error:", error));
```

<br>

## 💻 Development Guide

### Project Structure

```
Project/
│
├── OD Model/                     # API Application
│   ├── app.py                    # Main Flask application
│   ├── test_predict.py           # Prediction test script
│   ├── Dockerfile                # Docker file for deployment
│   ├── MY_coco30_yolov5.yaml     # Custom YAML configuration
│   ├── best.pt                   # Trained model weights
│   ├── requirements.txt          # Python dependencies
│   ├── railway.json              # Railway deployment settings
│   ├── .gitignore                # List of ignored files
│   ├── README.md                 # Project README
│   └── yolov5/                   # Embedded YOLOv5 library
│       ├── models/               # YOLOv5 model architectures
│       ├── utils/                # Utility functions
│       └── data/                 # Test images and data configuration
│
└── DataSets/                     # Dataset Directory
    └── COCO/                     # COCO Dataset
        ├── annotations/          # JSON annotation files
        ├── train2017/            # Training images
        ├── val2017/              # Validation images
        └── labels/               # YOLO format labels
```

### Core Components

#### 1. Flask Application (app.py)

The main application file contains:

- API endpoint definitions
- Model loading logic
- Image processing pipeline
- Error handling mechanisms

#### 2. Model Loading Strategy

The model is loaded in a background thread to avoid blocking the application startup:

```python
@app.before_first_request
def before_first_request():
    global model_loading
    if not model and not model_loading:
        model_loading = True
        thread = threading.Thread(target=load_model_in_background)
        thread.daemon = True
        thread.start()
```

#### 3. Inference Pipeline

The inference pipeline follows these steps:

1. Image is loaded and converted to RGB
2. Image is resized to 640x640 (YOLOv5 input size)
3. Pixel values are normalized to [0-1]
4. Model performs inference on the image
5. Detections are processed and converted to pixel coordinates
6. Results are formatted as JSON and returned

### Training and Retraining

If you want to retrain the model using the COCO dataset:

1. Ensure the COCO dataset is available at `D:\NCT\NCT-2\S2\Capston\DataSets\COCO`
2. Update the dataset path in `MY_coco30_yolov5.yaml` if necessary
3. Run the training script:

```bash
python train.py --img 640 --batch 4 --epochs 50 \
  --data MY_coco30_yolov5.yaml \
  --weights yolov5s.pt --device 0
```

### Best Practices for Development

1. **Testing Changes**

   - Use the `test_predict.py` script to validate API functionality
   - Test with different image types (JPG, PNG) and sizes
   - Verify correct bounding box coordinates on test images

2. **Adding Features**

   - Follow the existing code style and structure
   - Document new endpoints or parameters thoroughly
   - Update the README.md with any changes to functionality

3. **Performance Optimization**
   - Profile the application to identify bottlenecks
   - Consider batch processing for multiple images
   - Optimize image pre-processing and post-processing steps

<br>

## 🚀 Deployment Guide

### Railway Deployment

Railway is the recommended platform for deploying this application due to its simplicity and performance.

#### 1. Prerequisites

- [Railway account](https://railway.app)
- Git repository with your project code
- A fork or clone of this repository

#### 2. Deployment Steps

1. **Connect Your Repository**

   - Sign in to Railway and create a new project
   - Select the "Deploy from GitHub repo" option
   - Connect your GitHub account and select your repository

2. **Configure Environment Variables** (if needed)

   - No environment variables are required by default
   - Optional variables can be set in the Railway dashboard

3. **Deploy the Application**

   - Railway will automatically detect the Dockerfile and build the application
   - The first deployment may take 5-10 minutes due to model download and setup

4. **Verify Deployment**
   - Once deployed, Railway will provide a public URL for your API
   - Test the API using the `/health` endpoint to verify it's running
   - Test the `/predict` endpoint with a sample image

#### 3. Understanding Railway Configuration (railway.json)

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "numReplicas": 1,
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 60,
    "startupTime": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

- **numReplicas**: Number of instances to run (1 is sufficient for most cases)
- **startCommand**: Command to start the application
- **healthcheckPath**: Endpoint used to verify the application is running
- **healthcheckTimeout**: Time in seconds before a health check fails
- **startupTime**: Allows 5 minutes for the application to start (accommodates model loading)
- **restartPolicy**: Automatically restarts the application on failure

### Other Deployment Options

#### Heroku Deployment

1. Create a `Procfile` with:

   ```
   web: python app.py
   ```

2. Push to Heroku:
   ```bash
   heroku create
   git push heroku main
   ```

#### AWS Elastic Beanstalk

1. Install the EB CLI:

   ```bash
   pip install awsebcli
   ```

2. Initialize and deploy:
   ```bash
   eb init
   eb create
   ```

<br>

## ⚡ Performance Optimization

### Model Optimization Techniques

#### 1. Quantization

You can quantize the model to reduce its size and increase inference speed:

```python
# Example of loading a quantized model
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
```

#### 2. Batch Processing

For processing multiple images, implement batch processing:

```python
def process_batch(images, batch_size=4):
    results = []

    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)

    return results
```

#### 3. GPU Acceleration

Enable GPU acceleration if available:

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### API Optimization

#### 1. Response Caching

Implement caching for frequently requested images:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_image_hash(image_bytes):
    # Generate a hash for the image
    return hashlib.md5(image_bytes).hexdigest()

@app.route('/predict', methods=['POST'])
def predict():
    # ... existing code ...

    # Check cache
    image_hash = get_image_hash(img_bytes)
    cached_result = cache.get(image_hash)

    if cached_result:
        return jsonify(cached_result)

    # ... process image and get results ...

    # Store in cache
    cache[image_hash] = result
    return jsonify(result)
```

#### 2. Asynchronous Processing

For large images or high traffic, implement asynchronous processing:

```python
# Using a task queue like Celery
@celery.task
def process_image_task(image_data):
    # Process image
    # ...
    return results

@app.route('/predict_async', methods=['POST'])
def predict_async():
    # ... code to get image ...

    # Submit task to queue
    task = process_image_task.delay(image_data)

    # Return task ID
    return jsonify({"task_id": task.id})

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = process_image_task.AsyncResult(task_id)

    if task.state == 'PENDING':
        return jsonify({"status": "processing"})
    elif task.state == 'SUCCESS':
        return jsonify({"status": "complete", "result": task.result})
    else:
        return jsonify({"status": "failed"})
```

<br>

## 🛑 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Failures

**Issue**: Model fails to load or takes too long to load

**Solutions**:

- Verify the model file (`best.pt`) exists in the correct location
- Check system memory - the model requires at least 2GB of available RAM
- Ensure all dependencies in `requirements.txt` are installed correctly
- Modify the model loading timeout in `railway.json` if deploying on Railway

#### 2. Import Errors

**Issue**: "No module named 'utils'" or similar import errors

**Solutions**:

- Check that the YOLOv5 directory structure is intact
- Ensure `__init__.py` files exist in all subdirectories
- Add the necessary paths to `sys.path` as shown in the code
- Verify that all dependencies are installed with the correct versions

#### 3. API Timeout Issues

**Issue**: API requests time out or take too long

**Solutions**:

- Reduce image resolution before sending to the API
- Optimize the model using techniques in the Performance section
- Increase the timeout settings in your deployment platform
- Consider deploying on a more powerful instance with more CPU/GPU resources

#### 4. Incorrect Detections

**Issue**: Detections are inaccurate or missing

**Solutions**:

- Verify the image format is supported (JPG, PNG, etc.)
- Check that the image is properly formatted and not corrupted
- Adjust the confidence threshold for detections
- Consider retraining the model on more relevant data

#### 5. Dataset Path Issues

**Issue**: Training scripts cannot find the COCO dataset

**Solutions**:

- Verify the dataset exists at `D:\NCT\NCT-2\S2\Capston\DataSets\COCO`
- Check that the dataset path in `MY_coco30_yolov5.yaml` is correctly set
- For Linux/macOS deployments, adjust the path separators accordingly
- If deploying in a container, ensure the dataset is properly mounted or included

### Debugging Tools

#### 1. Logging Configuration

Enable detailed logging by adding the following to `app.py`:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### 2. Model Inspection

Test your model directly with:

```python
# In a Python script
import torch
model = torch.load('best.pt', map_location='cpu')
print(model.names)  # Class names
print(model.stride)  # Model stride
print(model.pt_path)  # Model path
```

#### 3. API Testing Script

Use the provided `test_predict.py` script to test your API:

```bash
python test_predict.py
```

Modify the script to test different scenarios or endpoints.

<br>

## ❓ Frequently Asked Questions

#### Q: What types of objects can the model detect?

A: The model is trained on the COCO dataset and can detect 80 common object categories including people, vehicles, animals, furniture, and household items.

#### Q: What image formats are supported?

A: The API supports common image formats including JPEG, PNG, BMP, and GIF (first frame only).

#### Q: What is the maximum image size supported?

A: There is no hard limit on image size, but larger images will take longer to process. Images are resized to 640x640 pixels for processing, but the API returns coordinates mapped to the original image dimensions.

#### Q: Where is the COCO dataset located in this project?

A: The full COCO dataset used for training is located at `D:\NCT\NCT-2\S2\Capston\DataSets\COCO`.

#### Q: Can I use this API in a commercial application?

A: Yes, the project is licensed under the MIT License, which allows commercial use. However, be sure to check the licenses of all dependencies, especially the YOLOv5 model.

#### Q: How many requests can the API handle per second?

A: Performance depends on your deployment environment. On a standard Railway deployment, the API can handle approximately 10-20 requests per minute. For higher throughput, consider optimizing as described in the Performance section.

#### Q: Can I deploy this on a Raspberry Pi or other edge devices?

A: Yes, but you may need to optimize the model further. Consider using YOLOv5s or even smaller variants like YOLOv5n for edge deployment.

#### Q: How do I update the model with my own custom-trained weights?

A: Replace the `best.pt` file with your custom-trained weights. Ensure your model follows the same YOLOv5 architecture and update the class names in your code if they differ from COCO.

<br>

## 👥 Contributing

We welcome contributions to improve the YOLOv5 Object Detection API!

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug and steps to reproduce
2. **Suggest Features**: Open an issue describing the new feature
3. **Submit Pull Requests**: Implement bug fixes or new features

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style Guidelines

- Follow PEP 8 Python style guidelines
- Include docstrings for all functions and classes
- Add comments for complex code segments
- Write unit tests for new features

<br>

---

<div align="center">
  <p>
    <b>YOLOv5 Object Detection API</b><br>
    Developed as a Capstone Project - NCT 2025
  </p>
  <p>
    <a href="https://github.com/7efnawi/OD-Model">GitHub Repository</a> |
    <a href="https://github.com/7efnawi/OD-Model/issues">Report Issues</a>
  </p>
</div>
