# 📚 YOLOv5 Object Detection API Documentation

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*QOGcQM9G4dFAYJq-RK0YYg.png" alt="Object Detection Banner" width="800"/>
  
  <h2>🔍 Real-time Object Detection with YOLOv5</h2>
  
  <p><strong>Comprehensive Technical Documentation | Version 1.0.0</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python" alt="Python 3.9+"/>
    <img src="https://img.shields.io/badge/Framework-Flask-red.svg?style=for-the-badge&logo=flask" alt="Flask"/>
    <img src="https://img.shields.io/badge/Model-YOLOv5-brightgreen.svg?style=for-the-badge&logo=pytorch" alt="YOLOv5"/>
    <img src="https://img.shields.io/badge/Deployment-Railway-blueviolet.svg?style=for-the-badge&logo=railway" alt="Railway"/>
  </p>
  
  <hr style="width: 80%; border: 1px solid #ddd;">
</div>

## 📋 Table of Contents

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
  <div>
    <ul>
      <li><a href="#-introduction"><b>🔍 Introduction</b></a></li>
      <li><a href="#-system-architecture"><b>🏗 System Architecture</b></a></li>
      <li><a href="#-installation-guide"><b>🛠️ Installation Guide</b></a></li>
      <li><a href="#-dataset-information"><b>📊 Dataset Information</b></a></li>
      <li><a href="#-api-reference"><b>🔌 API Reference</b></a></li>
    </ul>
  </div>
  <div>
    <ul>
      <li><a href="#-development-guide"><b>💻 Development Guide</b></a></li>
      <li><a href="#-deployment-guide"><b>🚀 Deployment Guide</b></a></li>
      <li><a href="#-performance-optimization"><b>⚡ Performance Optimization</b></a></li>
      <li><a href="#-troubleshooting"><b>🛑 Troubleshooting</b></a></li>
      <li><a href="#-frequently-asked-questions"><b>❓ Frequently Asked Questions</b></a></li>
      <li><a href="#-contributing"><b>👥 Contributing</b></a></li>
    </ul>
  </div>
</div>

<br>

## 🔍 Introduction

<img align="right" src="https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg" width="350">

The **YOLOv5 Object Detection API** is a high-performance REST API for real-time object detection in images. Built on top of the state-of-the-art YOLOv5 model trained on the COCO dataset, this service provides accurate object detection with minimal latency.

This project delivers:

- A ready-to-use API for object detection
- A complete deployment pipeline
- Comprehensive documentation for developers
- Performance optimization guidelines

### ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h4>⚡️ Real-time Detection</h4>
      Process images and return results with minimal latency
    </td>
    <td width="50%">
      <h4>🧠 Advanced Model</h4>
      Based on YOLOv5 with mAP@0.5 of ~0.54+
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>📊 High Accuracy</h4>
      Precision ~0.63, Recall ~0.47+
    </td>
    <td width="50%">
      <h4>🌐 RESTful API</h4>
      Simple and fully documented endpoints
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>🔄 Background Model Loading</h4>
      Responsive API even during model initialization
    </td>
    <td width="50%">
      <h4>🚂 Easy Deployment</h4>
      Optimized for Railway platform with Docker support
    </td>
  </tr>
</table>

<br>

## 🏗 System Architecture

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNp1ksFugzAMhl_F8qnTpL1BL1MvPVXaYZcpyhKDUUlClFBtQrz7AqVsGtuJxP_n3_bvDFIVCBLkZ2NrUa_8h1ZGc6skzZXduPM3hbQxsKWj-jjE9w-jraUvdVK2hnbmCnacLy3RN6k0JfMhL8EFp2ftEcCFqKUmYLIAZuqSEOyYRIB2-TL8w5PtHyJXdlnQQnpXeKXrLJOuaWet1qyh_iP48VLtOvVVUqW9XzLvsP046TrEgTQNvdQm1R_qf-P1dBqNw-MNZ8bGH6NRHyNI0cArFcrSBQFZWsdkTjddSdpAxm4T27LmLTb0CbJ3VtWQx4vJ3TCOorspipw9Cs6d-OV-Zr8s13w7TReT8BZHURzN5tFsPvsBGS9mqQ" alt="System Architecture" width="800"/>
</div>

The system follows a modular architecture designed for scalability, reliability, and ease of maintenance.

### 🔄 Request Flow

1. **Client sends an HTTP request** with an image to analyze
2. **Flask Web Server** receives the request and validates the input
3. **YOLOv5 Model** processes the image and detects objects
4. **Processing logic** converts model output to standardized JSON format
5. **Response** is sent back to the client with detection results

### 🧩 Core Components

<table>
  <tr>
    <th width="20%" style="background-color: #4CAF50; color: white;">Component</th>
    <th style="background-color: #4CAF50; color: white;">Description</th>
    <th width="25%" style="background-color: #4CAF50; color: white;">Key Features</th>
  </tr>
  <tr>
    <td><b>🖥️ Flask Web Server</b></td>
    <td>Lightweight Python web framework that serves the API endpoints and handles HTTP requests/responses</td>
    <td>
      • RESTful endpoints<br>
      • Request validation<br>
      • Error handling
    </td>
  </tr>
  <tr>
    <td><b>⚙️ Background Model Loader</b></td>
    <td>Asynchronous component that loads the YOLOv5 model in a separate thread to prevent blocking the application startup</td>
    <td>
      • Non-blocking design<br>
      • Status monitoring<br>
      • Error reporting
    </td>
  </tr>
  <tr>
    <td><b>🧠 YOLOv5 Detection Engine</b></td>
    <td>State-of-the-art object detection model that processes images and identifies objects with their locations</td>
    <td>
      • Fast inference<br>
      • 80 object classes<br>
      • High accuracy
    </td>
  </tr>
  <tr>
    <td><b>📊 Response Handler</b></td>
    <td>Component that formats the model outputs into standardized JSON responses and handles different response scenarios</td>
    <td>
      • JSON formatting<br>
      • Error responses<br>
      • Metadata inclusion
    </td>
  </tr>
</table>

### 🔐 Security Design

The API implements several security measures:

- Input validation to prevent malicious file uploads
- Response sanitization to prevent data leakage
- Rate limiting to prevent DoS attacks
- Stateless design for horizontal scaling

<div class="note" style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0;">
  <b>💡 Note:</b> For high-traffic deployments, consider implementing a load balancer in front of multiple API instances to improve throughput and reliability.
</div>

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

<div align="center">
  <img src="https://cocodataset.org/images/coco-logo.png" height="100" alt="COCO Dataset Logo">
  <h3>Common Objects in Context (COCO) Dataset</h3>
</div>

This project uses the **COCO dataset**, a large-scale object detection, segmentation, and captioning dataset that has become the standard benchmark in computer vision tasks.

<div class="info-box" style="display: flex; margin-bottom: 20px; background-color: #f8f9fa; border-radius: 5px; overflow: hidden;">
  <div style="padding: 15px; background-color: #e9ecef; width: 30%;">
    <h4>📁 Dataset Location</h4>
    <p>The complete dataset used for training is located at:</p>
    <code>D:\NCT\NCT-2\S2\Capston\DataSets\COCO</code>
  </div>
  <div style="padding: 15px; width: 70%;">
    <h4>📈 Dataset Statistics</h4>
    <ul>
      <li><b>Training Images:</b> ~118,000 images</li>
      <li><b>Validation Images:</b> ~5,000 images</li>
      <li><b>Categories:</b> 80 object categories</li>
      <li><b>Annotations:</b> >200,000 labeled images</li>
      <li><b>Project Usage:</b> ~60% of COCO (balanced across classes)</li>
    </ul>
  </div>
</div>

### 📂 Dataset Structure

The COCO dataset follows a standardized structure:

```
COCO/
│
├── 📁 annotations/              # JSON annotation files
│   ├── instances_train2017.json
│   └── instances_val2017.json
│
├── 📁 train2017/                # Training images (118K images)
│   └── [image files]
│
├── 📁 val2017/                  # Validation images (5K images)
│   └── [image files]
│
└── 📁 labels/                   # YOLO format labels
    ├── train2017/
    └── val2017/
```

### 🏷️ Object Classes

The model detects 80 different object classes from the COCO dataset, organized into 12 super-categories:

<table>
  <tr>
    <th colspan="2" style="background-color: #4a5568; color: white;">Person</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Vehicle</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Outdoor</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Animal</th>
  </tr>
  <tr>
    <td>👤 person</td>
    <td></td>
    <td>🚗 car</td>
    <td>✈️ airplane</td>
    <td>🚦 traffic light</td>
    <td>🔥 fire hydrant</td>
    <td>🐱 cat</td>
    <td>🐶 dog</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>🚲 bicycle</td>
    <td>🚂 train</td>
    <td>🛑 stop sign</td>
    <td>⛽ parking meter</td>
    <td>🐴 horse</td>
    <td>🐑 sheep</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>🏍️ motorcycle</td>
    <td>🚌 bus</td>
    <td>🪑 bench</td>
    <td></td>
    <td>🐄 cow</td>
    <td>🐘 elephant</td>
  </tr>
  <tr>
    <th colspan="2" style="background-color: #4a5568; color: white;">Accessory</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Sports</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Kitchen</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Food</th>
  </tr>
  <tr>
    <td>👜 handbag</td>
    <td>👔 tie</td>
    <td>⚽ sports ball</td>
    <td>🏄 surfboard</td>
    <td>🍼 bottle</td>
    <td>🍷 wine glass</td>
    <td>🍎 apple</td>
    <td>🍊 orange</td>
  </tr>
  <tr>
    <td>🎒 backpack</td>
    <td>👒 hat</td>
    <td>🏸 tennis racket</td>
    <td>⛷️ skis</td>
    <td>🍽️ plate</td>
    <td>🥄 spoon</td>
    <td>🥪 sandwich</td>
    <td>🥦 broccoli</td>
  </tr>
  <tr>
    <th colspan="2" style="background-color: #4a5568; color: white;">Furniture</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Electronic</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Appliance</th>
    <th colspan="2" style="background-color: #4a5568; color: white;">Indoor</th>
  </tr>
  <tr>
    <td>🛋️ couch</td>
    <td>🪑 chair</td>
    <td>📱 cell phone</td>
    <td>💻 laptop</td>
    <td>📺 tv</td>
    <td>🔦 lamp</td>
    <td>📚 book</td>
    <td>🕰️ clock</td>
  </tr>
  <tr>
    <td>🛏️ bed</td>
    <td>🪴 potted plant</td>
    <td>🖥️ computer</td>
    <td>🖨️ printer</td>
    <td>⌨️ keyboard</td>
    <td>🔌 power outlet</td>
    <td>🧸 teddy bear</td>
    <td>🏺 vase</td>
  </tr>
</table>

### ⚙️ Custom Dataset Configuration

The model was trained using a custom configuration specified in `MY_coco30_yolov5.yaml`, which defines:

<div style="display: flex; gap: 20px;">
  <div style="flex: 1; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
    <h4>🗂️ File Structure</h4>
    <ul>
      <li>Dataset paths (relative to project root)</li>
      <li>Train/val split configuration</li>
      <li>Image directory organization</li>
      <li>Annotation format specifications</li>
    </ul>
  </div>
  <div style="flex: 1; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
    <h4>🧮 Model Parameters</h4>
    <ul>
      <li>Number of classes (80)</li>
      <li>Class names and mappings</li>
      <li>Anchor configurations</li>
      <li>Image dimensions (640x640)</li>
    </ul>
  </div>
</div>

<div class="note" style="background-color: #f4f0ec; border-left: 4px solid #a1887f; padding: 15px; margin: 20px 0; border-radius: 3px;">
  <h4>📝 Note on Retraining</h4>
  <p>If you want to retrain the model on your own data, modify the <code>MY_coco30_yolov5.yaml</code> file to point to your dataset location and adjust class configurations as needed.</p>
</div>

<br>

## 🔌 API Reference

<div align="center">
  <img src="https://i.imgur.com/rEXcoMn.png" width="90%" alt="API Flow Diagram">
  <p><i>Object Detection API Flow Diagram</i></p>
</div>

This section provides a comprehensive reference for all API endpoints, request formats, and response structures.

### 🔍 API Endpoints Overview

<table>
  <tr>
    <th style="background-color: #4a5568; color: white;">Endpoint</th>
    <th style="background-color: #4a5568; color: white;">Method</th>
    <th style="background-color: #4a5568; color: white;">Description</th>
    <th style="background-color: #4a5568; color: white;">Authentication</th>
  </tr>
  <tr>
    <td><code>/</code></td>
    <td><span style="color: green;">GET</span></td>
    <td>Root endpoint - provides API status information</td>
    <td>None</td>
  </tr>
  <tr>
    <td><code>/health</code></td>
    <td><span style="color: green;">GET</span></td>
    <td>Health check endpoint for monitoring</td>
    <td>None</td>
  </tr>
  <tr>
    <td><code>/predict</code></td>
    <td><span style="color: blue;">POST</span></td>
    <td>Main object detection endpoint</td>
    <td>None</td>
  </tr>
</table>

### 📋 Detailed Endpoint Specifications

#### 1. Root Endpoint

<div style="display: flex; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; margin-bottom: 20px;">
  <div style="width: 30%; background-color: #f7fafc; padding: 15px; border-right: 1px solid #e2e8f0;">
    <h4>Endpoint Information</h4>
    <ul style="list-style-type: none; padding-left: 0;">
      <li><b>URL:</b> <code>/</code></li>
      <li><b>Method:</b> <span style="color: green;">GET</span></li>
      <li><b>Auth Required:</b> No</li>
      <li><b>Rate Limit:</b> 100 requests/min</li>
    </ul>
  </div>
  <div style="width: 70%; padding: 15px;">
    <h4>Description</h4>
    <p>Provides information about the API status and model readiness. Use this endpoint to check if the model is loaded and ready for inference.</p>
    
    <h5>Response Example:</h5>
    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>{
  "message": "YOLOv5 Object Detection API",
  "model_status": "loaded",
  "error": null
}</code></pre>
    
    <h5>Status Values:</h5>
    <ul>
      <li><code>"loading"</code>: Model is currently being loaded</li>
      <li><code>"loaded"</code>: Model is ready for inference</li>
      <li><code>"failed"</code>: Model failed to load (error will contain details)</li>
    </ul>
  </div>
</div>

#### 2. Health Check Endpoint

<div style="display: flex; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; margin-bottom: 20px;">
  <div style="width: 30%; background-color: #f7fafc; padding: 15px; border-right: 1px solid #e2e8f0;">
    <h4>Endpoint Information</h4>
    <ul style="list-style-type: none; padding-left: 0;">
      <li><b>URL:</b> <code>/health</code></li>
      <li><b>Method:</b> <span style="color: green;">GET</span></li>
      <li><b>Auth Required:</b> No</li>
      <li><b>Rate Limit:</b> 100 requests/min</li>
    </ul>
  </div>
  <div style="width: 70%; padding: 15px;">
    <h4>Description</h4>
    <p>Simple health check endpoint used by monitoring systems and deployment platforms to verify the service is running. This endpoint will always return a success response, even if the model is still loading.</p>
    
    <h5>Response Example:</h5>
    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>{
  "status": "ok"
}</code></pre>
  </div>
</div>

#### 3. Object Detection Endpoint

<div style="display: flex; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; margin-bottom: 20px;">
  <div style="width: 30%; background-color: #f7fafc; padding: 15px; border-right: 1px solid #e2e8f0;">
    <h4>Endpoint Information</h4>
    <ul style="list-style-type: none; padding-left: 0;">
      <li><b>URL:</b> <code>/predict</code></li>
      <li><b>Method:</b> <span style="color: blue;">POST</span></li>
      <li><b>Content-Type:</b> multipart/form-data</li>
      <li><b>Auth Required:</b> No</li>
      <li><b>Rate Limit:</b> 60 requests/min</li>
    </ul>
    
    <h5>Parameters:</h5>
    <ul style="list-style-type: none; padding-left: 0;">
      <li><b>file</b> (required)</li>
      <ul>
        <li>The image file to analyze</li>
        <li>Supported formats: JPG, PNG, BMP</li>
        <li>Max size: 10MB</li>
      </ul>
    </ul>
  </div>
  <div style="width: 70%; padding: 15px;">
    <h4>Description</h4>
    <p>The main endpoint for detecting objects in images. Send an image file, and the API will return a list of detected objects with their coordinates, class, and confidence scores.</p>
    
    <h5>Success Response Example:</h5>
    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"><code>{
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
}</code></pre>
    
    <h5>Error Response Examples:</h5>
    <div style="display: flex; gap: 10px;">
      <div style="flex: 1;">
        <pre style="background-color: #fff5f5; padding: 10px; border-radius: 5px; border-left: 3px solid #f56565;"><code>{
  "error": "No file part"
}</code></pre>
      </div>
      <div style="flex: 1;">
        <pre style="background-color: #fff5f5; padding: 10px; border-radius: 5px; border-left: 3px solid #f56565;"><code>{
  "error": "Model is still loading"
}</code></pre>
      </div>
    </div>
  </div>
</div>

### 🧪 API Usage Examples

<div style="display: flex; gap: 20px; margin-bottom: 20px;">
  <div style="flex: 1; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
    <h4>Using cURL</h4>
    <pre style="background-color: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto;"><code>curl -X POST \
  -F "file=@path/to/image.jpg" \
  http://localhost:8000/predict</code></pre>
  </div>
  
  <div style="flex: 1; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
    <h4>Using Python with Requests</h4>
    <pre style="background-color: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto;"><code>import requests

url = "http://localhost:8000/predict"
image_path = "path/to/image.jpg"

with open(image_path, "rb") as image_file:
files = {"file": image_file}
response = requests.post(url, files=files)

print(response.json())</code></pre>

  </div>
</div>

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
  <h4>Using JavaScript/Fetch API</h4>
  <pre style="background-color: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto;"><code>// Using FormData and fetch
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
.catch((error) => console.error("Error:", error));</code></pre>

</div>

### 📊 Response Format Explanation

<div style="background-color: #f0fff4; border-left: 4px solid #68d391; padding: 15px; margin: 20px 0; border-radius: 3px;">
  <h4>Detection Result Fields</h4>
  <ul>
    <li><code>class</code>: Numerical class ID as defined in the COCO dataset (0-79)</li>
    <li><code>name</code>: Human-readable class name (e.g., "person", "car")</li>
    <li><code>confidence</code>: Detection confidence score between 0 and 1</li>
    <li><code>xmin, ymin</code>: Top-left coordinates of the bounding box</li>
    <li><code>xmax, ymax</code>: Bottom-right coordinates of the bounding box</li>
  </ul>
  <p><b>Note:</b> Coordinates are returned in the original image's coordinate system, not the 640x640 coordinate system used internally by the model.</p>
</div>

<div style="background-color: #ebf8ff; border-left: 4px solid #4299e1; padding: 15px; margin: 20px 0; border-radius: 3px;">
  <h4>💡 Pro Tip</h4>
  <p>For batch processing of multiple images, consider implementing a client-side queue to avoid overwhelming the API with too many simultaneous requests.</p>
</div>

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

<div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px;">
  <h3 style="border-bottom: 2px solid #4a5568; padding-bottom: 10px; margin-top: 0;">Most Common Questions</h3>
  
  <div style="margin-top: 20px;">
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        What types of objects can the model detect?
      </h4>
      <p style="margin-bottom: 0;">The model is trained on the COCO dataset and can detect 80 common object categories including people, vehicles, animals, furniture, and household items. See the <a href="#-dataset-information">Dataset Information</a> section for a complete list of categories.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        What image formats are supported?
      </h4>
      <p style="margin-bottom: 0;">The API supports common image formats including JPEG, PNG, BMP, and GIF (first frame only). For best results, use uncompressed or lightly compressed images to preserve details important for detection.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        What is the maximum image size supported?
      </h4>
      <p style="margin-bottom: 0;">There is no hard limit on image size, but larger images will take longer to process. Images are resized to 640x640 pixels for processing, but the API returns coordinates mapped to the original image dimensions. For optimal performance, we recommend keeping images under 10MB.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        Where is the COCO dataset located in this project?
      </h4>
      <p style="margin-bottom: 0;">The full COCO dataset used for training is located at <code>D:\NCT\NCT-2\S2\Capston\DataSets\COCO</code>. This path is referenced in the configuration files for training and evaluation.</p>
    </div>
  </div>
  
  <div style="margin-top: 20px;">
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        Can I use this API in a commercial application?
      </h4>
      <p style="margin-bottom: 0;">Yes, the project is licensed under the MIT License, which allows commercial use. However, be sure to check the licenses of all dependencies, especially the YOLOv5 model, which is released under the GPL-3.0 license by Ultralytics.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        How many requests can the API handle per second?
      </h4>
      <p style="margin-bottom: 0;">Performance depends on your deployment environment. On a standard Railway deployment, the API can handle approximately 10-20 requests per minute. For higher throughput, consider optimizing as described in the <a href="#-performance-optimization">Performance Optimization</a> section or deploying multiple instances behind a load balancer.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        Can I deploy this on a Raspberry Pi or other edge devices?
      </h4>
      <p style="margin-bottom: 0;">Yes, but you may need to optimize the model further. Consider using YOLOv5s or even smaller variants like YOLOv5n for edge deployment. You can also quantize the model to reduce its size and increase inference speed on resource-constrained devices.</p>
    </div>
    
    <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        How do I update the model with my own custom-trained weights?
      </h4>
      <p style="margin-bottom: 0;">Replace the <code>best.pt</code> file with your custom-trained weights. Ensure your model follows the same YOLOv5 architecture and update the class names in your code if they differ from COCO. If your model has a different architecture, you may need to modify the model loading and inference code accordingly.</p>
    </div>
  </div>
</div>

## 👥 Contributing

<div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px;">
  <h3 style="border-bottom: 2px solid #4a5568; padding-bottom: 10px; margin-top: 0;">Join Our Community</h3>
  
  <p>We welcome contributions to improve the YOLOv5 Object Detection API! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.</p>
  
  <div style="display: flex; gap: 20px; margin-top: 20px;">
    <div style="flex: 1; background-color: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">Ways to Contribute</h4>
      <ul style="padding-left: 20px; margin-bottom: 0;">
        <li><b>Report Bugs</b>: Open an issue describing the bug and steps to reproduce</li>
        <li><b>Suggest Features</b>: Open an issue describing the new feature</li>
        <li><b>Submit Pull Requests</b>: Implement bug fixes or new features</li>
        <li><b>Improve Documentation</b>: Fix errors or add examples to the docs</li>
        <li><b>Share Feedback</b>: Help us understand how you're using the API</li>
      </ul>
    </div>
    
    <div style="flex: 1; background-color: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <h4 style="margin-top: 0; color: #4a5568;">Development Workflow</h4>
      <ol style="padding-left: 20px; margin-bottom: 0;">
        <li>Fork the repository</li>
        <li>Create a feature branch (<code>git checkout -b feature/amazing-feature</code>)</li>
        <li>Make your changes</li>
        <li>Commit your changes (<code>git commit -m 'Add some amazing feature'</code>)</li>
        <li>Push to the branch (<code>git push origin feature/amazing-feature</code>)</li>
        <li>Open a Pull Request</li>
      </ol>
    </div>
  </div>
  
  <div style="background-color: #fff; border-radius: 8px; padding: 15px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h4 style="margin-top: 0; color: #4a5568;">Code Style Guidelines</h4>
    <div style="display: flex; gap: 20px;">
      <div style="flex: 1;">
        <ul style="padding-left: 20px; margin-bottom: 0;">
          <li>Follow PEP 8 Python style guidelines</li>
          <li>Include docstrings for all functions and classes</li>
          <li>Add comments for complex code segments</li>
        </ul>
      </div>
      <div style="flex: 1;">
        <ul style="padding-left: 20px; margin-bottom: 0;">
          <li>Write unit tests for new features</li>
          <li>Keep functions focused and modular</li>
          <li>Use type hints where appropriate</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<br>

---

<div align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <img src="https://i.imgur.com/bPUGhBZ.png" width="150" alt="YOLOv5 Logo">
  
  <h2>YOLOv5 Object Detection API</h2>
  <p style="font-size: 1.2em; color: #4a5568;">Developed as a Capstone Project - NCT 2025</p>
  
  <div style="margin-top: 20px;">
    <a href="https://github.com/7efnawi/OD-Model" style="text-decoration: none; background-color: #4a5568; color: white; padding: 10px 20px; border-radius: 5px; margin-right: 10px;">
      GitHub Repository
    </a>
    <a href="https://github.com/7efnawi/OD-Model/issues" style="text-decoration: none; background-color: #e53e3e; color: white; padding: 10px 20px; border-radius: 5px;">
      Report Issues
    </a>
  </div>
  
  <p style="margin-top: 30px; font-style: italic; color: #718096;">
    "The best way to detect objects is to let the machine do the seeing, and the human do the understanding."
  </p>
</div>

<!-- Quick Navigation -->
<div style="position: fixed; right: 20px; top: 50%; transform: translateY(-50%); background-color: rgba(255, 255, 255, 0.9); border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: none;">
  <h4 style="margin-top: 0; margin-bottom: 10px; text-align: center; color: #4a5568; font-size: 14px;">Quick Nav</h4>
  <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0; font-size: 12px;">
    <li style="margin-bottom: 5px;"><a href="#-introduction" style="text-decoration: none; color: #4a5568;">🔍 Introduction</a></li>
    <li style="margin-bottom: 5px;"><a href="#-system-architecture" style="text-decoration: none; color: #4a5568;">🏗 Architecture</a></li>
    <li style="margin-bottom: 5px;"><a href="#-installation-guide" style="text-decoration: none; color: #4a5568;">🛠️ Installation</a></li>
    <li style="margin-bottom: 5px;"><a href="#-api-reference" style="text-decoration: none; color: #4a5568;">🔌 API Reference</a></li>
    <li style="margin-bottom: 5px;"><a href="#-troubleshooting" style="text-decoration: none; color: #4a5568;">🛑 Troubleshooting</a></li>
  </ul>
</div>

<script>
  // This will only work if the markdown is rendered in a browser environment
  window.addEventListener('scroll', function() {
    const quickNav = document.querySelector('div[style*="position: fixed"]');
    if (quickNav) {
      if (window.scrollY > 300) {
        quickNav.style.display = 'block';
      } else {
        quickNav.style.display = 'none';
      }
    }
  });
</script>
