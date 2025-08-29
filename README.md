# 📷 Camera Calibration using OpenCV

<div align="center">

![Camera Calibration](https://img.shields.io/badge/Computer%20Vision-Camera%20Calibration-blue?style=for-the-badge&logo=opencv)
![Python](https://img.shields.io/badge/Python-3.7+-green?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red?style=for-the-badge&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Latest-orange?style=for-the-badge&logo=numpy)

*A comprehensive implementation of camera calibration algorithms for determining intrinsic and extrinsic camera parameters*

</div>

## 🎯 Overview

This project implements a complete camera calibration pipeline using computer vision techniques and OpenCV. The system determines both **intrinsic** and **extrinsic** camera parameters by analyzing checkerboard patterns, enabling accurate 3D-to-2D projections and camera pose estimation.

### 🔬 What is Camera Calibration?

Camera calibration is the process of estimating the parameters of a camera model. These parameters include:
- **Intrinsic Parameters**: Internal camera characteristics (focal length, principal point, distortion coefficients)
- **Extrinsic Parameters**: Camera's position and orientation in 3D world coordinates

## ✨ Key Features

### 🔧 Core Functionality
- **3D Rotation Matrix Computation**: Implements Euler angle to rotation matrix conversion
- **Checkerboard Corner Detection**: Automated detection of calibration pattern corners
- **Intrinsic Parameter Estimation**: Calculates focal length (fx, fy) and principal point (cx, cy)
- **Extrinsic Parameter Estimation**: Determines camera rotation (R) and translation (T) matrices
- **Sub-pixel Accuracy**: Enhanced corner detection with sub-pixel precision

### 📊 Technical Capabilities
- Handles dual checkerboard patterns (8x4 configuration)
- Implements Direct Linear Transformation (DLT) algorithm
- Uses Singular Value Decomposition (SVD) for robust parameter estimation
- Provides verification through back-projection of 3D points

## 🏗️ Project Structure

```
📦 Camera-Calibration-using-OpenCV/
├── 📄 UB_Geometry.py          # Main implementation file
├── 📄 result_task2.json       # Sample calibration results
├── 🖼️ checkboard.png          # Calibration pattern image
├── 📄 README.md               # Project documentation
└── 📄 LICENSE                 # Apache 2.0 License
```

## 🚀 Implementation Details

### Task 1: 3D Rotation Transformations
```python
def findRot_xyz2XYZ(alpha, beta, gamma)
def findRot_XYZ2xyz(alpha, beta, gamma)
```
- Converts Euler angles (degrees) to 3x3 rotation matrices
- Implements ZYX rotation sequence
- Provides forward and inverse transformations

### Task 2: Camera Calibration Pipeline
```python
def find_corner_img_coord(image)      # Corner detection in image coordinates
def find_corner_world_coord(img_coord) # 3D world coordinates mapping
def find_intrinsic(img_coord, world_coord)  # Intrinsic parameter estimation
def find_extrinsic(img_coord, world_coord)  # Extrinsic parameter estimation
```

## 📋 Requirements

### Dependencies
```bash
pip install opencv-python numpy matplotlib
```

### System Requirements
- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+
- Matplotlib 3.0+

## 🔬 Algorithm Implementation

### 1. Corner Detection Process
- **Image Preprocessing**: Convert to grayscale for better corner detection
- **Pattern Recognition**: Detect 4x4 checkerboard patterns on each half of the image
- **Sub-pixel Refinement**: Enhance corner accuracy using `cornerSubPix()`
- **Coordinate Mapping**: Combine left and right checkerboard coordinates

### 2. Calibration Mathematics
The implementation uses the **Direct Linear Transformation (DLT)** method:

1. **Projection Matrix Estimation**: Solve the linear system Ap = 0 using SVD
2. **Parameter Extraction**: Decompose projection matrix into intrinsic and extrinsic components
3. **Normalization**: Apply proper scaling to ensure unit normal for rotation matrix

### 3. World Coordinate System
The calibration cube uses a 40mm × 40mm × 40mm coordinate system:
- **Origin**: Corner of the calibration cube
- **Units**: Millimeters
- **Pattern**: 32 corner points across two perpendicular faces

## 📊 Sample Results

The system achieves high accuracy calibration with typical results:

```json
{
  "fx": 2788.29,     // Focal length X
  "fy": 2788.25,     // Focal length Y  
  "cx": 995.45,      // Principal point X
  "cy": 564.41,      // Principal point Y
  "R": [...],        // 3x3 Rotation matrix
  "T": [...]         // 3x1 Translation vector
}
```

## 🎓 Educational Value

This implementation demonstrates:
- **Linear Algebra**: Matrix operations, SVD decomposition
- **Computer Vision**: Image processing, feature detection
- **3D Geometry**: Coordinate transformations, projective geometry
- **Numerical Methods**: Least squares optimization, parameter estimation

## 🔍 Usage Example

```python
import cv2
import numpy as np
from UB_Geometry import *

# Load calibration image
image = cv2.imread('checkboard.png')

# Detect corners
img_coords = find_corner_img_coord(image)
world_coords = find_corner_world_coord(img_coords)

# Calibrate camera
fx, fy, cx, cy = find_intrinsic(img_coords, world_coords)
R, T = find_extrinsic(img_coords, world_coords)

print(f"Focal Length: fx={fx:.2f}, fy={fy:.2f}")
print(f"Principal Point: cx={cx:.2f}, cy={cy:.2f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for excellent computer vision tools
- Academic resources on camera calibration theory
- Computer vision textbooks and research papers

---

**📚 This project is created for educational purposes as part of CSE 573 coursework.**
