# Infrared Tag Detection System

Real-time AprilTag detection system using infrared imaging to detect markers embedded inside 3D-printed objects. Developed for Senior Special Project at King Mongkut's Institute of Technology Ladkrabang (KMITL).

## 🎯 Project Overview

This system enables invisible tagging of 3D-printed objects by embedding AprilTag markers that are only detectable under infrared imaging. The tags remain completely invisible to the naked eye under normal lighting conditions, allowing objects to carry digital information without affecting their visual appearance.

### Key Innovation
- **Invisible Tagging**: Tags embedded in 3D-printed objects are invisible in visible light
- **IR Detection**: Only detectable using infrared camera imaging
- **Real-time Processing**: Fast detection and analysis pipeline
- **Noise Analysis**: Sophisticated noise measurement and filtering system

## 📊 Image Processing Pipeline

![Processing Pipeline](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/processing_pipeline.jpg)
*Image processing steps: Camera On → Grayscale → CLAHE → Binary Threshold → Erosion*

## 🔬 How It Works: Invisible to Visible

![Embedded Tag Comparison](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/embedded_tag_comparison.jpg)
*Left: 3D-printed object under normal light (tag invisible) | Right: Same object under IR camera (tag clearly visible)*

The system uses:
- **IR PLA filament** inside the 3D-printed object (black layer)
- **Regular PLA** as the outer shell
- **940 nm IR illumination** to reveal the hidden tag
- **850 nm IR pass filter** to capture only IR light

## 📈 Detection Performance

![Detection Results](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/detection_results_shell_thickness.jpg)
*Detection rate vs shell thickness: Optimal detection at 0-1.5mm shell thickness*

Key findings:
- **0.00 mm shell**: 95-100% detection rate (no obstruction)
- **1.50 mm shell**: 85-90% detection rate (optimal balance)
- **3.00 mm shell**: 60-70% detection rate (challenging but possible)

## 🚀 Features

- Real-time AprilTag 16h5 detection using Raspberry Pi 5
- Advanced image preprocessing (CLAHE, binary threshold, erosion)
- Multi-rotation detection (0°, 90°, 180°, 270°)
- Perspective warping for accurate noise measurement
- Automatic and manual ROI selection modes
- Comprehensive performance metrics:
  - Raw detection rate
  - Stable detection rate
  - Noise ratio analysis
  - FPS monitoring
  - Code white ratio measurement

## 🛠️ Hardware Requirements

- **Raspberry Pi 5**
- **Raspberry Pi Camera Module 3 NoIR** (No IR filter)
- **850 nm IR Pass Filter**
- **940 nm IR LED light source**
- **3D-printed objects** with embedded IR PLA tags

## 💻 Software Stack

- **Python 3.8+**
- **OpenCV** (cv2) - Image processing and AprilTag detection
- **NumPy** - Numerical computations
- **Picamera2** - Raspberry Pi camera interface

## 📦 Installation

### 1. Clone the repository
```bash
