# Infrared Tag Detection System

Real-time AprilTag detection system using infrared imaging to detect markers embedded inside 3D-printed objects. Developed for Senior Special Project at King Mongkut's Institute of Technology Ladkrabang (KMITL).

![Detection Demo](images/demo.jpg)
<!-- ใส่รูปผลลัพธ์ของคุณ -->

## 🎯 Project Overview

This system enables invisible tagging of 3D-printed objects by embedding AprilTag markers that are only detectable under infrared imaging. The tags remain completely invisible to the naked eye under normal lighting conditions, allowing objects to carry digital information without affecting their visual appearance.

### Key Innovation
- **Invisible Tagging**: Tags embedded in 3D-printed objects are invisible in visible light
- **IR Detection**: Only detectable using infrared camera imaging
- **Real-time Processing**: Fast detection and analysis pipeline
- **Noise Analysis**: Sophisticated noise measurement and filtering system

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
git clone https://github.com/folk2929/Infrared-Tags-Project.git
cd Infrared-Tags-Project
```

### 2. Install dependencies
```bash
pip install opencv-python numpy picamera2 --break-system-packages
```

### 3. Hardware setup
- Connect Raspberry Pi Camera Module 3 NoIR to Raspberry Pi 5
- Attach 850 nm IR pass filter to camera lens
- Set up 940 nm IR LED illumination

## 🎮 Usage

### Basic Detection
```bash
python infrared_tag_detection.py
```

### Keyboard Controls
- **S** - Start detection test
- **R** - Reset test
- **M** - Toggle Manual ROI mode
- **C** - Clear manual ROI
- **A/D** - Adjust max noise blob area
- **ESC** - Exit program

### Configuration Parameters

Key parameters in the code:
```python
TARGET_RES = (1280, 720)          # Camera resolution
LENS_POS = 6.5                     # Manual focus position
TARGET_ID = 0                      # AprilTag ID to detect
CLAHE_CLIP_LIMIT = 2.0            # Contrast enhancement
BINARY_THRESH = 100                # Binary threshold
ERODE_KERNEL_SIZE = 7             # Erosion kernel
```

## 📊 Detection Pipeline

1. **Image Acquisition** - Capture frame from Raspberry Pi NoIR camera
2. **Preprocessing**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Binary thresholding
   - Morphological erosion
3. **Multi-rotation Detection** - Detect tags at 0°, 90°, 180°, 270°
4. **Perspective Warping** - Transform tag region to canonical view
5. **Code Area Extraction** - Identify actual AprilTag code regions
6. **Noise Measurement** - Analyze background noise in tag area
7. **Metrics Calculation** - Compute detection rates and noise ratios

## 📈 Performance Metrics

The system measures:
- **Raw Detection Rate**: Percentage of frames where tag is detected
- **Stable Detection Rate**: Consistent detection over sliding window
- **Noise Ratio**: White pixels in background vs total background
- **Code White Ratio**: AprilTag code pixels vs total area
- **FPS**: Frames per second throughput
- **Processing Time**: Per-frame computation time

## 🔬 Research Applications

This technology enables:
- **Object-Digital Linking**: Connect physical objects to digital information
- **Covert Tagging**: Invisible identification for authentication
- **Manufacturing**: Quality control without visible marks
- **IoT Integration**: Seamless physical-digital interfaces

## 📄 Technical Details

### AprilTag Detection
Uses OpenCV's ArUco module with AprilTag 16h5 dictionary:
- Adaptive threshold window: 3-23 pixels
- Corner refinement: Sub-pixel accuracy
- Polygon approximation: 0.08 accuracy rate

### Noise Analysis Algorithm
1. Warp tag region to 240×240 canonical view
2. Extract code blobs (white regions > 500 px)
3. Dilate code mask to create exclusion zone
4. Identify noise blobs in background (1-999999 px)
5. Calculate noise ratio: noise pixels / background pixels

## 📸 Results

Expected detection performance:
- Detection rate: 85-95% (depends on tag size and shell thickness)
- FPS: 15-25 on Raspberry Pi 5
- Noise ratio: < 5% for quality prints

## 🎓 Academic Context

**Project:** Senior Special Project (2024-2025)  
**Institution:** King Mongkut's Institute of Technology Ladkrabang  
**Department:** Industrial Physics  
**Advisor:** Asst. Prof. Dr. Bhanupol Klongratog  
**Grade:** A

## 👥 Contributors

**Panupong Jaichop** (Student ID: 65050686)  
**Thanathorn Wongpayak** (Student ID: 65050378)

## 📧 Contact

- **Email**: lnwfolk29@gmail.com
- **LinkedIn**: [linkedin.com/in/panupong-jaichop-52505a322](https://linkedin.com/in/panupong-jaichop-52505a322)
- **GitHub**: [@folk2929](https://github.com/folk2929)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- KMITL Department of Industrial Physics
- Metrology & Inspection Laboratory
- Asst. Prof. Dr. Bhanupol Klongratog (Project Advisor)
- Asst. Prof. Dr. Thanavit Anuwongpinit (Co-advisor)

## 📚 References

- AprilTag: A robust and flexible visual fiducial system (Olson, 2011)
- OpenCV ArUco Module Documentation
- Raspberry Pi Camera Documentation

---

**Note**: This system requires IR-transparent 3D printing filament (IR PLA) and proper IR illumination for optimal results.
