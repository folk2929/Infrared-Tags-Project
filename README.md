# Infrared Tag Detection System

Real-time AprilTag detection system using infrared imaging to detect markers embedded inside 3D-printed objects. Developed for Senior Special Project at King Mongkut's Institute of Technology Ladkrabang (KMITL).

## 🎯 Project Overview

This system enables invisible tagging of 3D-printed objects by embedding AprilTag markers that are only detectable under infrared imaging. The tags remain completely invisible to the naked eye under normal lighting conditions, allowing objects to carry digital information without affecting their visual appearance.

### The Challenge
Traditional identification methods like QR codes or barcodes are visible and can be easily removed, damaged, or compromise product aesthetics. Industries need a way to embed digital information that is:
- Completely invisible to human eyes
- Tamper-resistant and permanent
- Reliably detectable with proper equipment
- Compatible with existing manufacturing processes

### The Solution
Our system addresses these challenges through innovative use of infrared-transparent materials and computer vision:

### Key Innovation
- **Invisible Tagging**: Tags embedded in 3D-printed objects are invisible in visible light
- **IR Detection**: Only detectable using infrared camera imaging
- **Real-time Processing**: Fast detection and analysis pipeline with 15-25 FPS
- **Noise Analysis**: Sophisticated noise measurement and filtering system
- **Robust Recognition**: Multi-rotation detection for reliable identification

## 📊 Image Processing Pipeline

![Processing Pipeline](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/processing_pipeline.jpg)
*Image processing steps: Camera On → Grayscale → CLAHE → Binary Threshold → Erosion*

## 🔬 How It Works: Invisible to Visible

![Embedded Tag Comparison](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/embedded_tag_comparison.jpg)
*Left: 3D-printed object under normal light (tag invisible) | Right: Same object under IR camera (tag clearly visible)*

The system uses:
- **IR PLA filament** inside the 3D-printed object (black layer) - transparent to infrared light
- **Regular PLA** as the outer shell - provides structural integrity
- **940 nm IR illumination** to reveal the hidden tag through the shell
- **850 nm IR pass filter** to capture only infrared light, blocking visible spectrum

### Technical Process
1. **Tag Embedding**: AprilTag markers are 3D-printed using IR-transparent PLA filament
2. **Shell Encapsulation**: Regular PLA shell (0-3mm thickness) completely hides the tag
3. **IR Illumination**: 940nm IR LEDs penetrate the shell and illuminate the hidden tag
4. **IR Imaging**: NoIR camera with 850nm filter captures only infrared light
5. **Computer Vision**: Advanced algorithms detect and decode the tag in real-time

## 📈 Detection Performance

![Detection Results](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/detection_results_shell_thickness.jpg)
*Detection rate vs shell thickness: Optimal detection at 0-1.5mm shell thickness*

### Performance Analysis

Key findings from systematic testing:
- **0.00 mm shell**: 95-100% detection rate (no obstruction, baseline performance)
- **1.50 mm shell**: 85-90% detection rate (optimal balance between concealment and detectability)
- **3.00 mm shell**: 60-70% detection rate (challenging but possible with optimized settings)

### Factors Affecting Detection
- **Shell Thickness**: Primary factor - thicker shells reduce IR transmission
- **Material Quality**: IR-transparent PLA consistency affects detection
- **Lighting Conditions**: Proper 940nm IR illumination is critical
- **Tag Size**: Larger tags more reliably detected through thicker shells
- **Processing Parameters**: CLAHE, threshold, and erosion settings tunable for different conditions

## 🚀 Features

### Detection Capabilities
- Real-time AprilTag 16h5 detection using Raspberry Pi 5
- Advanced image preprocessing (CLAHE, binary threshold, erosion)
- Multi-rotation detection (0°, 90°, 180°, 270°) for robust recognition
- Perspective warping for accurate noise measurement
- Sub-pixel corner refinement for precise localization
- Automatic and manual ROI selection modes

### Performance Monitoring
Comprehensive real-time metrics:
- **Raw Detection Rate**: Percentage of frames where tag is detected
- **Stable Detection Rate**: Consistent detection over sliding window (5 frames, minimum 4 hits)
- **Noise Ratio**: White pixels in background vs total background area
- **Code White Ratio**: AprilTag code pixels vs total tag area
- **FPS**: Frames per second throughput (15-25 FPS typical)
- **Processing Time**: Per-frame computation time (<67ms for real-time)

### Quality Assurance
- Noise blob analysis (size filtering, exclusion zones)
- Geometry validation (aspect ratio, minimum size, border detection)
- Code area extraction and verification
- Background masking and artifact removal

## 🛠️ Hardware Requirements

### Required Components
- **Raspberry Pi 5** (4GB or 8GB RAM recommended)
- **Raspberry Pi Camera Module 3 NoIR** (12MP, no IR filter)
- **850 nm IR Pass Filter** (blocks visible light, passes IR)
- **940 nm IR LED light source** (invisible to human eye)
- **3D-printed objects** with embedded IR PLA tags

### Optional Components
- Adjustable mounting system for camera positioning
- Diffused lighting for uniform IR illumination
- Cooling fan for extended operation

### Hardware Setup Notes
- Camera focus manually adjustable (LENS_POS parameter)
- IR LED placement affects shadow patterns
- Filter quality impacts detection reliability
- Stable mounting critical for consistent results

## 💻 Software Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **OpenCV 4.8+** (cv2) - Image processing and AprilTag detection
- **NumPy 1.24+** - Numerical computations and array operations
- **Picamera2** - Raspberry Pi camera interface

### Computer Vision Techniques
- **CLAHE** - Contrast Limited Adaptive Histogram Equalization
- **Binary Thresholding** - Adaptive and fixed threshold methods
- **Morphological Operations** - Erosion, dilation for noise removal
- **Perspective Transform** - Warping for canonical tag view
- **Connected Components** - Blob analysis and filtering
- **Corner Refinement** - Sub-pixel accuracy for tag corners

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/folk2929/Infrared-Tags-Project.git
cd Infrared-Tags-Project
```

### 2. Install dependencies
```bash
# On Raspberry Pi OS
pip install opencv-python numpy picamera2 --break-system-packages

# On other systems (development/testing)
pip install opencv-python numpy
```

### 3. Hardware setup
1. Connect Raspberry Pi Camera Module 3 NoIR to Raspberry Pi 5 camera port
2. Attach 850 nm IR pass filter securely to camera lens
3. Position 940 nm IR LED light source for uniform illumination
4. Ensure stable mounting and proper focus distance

### 4. Verify installation
```bash
python infrared_tag_detection.py
# Should open camera preview window
```

## 🎮 Usage

### Basic Detection
```bash
python infrared_tag_detection.py
```

### Keyboard Controls
- **S** - Start detection test (begins performance measurement)
- **R** - Reset test (clears all statistics)
- **M** - Toggle Manual ROI mode (custom region selection)
- **C** - Clear manual ROI (revert to automatic detection)
- **A/D** - Adjust max noise blob area (decrease/increase threshold)
- **ESC** - Exit program

### Configuration Parameters

Key parameters in the code (top of file):
```python
# Camera Settings
TARGET_RES = (1280, 720)          # Camera resolution
MANUAL_FOCUS = True                # Enable manual focus
LENS_POS = 6.5                     # Manual focus position (0-15)

# Detection Settings
TARGET_ID = 0                      # AprilTag ID to detect
TEST_FRAMES = 1005                 # Frames per test round
TEST_ROUNDS = 3                    # Number of test rounds
WARMUP_FRAMES = 5                  # Frames to skip at start

# Image Processing
CLAHE_CLIP_LIMIT = 2.0            # Contrast enhancement strength
CLAHE_TILE_GRID = (8, 8)          # CLAHE grid size
BINARY_THRESH = 100                # Binary threshold value
ERODE_KERNEL_SIZE = 7             # Erosion kernel size

# Noise Analysis
MIN_NOISE_BLOB_AREA = 1           # Minimum noise blob size (pixels)
MAX_NOISE_BLOB_AREA = 999999      # Maximum noise blob size (adjustable)
MIN_CODE_BLOB_AREA = 500          # Minimum code blob size
CODE_BORDER_MARGIN = 12           # Border exclusion margin
```

### Testing Workflow
1. **Setup**: Position object with embedded tag under IR illumination
2. **Preview**: Run program, adjust camera focus if needed
3. **Start Test**: Press 'S' to begin automated testing
4. **Monitor**: Watch real-time metrics and detection visualization
5. **Results**: System reports detection rates, FPS, noise levels after each round
6. **Analysis**: Review averaged results across multiple rounds

## 🖼️ Image Preprocessing Tool

Included is a utility script for automatically cropping white borders from demonstration images:

```bash
# Crop a single image
python crop_border.py image.jpg

# Process entire folder
python crop_border.py /path/to/folder

# Interactive mode
python crop_border.py
```

**Requirements:**
```bash
pip install opencv-python numpy
```

This tool was used to prepare the demonstration images in this repository.

## 📊 Detection Pipeline

### Step-by-Step Process

1. **Image Acquisition**
   - Capture frame from Raspberry Pi NoIR camera at 1280×720 resolution
   - Rotate 180° to correct camera orientation
   - Convert RGB to grayscale

2. **Preprocessing**
   - **CLAHE** - Enhance local contrast while limiting noise amplification
   - **Binary Threshold** - Convert to pure black/white (threshold = 100)
   - **Morphological Erosion** - Remove small noise pixels (kernel = 7×7)

3. **Multi-rotation Detection**
   - Detect tags at 0°, 90°, 180°, 270° rotations
   - Map detected corners back to original orientation
   - Validate tag geometry (size, aspect ratio, border clearance)
   - Select best detection across rotations

4. **Perspective Warping**
   - Transform tag region to 240×240 canonical view
   - Order corners visually (top-left, top-right, bottom-right, bottom-left)
   - Apply perspective transform for undistorted view

5. **Code Area Extraction**
   - Identify white regions that are actual AprilTag code
   - Filter by minimum size (500 pixels)
   - Exclude blobs touching border (margin = 12 pixels)
   - Dilate code mask to create exclusion zone

6. **Noise Measurement**
   - Define background as valid area minus dilated code mask
   - Find white pixels in background (noise candidates)
   - Filter noise by size range (1-999999 pixels, adjustable)
   - Calculate noise ratio: noise pixels / background pixels

7. **Metrics Calculation**
   - Compute detection rates (raw and stable)
   - Calculate FPS from processing time
   - Track noise statistics (ratio, pixels, blob count)
   - Update running averages across rounds

## 📈 Performance Metrics

### Real-time Measurements

**Detection Rates:**
- **Raw Detection Rate**: Instantaneous frame-by-frame detection (%)
- **Stable Detection Rate**: Requires 4/5 consecutive detections for stability
- Averaged over TEST_FRAMES (default 1005) per round

**Processing Performance:**
- **Total Time**: Complete pipeline processing time per frame (ms)
- **Median Total Time**: Robust central tendency across round
- **FPS**: Frames per second = 1000 / median_time

**Noise Analysis:**
- **Noise Ratio**: (noise_pixels / background_pixels) × 100%
- **Noise Pixels**: Total white pixels outside code areas
- **Noise Blob Count**: Number of discrete noise regions
- **Background Pixels**: Total valid background area

**Code Metrics:**
- **Code White Ratio**: (code_pixels / total_tag_area) × 100%
- **Total White Ratio**: (code_pixels + noise_pixels) / total_tag_area × 100%

### Statistical Reporting

After each round:
- Individual round performance summary
- Median values for noise and processing time

After all rounds:
- **Average Raw Rate**: Mean across all rounds
- **Average Stable Rate**: Mean stable detection
- **Average FPS**: Mean processing speed
- **Average Noise Ratio**: Mean noise level
- Confidence in results increases with more rounds

## 🔬 Research Applications

### Current Applications
This technology enables:
- **Object-Digital Linking**: Connect physical objects to databases, digital twins, or blockchain records
- **Covert Tagging**: Invisible identification for authentication without altering product appearance
- **Manufacturing Quality Control**: Track components without visible markings
- **IoT Integration**: Seamless physical-digital interfaces for smart objects
- **Supply Chain Tracking**: End-to-end traceability without compromising aesthetics
- **Anti-Counterfeiting**: Permanent, tamper-resistant authentication marks

### Future Possibilities
- **Medical Device Traceability**: FDA-compliant invisible marking
- **Art Authentication**: Invisible signatures in 3D-printed art
- **Security Documents**: Hidden authentication in official documents
- **Smart Packaging**: Invisible tags for consumer product interaction
- **Augmented Reality**: Physical anchors for AR experiences
- **Aerospace Components**: Critical part tracking without weight/appearance impact

## 📄 Technical Details

### AprilTag Detection Configuration
Uses OpenCV's ArUco module with AprilTag 16h5 dictionary:

**Detection Parameters:**
```python
adaptiveThreshWinSizeMin = 3      # Minimum window for adaptive threshold
adaptiveThreshWinSizeMax = 23     # Maximum window
adaptiveThreshWinSizeStep = 10    # Window size increment

minMarkerPerimeterRate = 0.01     # Minimum marker size (% of image)
maxMarkerPerimeterRate = 6.0      # Maximum marker size

polygonalApproxAccuracyRate = 0.08  # Corner approximation accuracy
minCornerDistanceRate = 0.02        # Minimum corner separation

minDistanceToBorder = 2            # Edge detection margin
cornerRefinementMethod = CORNER_REFINE_SUBPIX  # Sub-pixel accuracy
cornerRefinementWinSize = 7        # Refinement window
cornerRefinementMaxIterations = 80  # Refinement iterations
cornerRefinementMinAccuracy = 0.01  # Convergence threshold
```

**Why AprilTag 16h5:**
- High error correction capability (5-bit Hamming code)
- Robust to partial occlusion
- Reliable orientation detection
- Wide industry adoption
- 30,000+ unique tag IDs available

### Noise Analysis Algorithm

**Detailed Steps:**

1. **Warp tag region to 240×240 canonical view**
   - Extract perspective transform matrix
   - Apply to both detection image and noise image
   - Ensures consistent analysis across tag orientations

2. **Extract code blobs (white regions > 500 px)**
   - Connected components analysis
   - Size filtering (MIN_CODE_BLOB_AREA = 500 pixels)
   - Border exclusion (CODE_BORDER_MARGIN = 12 pixels)
   - Identifies actual AprilTag code regions

3. **Dilate code mask to create exclusion zone**
   - Dilation kernel size: CODE_DILATE_K = 2
   - Creates buffer around code to avoid false noise detection
   - Prevents edge artifacts from being counted as noise

4. **Identify noise blobs in background**
   - Define background = valid area - dilated code mask
   - Find white pixels in background
   - Connected components on noise candidates
   - Filter by size: MIN_NOISE_BLOB_AREA ≤ area ≤ MAX_NOISE_BLOB_AREA

5. **Calculate noise ratio: noise_pixels / background_pixels × 100%**
   - Normalized metric independent of tag size
   - Lower values indicate better print quality
   - Threshold: <5% considered good quality

### Geometry Validation

**Tag Acceptance Criteria:**
- Minimum bounding box area: 8000 pixels
- Minimum side length: 60 pixels
- Maximum aspect ratio deviation: 0.45 (target 1.0 for square tags)
- Must not touch image borders (2-pixel margin)

**Rejection Reasons:**
- "no corners" - No tag detected
- "bbox too small" - Tag too small to reliably detect
- "area too small" - Insufficient total area
- "aspect off" - Not square enough (damaged/distorted)
- "touch border" - Partial tag at image edge

## 🎓 Academic Context

**Project Details:**
- **Title:** Infrared Tag Detection System for 3D-Printed Objects
- **Type:** Senior Special Project (Individual)
- **Duration:** May 2024 - January 2025 (8 months)
- **Institution:** King Mongkut's Institute of Technology Ladkrabang (KMITL)
- **Faculty:** Faculty of Science
- **Department:** Industrial Physics
- **Program:** Dual Degree - B.Sc. Industrial Physics + B.Eng. IoT Systems

**Supervision:**
- **Advisor:** Asst. Prof. Dr. Bhanupol Klongratog
- **Co-advisor:** Asst. Prof. Dr. Thanavit Anuwongpinit
- **Laboratory:** Metrology & Inspection Laboratory

**Assessment:**
- **Grade:** A
- **Credits:** 6 credits (SPHY499 - Special Project I)
- **Evaluation:** Technical merit, innovation, documentation, presentation

## 👥 Contributors

**Panupong Jaichop** (Student ID: 65050686)
- Lead developer, system design, algorithm implementation
- Computer vision pipeline, noise analysis algorithms
- Hardware integration, testing, documentation

**Thanathorn Wongpayak** (Student ID: 65050378)
- 3D printing experimentation, IR PLA material testing
- Hardware setup, illumination optimization
- Performance testing, data collection

## 📧 Contact

**Panupong Jaichop**
- **Email**: lnwfolk29@gmail.com
- **LinkedIn**: [linkedin.com/in/panupong-jaichop-52505a322](https://linkedin.com/in/panupong-jaichop-52505a322)
- **GitHub**: [@folk2929](https://github.com/folk2929)
- **Institution**: King Mongkut's Institute of Technology Ladkrabang

**Open to:**
- Collaboration on computer vision projects
- Discussions about IoT and embedded systems
- Research opportunities in applied AI
- Industry partnerships for technology transfer

## 📝 License

This project is licensed under the MIT License - free for academic and commercial use with attribution.

## 🙏 Acknowledgments

**Academic Support:**
- KMITL Department of Industrial Physics
- Asst. Prof. Dr. Bhanupol Klongratog (Project Advisor)
- Asst. Prof. Dr. Thanavit Anuwongpinit (Co-advisor)
- Metrology & Inspection Laboratory staff

**Technical Resources:**
- OpenCV community and documentation
- Raspberry Pi Foundation
- AprilTag development team (University of Michigan)

**Material Support:**
- KMITL 3D Printing Lab
- Industrial Physics Laboratory facilities

## 📚 References

**AprilTag System:**
- Olson, E. (2011). "AprilTag: A robust and flexible visual fiducial system." IEEE International Conference on Robotics and Automation (ICRA).
- Wang, J., & Olson, E. (2016). "AprilTag 2: Efficient and robust fiducial detection." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

**Computer Vision:**
- OpenCV ArUco Module Documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Bradski, G. (2000). "The OpenCV Library." Dr. Dobb's Journal of Software Tools.

**Hardware:**
- Raspberry Pi Camera Documentation: https://www.raspberrypi.com/documentation/accessories/camera.html
- Raspberry Pi 5 Documentation: https://www.raspberrypi.com/documentation/computers/raspberry-pi-5.html

**Image Processing:**
- Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization." Graphics Gems IV. Academic Press Professional.

---

## 🚀 Getting Started Guide

**For Researchers:**
1. Read the academic context and technical details sections
2. Review the detection pipeline and algorithms
3. Examine the performance metrics and testing methodology
4. Contact for collaboration or questions

**For Developers:**
1. Clone the repository and install dependencies
2. Review the code structure and configuration parameters
3. Test with sample images or live camera
4. Modify parameters for your specific use case

**For Industry Partners:**
1. Review research applications section
2. Assess feasibility for your use case
3. Contact for technology transfer discussions
4. Explore customization possibilities

---

**Note**: This system requires IR-transparent 3D printing filament (IR PLA) and proper IR illumination for optimal results. Performance varies with material quality, shell thickness, and environmental conditions. Contact authors for guidance on specific applications.

**Last Updated:** May 15, 2026
**Version:** 1.0.0
**Status:** ✅ Complete - Tested and Documented
