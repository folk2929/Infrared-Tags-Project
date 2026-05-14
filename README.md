# Infrared Tag Detection System

Real-time AprilTag detection system using infrared imaging to detect markers embedded inside 3D-printed objects. Developed for Senior Special Project at King Mongkut's Institute of Technology Ladkrabang (KMITL).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Visual Documentation](#-visual-documentation)
- [Key Features](#-key-features)
- [Hardware Requirements](#️-hardware-requirements)
- [Software Stack](#-software-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Tools & Utilities](#-tools--utilities)
- [Detection Pipeline](#-detection-pipeline)
- [Performance Metrics](#-performance-metrics)
- [Research Applications](#-research-applications)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Academic Context](#-academic-context)
- [Contributors](#-contributors)
- [Contact](#-contact)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [References](#-references)

---

## 🎯 Project Overview

This system enables invisible tagging of 3D-printed objects by embedding AprilTag markers that are only detectable under infrared imaging. The tags remain completely invisible to the naked eye under normal lighting conditions, allowing objects to carry digital information without affecting their visual appearance.

### The Challenge

Traditional identification methods like QR codes or barcodes are visible and can be easily removed, damaged, or compromise product aesthetics. Industries need a way to embed digital information that is:
- **Completely invisible** to human eyes
- **Tamper-resistant** and permanent
- **Reliably detectable** with proper equipment
- **Compatible** with existing manufacturing processes

### The Solution

Our system addresses these challenges through innovative use of infrared-transparent materials and computer vision:

### Key Innovation

- **Invisible Tagging**: Tags embedded in 3D-printed objects are invisible in visible light
- **IR Detection**: Only detectable using infrared camera imaging
- **Real-time Processing**: Fast detection and analysis pipeline (15-25 FPS)
- **Noise Analysis**: Sophisticated noise measurement and filtering system
- **Robust Recognition**: Multi-rotation detection for reliable identification
- **Open Source**: Complete implementation with optimization tools

---

## 📊 Visual Documentation

### Image Processing Pipeline

![Processing Pipeline](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/processing_pipeline.jpg)

*Image processing steps: Camera On → Grayscale → CLAHE → Binary Threshold → Erosion*

### How It Works: Invisible to Visible

![Embedded Tag Comparison](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/embedded_tag_comparison.jpg)

*Left: 3D-printed object under normal light (tag invisible) | Right: Same object under IR camera (tag clearly visible)*

### Technical Process

The system uses:
- **IR PLA filament** inside the 3D-printed object (black layer) - transparent to infrared light
- **Regular PLA** as the outer shell - provides structural integrity
- **940 nm IR illumination** to reveal the hidden tag through the shell
- **850 nm IR pass filter** to capture only infrared light, blocking visible spectrum

**Detection Process:**
1. **Tag Embedding**: AprilTag markers are 3D-printed using IR-transparent PLA filament
2. **Shell Encapsulation**: Regular PLA shell (0-3mm thickness) completely hides the tag
3. **IR Illumination**: 940nm IR LEDs penetrate the shell and illuminate the hidden tag
4. **IR Imaging**: NoIR camera with 850nm filter captures only infrared light
5. **Computer Vision**: Advanced algorithms detect and decode the tag in real-time

### Detection Performance

![Detection Results](https://raw.githubusercontent.com/folk2929/Infrared-Tags-Project/main/detection_results_shell_thickness.jpg)

*Detection rate vs shell thickness: Optimal detection at 0-1.5mm shell thickness*

#### Performance Analysis

Key findings from systematic testing:
- **0.00 mm shell**: 95-100% detection rate (no obstruction, baseline performance)
- **1.50 mm shell**: 85-90% detection rate (optimal balance between concealment and detectability)
- **3.00 mm shell**: 60-70% detection rate (challenging but possible with optimized settings)

#### Factors Affecting Detection

- **Shell Thickness**: Primary factor - thicker shells reduce IR transmission
- **Material Quality**: IR-transparent PLA consistency affects detection
- **Lighting Conditions**: Proper 940nm IR illumination is critical
- **Tag Size**: Larger tags more reliably detected through thicker shells
- **Processing Parameters**: CLAHE, threshold, and erosion settings tunable for different conditions

---

## 🚀 Key Features

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

---

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

---

## 💻 Software Stack

### Core Technologies

- **Python 3.8+** - Primary programming language
- **OpenCV 4.8+** (cv2) - Image processing and AprilTag detection
- **NumPy 1.24+** - Numerical computations and array operations
- **Picamera2** - Raspberry Pi camera interface
- **Pandas 2.0+** - Data analysis for benchmarking (optional)

### Computer Vision Techniques

- **CLAHE** - Contrast Limited Adaptive Histogram Equalization
- **Binary Thresholding** - Adaptive and fixed threshold methods
- **Morphological Operations** - Erosion, dilation for noise removal
- **Perspective Transform** - Warping for canonical tag view
- **Connected Components** - Blob analysis and filtering
- **Corner Refinement** - Sub-pixel accuracy for tag corners

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/folk2929/Infrared-Tags-Project.git
cd Infrared-Tags-Project
```

### 2. Install dependencies

**On Raspberry Pi 5 (Production):**
```bash
pip install opencv-contrib-python numpy picamera2 --break-system-packages
```

**On Development Machine (Benchmarking & Analysis):**
```bash
pip install -r requirements.txt
```

**Manual installation:**
```bash
# Core (minimum for basic detection)
pip install opencv-contrib-python>=4.8.0
pip install numpy>=1.24.0

# Raspberry Pi Camera (only on Pi hardware)
pip install picamera2>=0.3.0

# Data Analysis (for benchmarking tool)
pip install pandas>=2.0.0

# Optional (visualization, analysis)
pip install matplotlib>=3.7.0
```

### 3. Verify OpenCV with ArUco support

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
# Should print version and True
```

### 4. Hardware setup

1. Connect Raspberry Pi Camera Module 3 NoIR to Raspberry Pi 5 camera port
2. Attach 850 nm IR pass filter securely to camera lens
3. Position 940 nm IR LED light source for uniform illumination
4. Ensure stable mounting and proper focus distance

### 5. Verify installation

```bash
# Test basic detection (requires camera)
python infrared_tag_detection.py

# Test benchmarking tool (no camera needed)
python benchmark_apriltag_pipelines.py

# Test image preprocessing
python crop_border.py test_image.jpg
```

---

## 🚀 Quick Start

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

Edit top of `infrared_tag_detection.py`:

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

---

## 📖 Usage Guide

### Main Detection System

**File:** `infrared_tag_detection.py`

**Purpose:** Real-time AprilTag detection on Raspberry Pi 5 with comprehensive performance monitoring.

**Features:**
- Live camera preview with detection overlay
- Multi-rotation detection (0°, 90°, 180°, 270°)
- Real-time noise analysis and filtering
- Automatic and manual ROI modes
- Statistical reporting (detection rates, FPS, processing time)

**Output:**
- Real-time visual feedback with detection boxes
- Console statistics after each test round
- Average performance metrics across multiple rounds

**Use Cases:**
- Production deployment on Raspberry Pi
- Real-time tag verification
- Quality control in manufacturing
- Performance benchmarking of hardware setup

---

## 🛠️ Tools & Utilities

### 1. Image Preprocessing Tool

**File:** `crop_border.py`

**Purpose:** Automatically crop white borders from images.

**Usage:**
```bash
# Crop a single image
python crop_border.py image.jpg

# Process entire folder
python crop_border.py /path/to/folder

# Interactive mode
python crop_border.py
```

**Features:**
- Automatic border detection with configurable threshold
- Batch processing for folders
- Optional preview mode
- Preserves image quality

**Requirements:**
```bash
pip install opencv-python numpy
```

**Use Cases:**
- Prepare demonstration images
- Clean up scanned documents
- Standardize image datasets
- Remove unwanted margins

---

### 2. Pipeline Benchmarking Tool

**File:** `benchmark_apriltag_pipelines.py`

**Purpose:** Advanced benchmarking utility for comparing different image processing pipelines across multiple AprilTag detection scenarios.

#### Features

- **Multi-pipeline Testing**: Automatically tests 10,000+ preprocessing combinations
- **Multi-distance Analysis**: Evaluates performance across various tag distances/orientations
- **Statistical Analysis**: Median/mean processing times with standard deviation
- **Detection Metrics**: Hit rate, target detection ratio, processing speed
- **Automated Reporting**: CSV outputs with ranked results
- **Visual Output**: Contact sheets showing top-performing pipelines

#### Usage

```bash
python benchmark_apriltag_pipelines.py
```

#### Configuration

Edit the user settings section in the script:

```python
# Input/Output
IMAGE_SOURCE = "path/to/test/images"    # Folder with multiple test images
OUTPUT_DIR = "path/to/output"            # Results directory
TARGET_TAG_ID = 0                        # Target AprilTag ID (None for any)

# Benchmark Settings
WARMUP_RUNS = 5                          # Warmup iterations
MEASURE_RUNS = 10                        # Measurement iterations
TOP_N = 50                               # Number of top pipelines to analyze

# Reuse Previous Results
REUSE_EXISTING_BENCHMARK_CSV = False     # Skip re-benchmarking
EXISTING_BENCHMARK_CSV = "results.csv"   # Previous benchmark file
```

#### Pipeline Parameters

The tool tests combinations of:

**Preprocessing:**
- Gray (grayscale only)
- CLAHE (contrast enhancement)

**Blur Methods:**
- None, Median, Gaussian, Bilateral, Average
- Kernel sizes: 3, 5, 7, 9, 11

**Threshold Methods:**
- Otsu, Adaptive Mean, Adaptive Gaussian
- Binary: 80, 100, 127, 150, 180

**Morphological Operations:**
- None, Open, Close, Close+Open, Open+Close, Dilate, Erode
- Kernel sizes: 3, 5, 7, 9

#### Output Files

The tool generates:

1. **`top_apriltag_pipelines_multi_distance.csv`**
   - Ranked pipeline performance across all test images
   - Columns: Pipeline params, hit ratios, processing times

2. **`top50_output_summary.csv`**
   - Detailed per-image, per-pipeline results
   - Detected IDs, success/failure flags

3. **`image_detect_ranking.csv`**
   - Images ranked by detection success rate
   - Best/worst performing test cases

4. **`[image_name]/`** folders
   - Individual rank visualizations for each image
   - Contact sheets showing all top N pipelines
   - Visual comparison of preprocessing results

#### Use Cases

**Research & Development:**
- Find optimal preprocessing for specific lighting conditions
- Compare performance across different tag sizes/distances
- Identify robust pipelines for varying environments

**Quality Assurance:**
- Validate detection reliability before deployment
- Benchmark system performance requirements
- Document preprocessing decisions with evidence

**Optimization:**
- Select fastest pipeline meeting accuracy requirements
- Balance detection rate vs. processing speed
- Tune parameters for specific hardware constraints

#### Performance Notes

- **Computation Time**: ~30-60 minutes for full benchmark (depends on image count and hardware)
- **Memory Usage**: ~2-4 GB RAM (caches images for speed)
- **Disk Space**: ~100-500 MB output (50 images × 50 pipelines)
- **CPU**: Single-threaded by design for benchmark stability

#### Example Workflow

```bash
# 1. Prepare test images at various distances
mkdir test_images
# Add images: close.jpg, medium.jpg, far.jpg, etc.

# 2. Run benchmark (first time)
python benchmark_apriltag_pipelines.py

# 3. Review results
# Check top_apriltag_pipelines_multi_distance.csv for best performers

# 4. Visualize results
# Open output/[image_name]/ folders for visual comparison

# 5. Implement winner
# Use top pipeline parameters in your production code
```

#### Integration with Main Detection System

The best pipeline found via benchmarking can be directly applied to `infrared_tag_detection.py`:

```python
# Example: Benchmark winner was CLAHE + Median(5) + Binary_100 + Erode(7)

# In infrared_tag_detection.py, update:
CLAHE_CLIP_LIMIT = 2.0          # Keep CLAHE
CLAHE_TILE_GRID = (8, 8)
BINARY_THRESH = 100              # From benchmark
ERODE_KERNEL_SIZE = 7            # From benchmark

# Add median blur before threshold (new step):
def build_processed_image(gray_img):
    clahe_img = create_clahe_image(gray_img)
    median_img = cv2.medianBlur(clahe_img, 5)  # NEW: Add blur
    _, binary_img = cv2.threshold(median_img, BINARY_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8)
    processed_img = cv2.erode(binary_img, kernel, iterations=1)
    return processed_img
```

#### Technical Details

**Benchmark Stability:**
- Single-threaded execution (OpenCV, BLAS, OpenBLAS threads = 1)
- OpenCL disabled for consistent timing
- Warmup runs to eliminate cache effects
- Garbage collection paused during measurement

**Ranking Algorithm:**
Primary: Target Hit Count (descending)
Secondary: Total Hit Count (descending)
Tertiary: Target Hit Ratio (descending)
Quaternary: Total Hit Ratio (descending)
Quinary: Median Processing Time (ascending)
Senary: Standard Deviation (ascending)

**Statistical Reporting:**
- Median time: Robust to outliers
- Mean time: Overall average
- Std deviation: Consistency metric
- Hit ratio: Success rate across test set

---

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

---

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

---

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

---

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

---

## 📁 Project Structure
Infrared-Tags-Project/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── infrared_tag_detection.py             # Main detection system (Raspberry Pi)
├── benchmark_apriltag_pipelines.py       # Pipeline optimization tool
├── crop_border.py                        # Image preprocessing utility
│
├── processing_pipeline.jpg               # Documentation images
├── embedded_tag_comparison.jpg
├── detection_results_shell_thickness.jpg
│
└── Demonstration images/                 # (Optional) Sample images
├── example1.jpg
└── example2.jpg

### File Descriptions

**Core System Files:**
- `infrared_tag_detection.py` - Real-time detection on Raspberry Pi with performance monitoring
- `benchmark_apriltag_pipelines.py` - Multi-pipeline comparison for parameter optimization
- `crop_border.py` - Automatic white border removal for preprocessing

**Configuration:**
- `requirements.txt` - Python package dependencies with versions
- `LICENSE` - MIT License for open-source distribution

**Documentation:**
- `README.md` - Complete project documentation (you're reading it)
- `processing_pipeline.jpg` - Image processing pipeline visualization
- `embedded_tag_comparison.jpg` - Visible vs. IR comparison
- `detection_results_shell_thickness.jpg` - Performance vs. thickness graph

---

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

---

## 👥 Contributors

**Panupong Jaichop** (Student ID: 65050686)
- Lead developer, system design, algorithm implementation
- Computer vision pipeline, noise analysis algorithms
- Hardware integration, testing, documentation

**Thanathorn Wongpayak** (Student ID: 65050378)
- 3D printing experimentation, IR PLA material testing
- Hardware setup, illumination optimization
- Performance testing, data collection

---

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

---

## 📝 License

This project is licensed under the MIT License - free for academic and commercial use with attribution.

See [LICENSE](LICENSE) file for details.

---

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

---

## 📚 References

**AprilTag System:**
- Olson, E. (2011). "AprilTag: A robust and flexible visual fiducial system." *IEEE International Conference on Robotics and Automation (ICRA)*.
- Wang, J., & Olson, E. (2016). "AprilTag 2: Efficient and robust fiducial detection." *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.

**Computer Vision:**
- OpenCV ArUco Module Documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Bradski, G. (2000). "The OpenCV Library." *Dr. Dobb's Journal of Software Tools*.

**Hardware:**
- Raspberry Pi Camera Documentation: https://www.raspberrypi.com/documentation/accessories/camera.html
- Raspberry Pi 5 Documentation: https://www.raspberrypi.com/documentation/computers/raspberry-pi-5.html

**Image Processing:**
- Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*. Academic Press Professional.

---

## 🚀 Getting Started Guide

### For Researchers

1. Read the [Academic Context](#-academic-context) and [Technical Details](#-technical-details) sections
2. Review the [Detection Pipeline](#-detection-pipeline) and algorithms
3. Examine the [Performance Metrics](#-performance-metrics) and testing methodology
4. [Contact](#-contact) for collaboration or questions

### For Developers

1. Clone the repository and [install dependencies](#-installation)
2. Review the [Project Structure](#-project-structure) and code organization
3. Test with [Quick Start](#-quick-start) guide
4. Explore [Tools & Utilities](#️-tools--utilities) for optimization

### For Industry Partners

1. Review [Research Applications](#-research-applications) section
2. Assess feasibility for your use case
3. [Contact](#-contact) for technology transfer discussions
4. Explore customization possibilities

---

## 📊 Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Detection Rate (0mm) | 95-100% | Baseline, no obstruction |
| Detection Rate (1.5mm) | 85-90% | Optimal shell thickness |
| Detection Rate (3.0mm) | 60-70% | Maximum tested thickness |
| Processing Speed | 15-25 FPS | Raspberry Pi 5 |
| Noise Ratio | <5% | Good quality threshold |
| Tag Distance | 10-50 cm | Typical working range |
| Resolution | 1280×720 | Camera capture |

---

## 🔧 Troubleshooting

### Common Issues

**Issue: No tags detected**
- Check IR LED is on (may be invisible to eye)
- Verify 850nm filter is installed correctly
- Adjust LENS_POS for proper focus
- Increase BINARY_THRESH if image too dark

**Issue: Low FPS (<15)**
- Reduce TARGET_RES to (640, 480)
- Check CPU temperature (throttling)
- Disable other running processes
- Verify single-threaded operation

**Issue: High noise ratio (>10%)**
- Check 3D print quality
- Verify IR PLA material quality
- Adjust CLAHE_CLIP_LIMIT (try 1.5-2.5)
- Modify ERODE_KERNEL_SIZE (try 5-9)

**Issue: Unstable detection (flashing)**
- Adjust STABLE_HIT_WINDOW (try 7 or 10)
- Check IR illumination uniformity
- Verify camera mounting stability
- Increase warmup frames

---

## 🎯 Future Work

### Planned Enhancements

- **Multi-tag Tracking**: Simultaneous detection of multiple tags
- **Depth Estimation**: Calculate tag distance from size
- **Motion Compensation**: Stabilization for moving cameras
- **Web Interface**: Remote monitoring and configuration
- **Mobile App**: Smartphone-based detection

### Research Directions

- **Material Science**: Optimize IR-transparent filaments
- **Deep Learning**: Neural network-based detection
- **Edge Computing**: Optimize for lower-power devices
- **Standards Compliance**: ISO/FDA regulatory approval

---

**Last Updated:** May 15, 2026  
**Version:** 2.0.0  
**Status:** ✅ Complete - Production Ready  

---

**Note**: This system requires IR-transparent 3D printing filament (IR PLA) and proper IR illumination for optimal results. Performance varies with material quality, shell thickness, and environmental conditions. Contact authors for guidance on specific applications.

---

**⭐ If you find this project useful, please give it a star on GitHub!**

**🐛 Found a bug? [Open an issue](https://github.com/folk2929/Infrared-Tags-Project/issues)**

**💡 Have an idea? [Start a discussion](https://github.com/folk2929/Infrared-Tags-Project/discussions)**
