# Semiconductor Navigation Drift Recovery — SEMICON India Hackathon 2026
AI and computer vision based target localization and navigation recovery algorithm for high precision Scanning Electron Microscope (SEM) wafer inspection systems.
The algorithm solves the problem of localizing a high-resolution reference pattern inside a low-resolution wide field search image, where the two images differ approximately by a 10× difference in physical field of view. The solution is expected to be robust to SEM sensor noise, repetitive semiconductor structures, stage rotation and scaling drift.

## Problem Statement
During a semiconductor inspection, it is required to locate a high-resolution reference image inside a much wider search region which is captured with a lower resolution.

The difficulties are:
- Large field-of-view differences
- SEM sensor noise and degradation
- Repetitive DRAM and FinFET structures
- Stage rotation and scaling drift
- Ambiguous pattern matching
- Variations in illumination and channels in RGB optical images

The algorithm implements scale compensation followed by multi-scale normalized cross-correlation to localize the target.

## Repository Structure
```text
├── README.md                 # Project documentation & setup guide
├── dataset_generator.py      # Synthetic dataset generator
├── infer.py                  # Target localizer script
├── export_weights.py         # PyTorch-to-ONNX weight exporter
├── model.onnx                # Pre-trained deep feature extractor weights
├── CITATIONS.md              # Physical noise models & literature citations
├── requirements.txt          # Python dependencies
└── data/                     # Output directory for generated datasets
```
---


## Quickstart Guide

### Step 1: Environment Setup

Clone this repository and install all required dependencies:

```bash
git clone https://github.com/Tanisha110705/Semiconductor-NavigationDriftRecovery-SemiconIndia-2026.git
cd Semiconductor-NavigationDriftRecovery-SemiconIndia-2026
pip install -r requirements.txt
```

---

### Step 2: Create Synthetic Data

The `dataset_generator.py` script generates synthetic training data consisting of:

- A high-resolution reference image of **1000 × 1000 pixels**
- A wide-field search image representing approximately a **10× larger physical field of view**
- Realistic SEM noise and imaging variations
- Stage rotation and scaling drift

The synthetic dataset includes physical noise models such as **edge emission brightening, dielectric charging bloom, stage rotation, and scaling drift**.

#### Generate SEM Grayscale Images

**DRAM periodic array pairs:**

```bash
python dataset_generator.py --style DRAM --num_pairs 5 --output_dir ./data
```

**FinFET logic pairs:**

```bash
python dataset_generator.py --style FinFET --num_pairs 5 --output_dir ./data
```

#### Generate RGB Optical Microscope Images

To generate 3-channel RGB optical microscope image pairs:

```bash
python dataset_generator.py --style DRAM --num_pairs 5 --output_dir ./data_rgb --is_rgb
```

Each run automatically generates a `ground_truth.csv` file inside the specified output directory containing the exact **(x, y) target center coordinates**.

---

### Step 3: Perform Target Localization Inference

The `infer.py` script is the main target localization and evaluation script.

It automatically:

1. Detects whether the input consists of **1-channel SEM grayscale** or **3-channel RGB optical images**.
2. Compensates for the approximately **10× spatial scale difference** between the reference and search images.
3. Performs multi-scale normalized cross-correlation.
4. Determines the most likely target location.
5. Outputs the estimated **(x, y)** target coordinates.

Run the inference script using:

```bash
python infer.py
```

---

## Algorithmic Approach & Methodology

### 1. Resolution Compensation

The **1000 × 1000 pixel reference patch** is initially scaled down by a base factor of approximately **0.10×** to match the spatial scale of the wide-field search image.

### 2. Multi-Scale Correlation Map

Normalized Cross-Correlation (NCC) is performed over a scale range of approximately:

```text
0.08× – 0.12×
```

This allows the algorithm to compensate for **stage scaling drift** and variations in the physical imaging scale.

### 3. Multi-Channel Spectral Fusion — RGB Bonus

For 3-channel optical images, normalized cross-correlation is independently performed on the:

- Blue (B) channel
- Green (G) channel
- Red (R) channel

The individual correlation maps are then combined to improve robustness against illumination and channel variations.

### 4. Periodic Array Tie-Breaker

DRAM and FinFET structures contain highly repetitive patterns, which can produce multiple similar correlation peaks.

To resolve ambiguous matches, the algorithm uses a **center-based tie-breaker**, favoring candidates closer to the search image center:

```text
Search center = (500, 500)
```

This helps distinguish the most likely target location when multiple structurally similar regions are detected.

---

##  Physical Noise & Augmentation Models

The synthetic data generation process incorporates physical imaging variations based on semiconductor device and inspection literature.

### 1. Edge Enhancement

Models the **secondary electron emission** behavior observed in SEM imaging, producing enhanced intensity near structural edges.

**Reference:** IRDS 2024 More Moore Roadmap.

### 2. Surface Charge Glow

Models image intensity variations caused by **surface/dielectric charging effects** during SEM imaging.

**Reference:** ITRS Yield Enhancement Specifications.

### 3. Stage Rotation & Scaling Drift

Models small variations in imaging caused by stage movement, rotation, and scale changes during semiconductor inspection.

**Reference:** TI Semiconductor Alignment Patent — EP0780901A2.

For detailed literature references, mathematical models, and implementation details, refer to:

```text
CITATIONS.md
```

---

##  Output

The localization algorithm produces the estimated target center coordinates:

```text
(x, y)
```

These coordinates represent the position of the high-resolution reference pattern within the wide-field search image.

The generated datasets also contain a `ground_truth.csv` file for evaluating localization accuracy.

---

##  References

Detailed physical noise models, mathematical formulations, and literature references are provided in:

**`CITATIONS.md`**
