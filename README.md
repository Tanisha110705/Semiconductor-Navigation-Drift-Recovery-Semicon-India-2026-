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
├── README.md                 # Documentation for the project
├── dataset_generator.py      # Synthetic dataset generator
├── infer.py                  # Target localizer
├── CITATIONS.md              # Citations and methodology
├── requirements.txt          # Python requirements
└── data/                     # Dataset generat

Quickstart Guide

Step 1: Environment Setup
Clone this repository and install all required dependencies:
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
Requirements: Python 3.8+ with opencv-python, numpy, and scipy.

Step 2: Create Synthetic Data (dataset_generator.py)In order to create synthetic data for training our network, we need to generate the high-resolution reference image of (1000px X 1000px) and wide search image of (1000px X 1000px), which is 10X the physical field of view area. This dataset will have the physical noise SEM models (edge emission brightening, dielectric charging bloom, stage rotation, and scaling drift)

Generate SEM Grayscale Images (DRAM / FinFET):
# Generate DRAM periodic array pairs (Grayscale SEM)
python dataset_generator.py --style DRAM --num_pairs 5 --output_dir ./data
# Generate FinFET logic pairs (Grayscale SEM)
python dataset_generator.py --style FinFET --num_pairs 5 --output_dir ./data

Generate RGB Optical Microscope Images
# Generate 3-channel RGB optical microscope pairs
python dataset_generator.py --style DRAM --num_pairs 5 --output_dir ./data_rgb --is_rgb
Each run automatically writes a ground_truth.csv file inside the output directory containing exact (x, y) target center coordinates.

Step 3: Perform Target Localization Inference (infer.py)
infer.py is the main script used for evaluation purposes. This automatically distinguishes between 1-channel SEM gray-scale and 3-channel RGB optical images and scales down the reference patch by taking into account the spatial scale difference of 10X. The output of the process is the location (x, y).

Algorithmic Approach & Methodology
i)Resolution Compensation: Scales down the 1000px X 1000px reference patch to a base factor of 0.10X in order to be in-line with the spatial scale of the wide-search field of view.
ii)Multiple Scale Correlation Map: Conducts normalized cross-correlation (NCC) over a scale interval (0.08Xto 0.12X) such that it becomes immune to 20 stage scaling drift.
iii)Multi-Channel Spectral Fusion (RGB Bonus): In cases of 3-channel optical imaging, conducts independent normalized cross-correlation on B, G, and R channels, merging them to survive any changes in lighting conditions. Periodic Array Tie-Breaker: Solves structural repetition (a natural problem in DRAM/FinFET design) by favoring match candidates closer to the search center (500, 500).

Physical Noise & Augmentation Citations
All synthetically generated augmentations are based on physical semiconductor devices literature
i)Edge Enhancement (Secondary Electron Emission): Based on IRDS 2024 More Moore Roadmap.
ii)Surface Charge Glow: Based on ITRS Yield Enhancement Specifications.
iii)Stage Rotations & Scaling: Based on TI Semiconductor Alignment Patent EP0780901A2.
For more literature citations and mathematical models, see CITATIONS.md.
