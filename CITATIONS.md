# Physical Augmentation & Noise Citations
This file outlines the academic and professional literature used to back up the physical SEM and Optical microscopy noise models of `dataset_generator.py`.

## 1. Secondary Electron Emission & Edge Brightening
* **Reference:** *IRDS 2024 More Moore Roadmap / FreePDK15 Predictive PDK Paper*
* **Physical Justification:** In Scanning Electron Microscopy (SEM), secondary electrons escape easier at sharp polygon edges and high feature corners. This results in local edge intensity brightening.
* **Code Implementation:** Simulated using edge boosting with the Sobel gradient magnitude filter.

---

## 2. Dielectric Surface Charging & Potential Bloom
* **Reference:** *ITRS 2015 Yield Enhancement Standards / IBM Research FinFET CMOS Process Guidelines*
* **Physical Justification:** Dielectric oxide surface accumulates an electrostatic surface charge due to electron beam interaction, scattering landing electrons and creating soft surface blooming.
* **Code Implementation:** Simulated using large kernel Gaussian blur of the surface ($121 \times 121\text{ px}$).

---

## 3. Wafer Stage Mechanical Alignment & Navigation Drift
* **Reference:** *TI Patent EP0780901A2 — Arcuate Moats and Semiconductor Alignment Systems*
* **Physical Justification:** During semiconductor wafer inspection, stepper stages suffer from mechanical tolerance and thermal drifts which cause mechanical stage misalignment of around $1^\circ\text{--}3^\circ$ and scaling drift.
* **Code Implementation:** Simulated using rotation matrix (`cv2.getRotationMatrix2D`).
