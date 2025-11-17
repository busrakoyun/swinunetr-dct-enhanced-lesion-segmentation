# Segmenting Neonatal Brain Lesions from Low-Resolution Diffusion MRI

**Graduation Project – Istanbul Technical University, Faculty of Computer and Informatics, Department of Computer Engineering**  
**Graduation Project Achievement Award – 2025**  
**Accepted for presentation at AICCC 2025 – 8th Artificial Intelligence and Cloud Computing Conference (Tokyo, Japan)**

Authors: **Büşra Koyun**, **Ayça Pektaş**, **Senih Yıldırım** 
Advisor: **Asst. Prof. Dr. Yusuf Hüseyin Şahin**

---

## Overview

This repository contains the implementation of our graduation project,  
**“Segmenting Neonatal Brain Lesions from Low-Resolution Diffusion MRI”**, conducted at  
**Istanbul Technical University (ITU), Faculty of Computer and Informatics, Department of Computer Engineering**.

The project addresses the challenging task of segmenting **Hypoxic Ischemic Encephalopathy (HIE)** lesions in **neonatal diffusion MRI** acquired under **low-resolution, undersampled clinical conditions**. We study how **frequency-domain priors** can improve lesion segmentation when data is heavily constrained by accelerated MRI acquisition.

Our work has been:

- 🎓 **Awarded the Graduation Project Achievement Award** by ITU Computer Engineering Department  
- 🧠 **Accepted for presentation at AICCC 2025 (Tokyo University of Agriculture and Technology, Japan)**

---

## Method in Brief

We investigate three SwinUNETR-based architectures under simulated accelerated MRI acquisition:

1. **Baseline SwinUNETR (Dual-Channel)**
   - Inputs: ADC and ZADC volumes
   - Purely spatial-domain segmentation (no frequency information)

2. **Global DCT Model**
   - Slice-wise **2D Discrete Cosine Transform (DCT)** applied to ADC
   - Global DCT map concatenated as a third input channel
   - Provides a compact **global frequency representation**

3. **Block DCT Model (Frequency-Aware Encoder)**
   - Learnable **VolumeDCTPatchLayer** for patch-wise DCT on ADC
   - Localized frequency features fused into early SwinUNETR encoder stages
   - Captures **spatially distributed spectral cues** critical for small, diffuse lesions

All models are trained directly on **low-resolution, aliased MRI** without reconstructing to high-resolution, mirroring realistic neonatal clinical workflows.

---

## Dataset and Undersampling

We use the **BONBID-HIE** dataset, which contains skull-stripped:

- Apparent Diffusion Coefficient maps (**ADC**)
- Z-score normalized ADC maps (**ZADC**)
- Expert-annotated binary lesion masks (**LABEL**)

To simulate accelerated acquisition and low-resolution conditions, we apply:

- **4× and 8× k-space undersampling**
- Both **random** and **equispaced** sampling masks
- Slice-wise:
  1. Centered 2D FFT
  2. Masked k-space
  3. Zero-filled inverse FFT
  4. Magnitude extraction → aliased ADC / ZADC volumes

This produces realistic **low-resolution, artifact-prone inputs** for segmentation.

---

## Quantitative Results

### Mean Dice Scores (4× undersampled data)

| Model                       | Dice Score |
|-----------------------------|-----------:|
| Baseline (ADC + ZADC)      | 0.52       |
| Global DCT (Patch 128)     | 0.55       |
| **Block DCT (Patch 48)**   | **0.56**   |

### Block DCT – Patch Size and Mask Type

| Mask Type  | Patch Size | 4× Dice (Mean ± Std) | 8× Dice (Mean ± Std) |
|------------|------------|----------------------|----------------------|
| Random     | 32         | 0.5233 ± 0.0138      | 0.4570 ± 0.0149      |
| Random     | 48         | 0.4965 ± 0.0174      | 0.4453 ± 0.0077      |
| Random     | 64         | 0.4942 ± 0.0214      | 0.4364 ± 0.0214      |
| Equispaced | 32         | **0.5269 ± 0.0137**  | 0.4583 ± 0.0148      |
| Equispaced | 48         | 0.5013 ± 0.0175      | 0.4483 ± 0.0076      |
| Equispaced | 64         | 0.4991 ± 0.0232      | 0.4383 ± 0.0212      |

**Key findings:**

- Incorporating **frequency-domain priors** consistently improves segmentation over the baseline.
- The **Block DCT configuration with smaller patches (e.g., 32)** and equispaced sampling yields the most stable performance.
- Even under aggressive acceleration (8×), frequency-aware models show improved robustness.

---

## Repository Structure

```text
dct_layer.py                         # Global and Block DCT feature extraction modules
dct_model_train.py                   # Training script / experiment setup
requirements.txt                     # Python dependencies
Graduation_Project_Final_Report.pdf  # Full project report (ITU format)
