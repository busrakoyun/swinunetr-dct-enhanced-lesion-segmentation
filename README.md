# Segmenting Neonatal Brain Lesions from <br> Low-Resolution Diffusion MRI

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

## Dataset and Undersampling

This project uses the **BONBID-HIE 2024** dataset (Boston Neonatal Brain Injury Dataset for Hypoxic Ischemic Encephalopathy).  
Dataset access:  
🔗 **https://bonbid-hie2024.grand-challenge.org/data/**
*(If you encounter a 403 error, simply open the link in a new tab.)*

The BONBID-HIE dataset provides skull-stripped neonatal diffusion MRI volumes, including:

- Apparent Diffusion Coefficient maps (**ADC**)  
- Z-score normalized ADC maps (**ZADC**)  
- Expert-annotated binary lesion masks (**LABEL**)  

To simulate clinically realistic accelerated MRI acquisition and low-resolution imaging conditions, we apply:

### **K-space Undersampling & Aliasing Pipeline**
1. Slice-wise centered **2D FFT**  
2. Application of **4× or 8× undersampling masks**  
   - both **random** and **equispaced** patterns  
3. **Zero-filled inverse FFT** reconstruction  
4. Magnitude extraction → aliased **ADC** / **ZADC** volumes  

This produces realistic **low-resolution, artifact-prone diffusion MRI** consistent with neonatal clinical constraints.

Full preprocessing and undersampling details can be found in the project report (`Graduation_Project_Final_Report.pdf`).

## Key Findings

- Incorporating **frequency-domain priors** (DCT features) consistently improved segmentation performance over the baseline SwinUNETR model.
- The **Block DCT** variant provided the most stable and accurate results, particularly for small and diffuse neonatal HIE lesions.
- **Smaller patch sizes (e.g., 32)** captured localized spectral information more effectively than larger patches.
- Both **4× and 8× undersampling** scenarios benefited from frequency-aware features, demonstrating improved robustness under low-resolution and aliasing-heavy conditions.
- All quantitative results and detailed evaluations are provided in the project report (`Graduation_Project_Final_Report.pdf`).

## Installation

```bash
git clone https://github.com/busrakoyun/swinunetr-dct-enhanced-lesion-segmentation.git
cd swinunetr-dct-enhanced-lesion-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Acknowledgments

We would like to express our sincere gratitude to **Asst. Prof. Dr. Yusuf Hüseyin Şahin** for his continuous guidance, technical insights, and invaluable support throughout this project.  
We also thank the **Istanbul Technical University – Faculty of Computer and Informatics** for providing the computational resources required for training and experimentation.  
Finally, we acknowledge the creators of the **BONBID-HIE 2024** dataset for enabling open research on neonatal brain injury segmentation.

## License

This project is released under the **MIT License**.  
See the `LICENSE` file for full terms.
