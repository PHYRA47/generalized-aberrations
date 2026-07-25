# Lens Design Files for "Generalized Aberrations for Processing-Aware Optical Design"

This directory contains lens design files in `.seq` format for all designs discussed in the paper *"Generalized Aberrations for Processing-Aware Optical Design"*.

## Folder Structure and Contents

Each subfolder contains a set of lens designs described in detail below:

### **smartphone_telephoto_lenses**
- **Total Designs**: 70
- **Description**: Contains smartphone telephoto lens designs discussed in the relevant section of the manuscript.
- **Configurations**: 7 distinct configurations, each covering 5 telephoto ratios.
- **Optimization Types**:
  - *Conventional*: Optimized for minimum spot radius.
  - *End-to-End (E2E)*: Co-optimized with an image restoration model for optimal post-restoration image quality.

### **microscope_objective_lenses**
- **Total Designs**: 30
- **Description**: Microscope objective lens designs presented in the corresponding section of the manuscript.
- **Configurations**: 3 distinct configurations, each covering 5 working distances.
- **Optimization Types**:
  - *Conventional*: Optimized for minimum spot radius.
  - *Image-Driven*: Optimized for optimal pre-restoration image quality.

### **toy_lens_design_problem**
- **Total Designs**: 3
- **Description**: Two-element aspherical lens designs discussed in Fig. 1c-e of the manuscript.
- **Lens Variants**:
  - *Starting Point*: The initial lens design.
  - *Conventional*: Optimized for minimum spot radius.
  - *Image-Driven*: Optimized for optimal pre-restoration image quality

### **smartphone_wideangle_lenses**
- **Total Designs**: 4
- **Description**: Wide-angle smartphone lens designs, detailed in Supplementary Note 6, with one design (labeled "ours") shown in Fig. 10a of the manuscript.
- **Lens Variants**:
  - *Starting Point*: The baseline design.
  - *Ours*: The design optimized with our optimization method.
  - *Comparison Lenses*: Includes two designs from Yeng et al. (2024), with one refocused for fair comparison.

### **cmount_lenses**
- **Total Designs**: 3
- **Description**: C-mount lens designs explored in Supplementary Note 7.
- **Optimization Strategies**:
  - *Conventional*: Optimized for spot radius.
  - *Image-Driven*: Optimized for optimal pre-restoration image quality.
  - *E2E*: Co-optimized with an image restoration model for optimal post-restoration image quality.

### **demo_4p_smartphone_telephoto_lens**
- **Total Designs**: 1
- **Description**: A 4-element smartphone telephoto lens presented in Supplementary Note 1, designed to validate the modeling operations used in the study.
