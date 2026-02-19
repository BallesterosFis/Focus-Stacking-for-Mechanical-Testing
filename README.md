# Robotic Focus Stacking for Mechanical Surface Analysis

## Abstract

This repository contains the hardware control scripts and computational tools supporting the robotic focus stacking system.

The system addresses depth of field limitations in optical microscopy during tribological and mechanical surface analysis. By integrating a motorized Z-axis stage with automated image acquisition and post-processing, the setup enables reproducible extended depth-of-field (EDOF) imaging using a low-cost optical platform.


---

## Scientific Motivation

Extended depth of field (EDOF) imaging mitigates the limitations of conventional optical microscopy by computationally merging multiple focal planes into a single fully focused image. However, commercial automated systems remain costly and frequently closed-source.

This project implements a fully reproducible robotic focus stacking workflow based on:

- Motorized Z-axis control  
- Theoretical depth of field estimation  
- Automated stack acquisition  
- Alignment and EDOF reconstruction  
- Optional 3D surface visualization  

---

## System Architecture

The complete workflow is structured as follows:

Optical Setup  
↓  
Experimental DOF Estimation (`DOF.py`)  
↓  
Motorized Z-Stack Acquisition (`z_stack_capture.py`)  
↓  
Stack Alignment (SIFT-based registration in Fiji)  
↓  
Extended Depth of Field Reconstruction  
↓  
3D Surface Visualization  

The hardware implementation is based on an OpenFlexure microscope platform integrated with a motorized Z-stage controlled by a Raspberry Pi.

---

## Repository Structure

- `z_stack_capture.py`  
  Controls Z-axis motion and automated image acquisition via Raspberry Pi and Sangaboard.

- `DOF.py`  
  Calculates the experimental depth of field based on three metrics (Laplacian variance, Tenengrad gradient, and wavelet energy) and the 80% and FWHM creteria.

- `BOM.md`  
  Bill of materials including hardware components and cost estimation (~300 USD total system cost).

## Experimental Results

The system enables:

- Fully focused composite images of wear tracks and surface defects  
- Improved visualization of depth-dependent features  
- 3D surface rendering from focal stacks  

---

## Reproducibility

All scripts are designed to run on a Raspberry Pi (Python 3.7+) with standard scientific libraries:

- numpy  
- opencv-python  
- RPi.GPIO  
- picamera  

Post-processing is performed using Fiji (ImageJ) with:

- [Linear Stack Alignment with SIFT ](https://imagej.net/plugins/linear-stack-alignment-with-sift) 
- [Extended Depth of Field (Expert Mode) ](https://imagej.net/plugins/extended-depth-of-field) 
- [3D Surface Plot ](https://imagej.net/ij/plugins/surface-plot-3d.html) 

---

## Limitations

Current implementation:

- Requires post-processing in Fiji (not fully integrated in Python)  
- Assumes stable illumination and minimal lateral drift   

Future work will focus on full pipeline automation and improved illumination stability control.


## License

CERN Open Hardware Licence v1.2
