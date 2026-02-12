# Focus Stacking for Mechanical Testing

## Overview
This project provides an open-source hardware and software solution for **Focus Stacking** in mechanical testing analysis. It helps overcome depth-of-field limitations in optical microscopy when analyzing tribological samples.
The system integrates a motorized Z-axis controlled by a Raspberry Pi with custom Python scripts to capture and merge focal planes into a single, fully focused composite image.

## Repository Structure
* **`z_stack_capture.py`**: Main control script. Coordinates the Raspberry Pi, stepper motors (via Sangaboard), and camera to capture the image stack automatically.
* **`DOF.py`**: Utility script to calculate the Depth of Field (DOF) based on optical parameters.
* **`BOM.md`**: Detailed list of all hardware components and costs.

## Hardware
The microscope is built using off-the-shelf optics, standard electronics, and 3D printed parts.

* **Estimated Cost:** ~$300 USD
* **Key Components:** OpenFlexure Microscope, 20x Achromatic Lens.

**[View the full Bill of Materials (BOM)](BOM.md)**

## Requirements
To run the control scripts on the Raspberry Pi:
* Python 3.7+
* `picamera`
* `RPi.GPIO`
* `numpy`
* `opencv-python`

## Usage

1.  **Calculate Step Size:**
    Run `DOF.py` to find the optimal Z-step for your lens.
    ```bash
    python DOF.py
    ```

2.  **Capture Stack:**
    Ensure the Sangaboard is connected and run:
    ```bash
    python z_stack_capture.py
    ```

## License
This project is open-source.
