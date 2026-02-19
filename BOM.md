# Bill of Materials (BOM)

This document lists all components required to build the automated Focus Stacking microscope (OpenFlexure microscope).

**Estimated Grand Total:** $291.50 USD

## 1. Optics
| Component | Specification | Quantity | Approx. Cost (USD) | Supplier|
| :--- | :--- | :--- | :--- | :--- |
| Objective Lens | 20x Plan Achromatic (RMS, NA 0.40) | 1 | $83.00 | [20X microscope infinite plane long working distance objective](https://es.aliexpress.com/item/1005003369305362.html?spm=a2g0o.order_list.order_list_main.10.6d5c194daCU6Aa&gatewayAdapt=glo2esp)
| Tube Lens | Achromatic doublet ($f=50mm$) | 1 | $25.00 | [Io Rodeo Achromatic lens](https://iorodeo.com/products/achromatic-lens)
| Camera | Raspberry Pi Camera 8MP | 1 | $30.00 | [Raspberry Pi Camera Module V2](https://www.amazon.com/-/es/dp/B01ER2SKFS?ref=ppx_yo2ov_dt_b_fed_asin_title&language=en_US)
| Condenser Lens | PMMA lens $5mm \times 13mm$ | 1 | $0.50 | [Condenser Lens ](https://es.aliexpress.com/item/1052713746.html?spm=a2g0o.order_list.order_list_main.40.6ee7194dbV5ZNl&gatewayAdapt=glo2esp)
| Light Source | Star LED | 1 | $1.00 | [White LED](https://www.sigmaelectronica.net/producto/led-3w-blanco/)
| Beamsplitter | $12 mm \times 18 mm$ | 1 | $2.00 | [Beamsplitter](https://es.aliexpress.com/item/1005007473867436.html?spm=a2g0o.order_list.order_list_main.25.6ee7194dbV5ZNl&gatewayAdapt=glo2esp)

## 2. Motion Control & Electronics
| Component | Specification | Quantity | Approx. Cost (USD) | Supplier|
| :--- | :--- | :--- | :--- | :--- |
| Motors | 28BYJ-48 Stepper Motors (5V) | 3 | $12.00 | [Stepper Motors](https://electronilab.co/tienda/motor-paso-a-paso-28byj-48-0-3kgr-cm/) 
| Motor Driver | Sangaboard | 1 | $35.00 | [Sangaboard](https://iorodeo.com/products/sangaboard-v0-5)
| Computer | Raspberry Pi 4 Model B (4 GB RAM) | 1 | $55.00 | [Raspberry Pi 4](https://www.amazon.com/-/es/dp/B07TC2BK1X?ref=ppx_yo2ov_dt_b_fed_asin_title)
| Storage | microSDXC UHS-I Extreme 64 GB | 1 | $15.00 | [Micro SD 64 GB](https://www.amazon.com/-/es/SanDisk-Tarjeta-microSDXC-adaptador-SDSQXAH-064G-GN6MA/dp/B09X7C7LL1/ref=sr_1_7?crid=35PJ0NX11AD85&dib=eyJ2IjoiMSJ9.0FE4nA_s86FY4NOhTWHY-G0dwlm42X8vpn7PNH8ESP1O-R9hFIWcEybP-Vww_ypeuZXVNAOJMWVjDnTwFl2__1yT390ed42VzZVHYlbKhp3u-lykle4kN8z7akbjQwOgx2A_Q0fOmKU0BZPem154s4VBqNJ2WsM8RzsUIGM72bAnP2x4xTz20HjM2mONkRatXfE6jZjqEPL4P0-iQVoIhz7DOQqM9rWkiwsDXg7hswA.dth3rwixAfQtb9YpYiI5XoSI_CFaIRGWANpeD1ZxSo8&dib_tag=se&keywords=micro%2Bsd%2Bcard%2B64gb&qid=1771480277&sprefix=micro%2Bsd%2Bcard%2B64%2Caps%2C322&sr=8-7&th=1)
| Power | Raspberry Pi Power Supply | 1 | $5.00 | [Power Supply](https://iorodeo.com/products/raspberry-pi-power-supply)

## 3. Mechanics & Hardware
| Component | Specification | Quantity | Approx. Cost (USD) | Supplier|
| :--- | :--- | :--- | :--- | :--- |
| O-Rings | Nitrile, OD: 34mm | 3 | $1.50 | [O-Rings](https://es.aliexpress.com/item/1005006417147355.html?spm=a2g0o.order_list.order_list_main.45.6ee7194dbV5ZNl&gatewayAdapt=glo2esp)
| Screws M2 | M2 x 6mm | 2 | $0.50 |-
| Nuts/Washers M2.5 | Nut M2.5, Washer M2.5, Screw M2.5 x 8mm | Set | ~$4.00 | -
| Screws M3 | M3 (6mm, 8mm, 12mm, 25mm), Nuts, Washers | Set | ~$6.50 | -
| Screws M4 | M4 x 6mm | 6 | $1.00 | -

## 4. 3D Printed Parts
*Material Recommendation: PLA*

* 1x [Main Body](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/Main_Body.stl)
* 1x [Raspberry Pi & Sangaboard Base](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/Illumination_Condenser%20(Modified).stl)
* 1x [Reflection Illumination Condenser](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/Illumination_Condenser%20(Modified).stl)
* 1x [Reflection Illumination Holder](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/Illumination_Holder.stl)
* 1x [Reflection Optics Module Casing](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/optics_picamera2_rms_f50d13_beamsplitter_delta.stl)
* 1x [Camera Cover](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/picamera_2_cover.stl)
* 1x [Filter Cube](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/Filter_Cube.stl)
* 3x [Large Gear](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/large_gears.stl)
* 3x [Small Gear](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/small_gears.stl)
* 2x [Sample Clip](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/sample_clips.stl)
* 3x [Feet](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/feet.stl)
* Tools: [Lens Tool](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/lens_tool.stl), [Nut Tool, Band Tool](https://github.com/BallesterosFis/Focus-Stacking-for-Mechanical-Testing/blob/main/3D_Parts/actuator_assembly_tools.stl)

**Estimated 3D Printing Cost:** ~$15.00 (Filament cost)
