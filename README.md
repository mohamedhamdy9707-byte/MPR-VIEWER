# MPR-VIEWER
MultiPlanarViewer for medical imaging with impressive features

# OVERVIEW

MPR viewer that visualise 3d medical data.

 The application views MPR,oplique plane and segmentation outline.
 
 The application supports NIFTI and DICOM formats , and comes with AI models that detect organ nad orientation names.
 ![]( https://github.com/mohamedhamdy9707-byte/MPR-VIEWER/blob/main/assets/MPR.png)
<div align="center">
</div>

 # Features

This application is a comprehensive medical imaging viewer with advanced features for visualization, analysis, and interaction, built with Python and PyQt5.

* **Multi-Format Support**: Load 3D medical volumes from various formats, including DICOM series, NIfTI (`.nii`, `.nii.gz`), and NumPy (`.npy`).
* **Advanced MPR Viewing**: Visualize data in four linked panels: standard **Axial**, **Sagittal**, and **Coronal** views, plus a fully interactive **3D Oblique** view.
* **Synchronized Crosshair Navigation**: Click any point in one of the orthogonal views to instantly re-center the other views to the same anatomical location, ensuring precise spatial correlation.
* **Interactive Oblique Plane**: Freely adjust the oblique slice with intuitive sliders for pitch, yaw, roll, and depth to explore any anatomical plane.
* **AI-Powered Organ Detection**: Utilizes the `TotalSegmentator` library to run in the background and automatically identify the primary organ within the loaded volume, displaying its name.
* **ROI Analysis & Export**: Draw a Region of Interest (ROI) on any 2D view and export the corresponding 3D sub-volume as a new NIfTI or NumPy file with a user-defined number of slices.
* **Dedicated Contour Mode**: Load an external segmentation mask and enter a specialized mode to visualize the 2D contour outlines for specific labels, allowing for easy verification of segmentation results.
* **Flexible Interaction**: Navigate slices with sliders, pan/zoom views, and use the cine (play) mode for dynamic visualization.
* **Export Capabilities**: Save the full loaded volume, individual 2D slices, or the extracted ROI volume for use in other applications.
  ![]( https://github.com/mohamedhamdy9707-byte/MPR-VIEWER/blob/main/assets/ROI.png)
<div align="center">
</div>

 ![](https://github.com/mohamedhamdy9707-byte/MPR-VIEWER/blob/main/assets/ORGAN%20DETECTION.jpg)
<div align="center">
</div>




