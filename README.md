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

 ![](https://github.com/mohamedhamdy9707-byte/MPR-VIEWER/blob/main/assets/ORGAN%20DETECTION.jpg)
<div align="center">
</div>

 ![]([https://github.com/mohamedhamdy9707-byte/MPR-VIEWER/blob/main/assets/SEG%20MODE.png ))
<div align="center">
</div>

# Requirements
```
pip install -r requirements.txt
```

## How to Use

### 1. Loading Data
- Use the **"Load DICOM"** button to select a folder containing a DICOM series. The application will automatically sort the slices and construct a 3D volume.
- Use the **"Load NIfTI/NumPy"** button to open a `.nii`, `.nii.gz`, or `.npy` file.

### 2. Navigating the Views
- **Slice Navigation**: Use the sliders below each of the Axial, Sagittal, and Coronal views to scroll through the slices. You can also use the mouse wheel while hovering over a view.
- **Crosshair Sync**: Click anywhere inside the Axial, Sagittal, or Coronal views. A green crosshair will appear, and all three views will instantly jump to that anatomical coordinate.
- **Zooming**: Hold down the `Ctrl` key and scroll the mouse wheel to zoom in or out. The zoom level is synchronized across all views.
- **Panning**: Click and drag with the left mouse button to pan the view.

### 3. Using the 3D Oblique View
- The bottom-right panel displays a 3D oblique view, which is also centered on the crosshair location.
- Use the **Pitch, Yaw, and Roll** sliders on the right-hand panel to rotate the oblique plane to any angle.
- The **Depth** slider moves the plane along its normal vector.
- To quickly align the oblique view, select a standard view (e.g., "Axial") from the dropdown and click **"Apply View → Oblique"**.

### 4. AI Organ Detection
- After loading a volume, click the **"Detect Organ (AI)"** button.
- The application will run `TotalSegmentator` in the background. This may take a moment.
- Once complete, the name of the largest detected organ will be displayed in the top-right and overlaid on the Oblique view.

### 5. Extracting an ROI
1.  Click the **"ROI (r)"** button corresponding to the view you want to draw on (e.g., Axial). The status bar will confirm ROI mode is active.
2.  Click and drag to draw a yellow rectangle defining your Region of Interest.
3.  Click the **"Save ROI Volume"** button.
4.  You will be prompted to enter the number of slices (depth) to include in the sub-volume.
5.  Save the resulting ROI volume as a new NIfTI or NumPy file.

### 6. Contour Mode
1.  First, load a primary imaging volume (e.g., a CT scan).
2.  Click **"Load Seg (Contour)"** and select the corresponding segmentation mask file (in NIfTI or NumPy format).
3.  Choose the desired label from the **"Label"** dropdown menu.
4.  Click **"Enter Contour Mode"**. The three main views will be disabled, and the fourth panel will become a dedicated contour viewer.
5.  Use the slider and the "View" button in the contour panel to inspect the segmentation outlines.
6.  Click **"Exit Contour Mode"** to return to the standard MPR view.
# CONTRIBUTERS
[@mhmdhamddyy](https://github.com/mohamedhamdy9707-byte) 


[@MahmoudMazen0](https://github.com/MahmoudMazen0) 

[ebrahimnas577](https://github.com/ebrahimnas577) 
# Under the Supervision of
Prof. Dr. Tamer Basha


Eng. Alaa Tarek




