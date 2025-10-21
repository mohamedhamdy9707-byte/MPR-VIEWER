

import sys
import os
import glob
import tempfile
import shutil
import numpy as np
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QFileDialog, QPushButton, QComboBox, QMessageBox,
    QProgressDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates, rotate, find_objects

# optional libs
try:
    import nibabel as nib

    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

try:
    import pydicom

    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

try:
    from skimage import measure

    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


# -------------------------
# Background worker (runs totalsegmentator via python API)
# -------------------------
class SegThread(QThread):
    progress_line = pyqtSignal(str)  # emits status lines for UI
    finished_success = pyqtSignal(str)  # emits output_folder path on success
    finished_error = pyqtSignal(str)  # emits error message on failure

    def __init__(self, input_nifti_path, output_folder, fast=True):
        super().__init__()
        self.input_nifti_path = input_nifti_path
        self.output_folder = output_folder
        self.fast = fast

    def run(self):
        """
        Run TotalSegmentator using its python API (totalsegmentator.python_api.totalsegmentator).
        Emit progress_line messages for UI feedback and finished_success / finished_error signals.
        """
        try:
            # Import inside thread to avoid requiring it at module import time
            from totalsegmentator.python_api import totalsegmentator
        except Exception as e:
            self.finished_error.emit(f"Could not import totalsegmentator python_api: {e}")
            return

        try:
            self.progress_line.emit("TotalSegmentator: starting (fast mode = {})...".format(bool(self.fast)))
            # ensure output folder exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Call Totalsegmentator API (this runs inference and writes masks to output folder)
            # Note: totalsegmentator will log to stdout/stderr internally; we emit simple messages here.
            totalsegmentator(input=self.input_nifti_path, output=self.output_folder, fast=self.fast)

            # finished successfully
            self.progress_line.emit("TotalSegmentator: finished.")
            self.finished_success.emit(self.output_folder)
        except Exception as e:
            # propagate error message
            self.finished_error.emit(f"TotalSegmentator run failed: {e}")


# -------------------------
# Oblique reslice helper (FROM MPR 2)
# -------------------------
def reslice_oblique(volume, center, normal, up_vec, output_size=256, spacing=1.0):
    """
    Sample a 2D oblique plane from `volume` given: center (z,y,x), normal and up vector.
    Uses map_coordinates for trilinear sampling.

    volume expected as (z,y,x)
    """
    normal = np.array(normal, dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-12)
    up = np.array(up_vec, dtype=float)
    # make orthonormal basis (u = right, v = up, n = normal)
    u = np.cross(up, normal)
    if np.linalg.norm(u) < 1e-8:
        # if up is parallel to normal, choose arbitrary up
        up = np.array([0, 1, 0])
        u = np.cross(up, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    coords = (np.linspace(-output_size / 2, output_size / 2 - 1, output_size) * spacing)
    yy, xx = np.meshgrid(coords, coords)
    pts = np.stack([
        center[0] + xx.ravel() * u[0] + yy.ravel() * v[0],
        center[1] + xx.ravel() * u[1] + yy.ravel() * v[1],
        center[2] + xx.ravel() * u[2] + yy.ravel() * v[2],
    ], axis=0)

    slice_vals = map_coordinates(volume, pts, order=1, mode='nearest')
    return slice_vals.reshape(output_size, output_size)


# -------------------------
# ViewPanel tiny container
# -------------------------
class ViewPanel:
    def __init__(self, name):
        self.name = name
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax.axis('off')
        self.label = QLabel(name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)

        self.controls_widget = None  # Will store the play/roi/etc buttons

        # ROI state
        self.roi_patch = None
        self.roi_start = None
        self.roi_rect = None
        self.is_selecting_roi = False

        # panning
        self.is_panning = False
        self.pan_start = None

        # crosshair
        self.crosshair_h = None
        self.crosshair_v = None

        self.cine_timer = None

        # Store original name
        self.original_name = name


# -------------------------
# Main application
# -------------------------
class MPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPR Viewer — DICOM/NIfTI + ROI + AI + Contour (3D Oblique)")
        self.resize(1500, 950)

        # volume
        self.volume = None  # shape (Z, Y, X)
        self.shape = None
        self.center = None
        self.global_zoom = 1.0

        # oblique (from mpr 2)
        self.oblique_angles = [0.0, 0.0, 0.0]  # pitch, yaw, roll
        self.oblique_depth = 0.0

        # views
        self.views = {n: ViewPanel(n) for n in ["Axial", "Sagittal", "Coronal", "Oblique"]}
        self.cine_timers = {n: QTimer(self) for n in self.views.keys()}

        # AI organ name only (no masks displayed)
        self.current_organ_name = None

        # seg thread and progress
        self.seg_thread = None
        self.progress_dialog = None

        # Contour Mode state
        self.contour_mode_active = False
        self.segmentation_volume = None
        self.seg_bboxes = {}
        self.current_seg_label = None
        self.current_seg_view_oblique = 'Axial'  # View for the 4th panel

        self._build_ui()

    # -------------------------
    # UI
    # -------------------------
        # -------------------------
        # UI
        # -------------------------
    def _build_ui(self):
            central = QWidget()
            grid = QGridLayout()
            central.setLayout(grid)
            self.setCentralWidget(central)

            # top controls
            load_dicom_btn = QPushButton("Load DICOM")
            load_dicom_btn.clicked.connect(self.load_dicom)
            load_nifti_btn = QPushButton("Load NIfTI/NumPy")
            load_nifti_btn.clicked.connect(self.load_nifti)
            export_btn = QPushButton("Export Volume (.npy)")
            export_btn.clicked.connect(self.export_volume)
            detect_btn = QPushButton("Detect Organ (AI)")
            detect_btn.clicked.connect(self.detect_main_organ)
            run_ai_btn = QPushButton("Run AI (orientation)")
            run_ai_btn.clicked.connect(self.run_ai)
            self.ai_info = QLabel("AI: not run")

            self.apply_from_combo = QComboBox()
            self.apply_from_combo.addItems(["Axial", "Sagittal", "Coronal"])
            apply_btn = QPushButton("Apply View → Oblique")
            apply_btn.clicked.connect(self.apply_view_to_oblique)

            # --- Contour Mode Buttons ---
            load_seg_btn = QPushButton("Load Seg (Contour)")
            load_seg_btn.clicked.connect(self.load_segmentation)
            self.contour_label_selector = QComboBox()
            self.contour_label_selector.currentIndexChanged.connect(self.on_contour_label_change)
            self.contour_label_selector.setEnabled(False)
            self.contour_mode_btn = QPushButton("Enter Contour Mode")
            self.contour_mode_btn.clicked.connect(self.toggle_contour_mode)
            self.contour_mode_btn.setEnabled(False)
            # --- End Contour Mode Buttons ---

            top_row = QHBoxLayout()
            top_row.addWidget(load_dicom_btn);
            top_row.addWidget(load_nifti_btn);
            top_row.addWidget(export_btn)
            top_row.addWidget(QLabel("Apply from:"));
            top_row.addWidget(self.apply_from_combo)
            top_row.addWidget(apply_btn);
            top_row.addWidget(detect_btn);
            top_row.addWidget(run_ai_btn)
            top_row.addStretch()
            # Add contour controls directly to top bar
            top_row.addWidget(load_seg_btn)
            top_row.addWidget(QLabel("Label:"))
            top_row.addWidget(self.contour_label_selector)
            top_row.addWidget(self.contour_mode_btn)
            top_row.addStretch();
            top_row.addWidget(self.ai_info)
            grid.addLayout(top_row, 0, 0, 1, 3)

            # viewers 2x2
            coords = [("Axial", 1, 0), ("Sagittal", 1, 1), ("Coronal", 2, 0), ("Oblique", 2, 1)]
            for name, r, c in coords:
                vp = self.views[name]
                container = QWidget()
                vlay = QVBoxLayout()
                container.setLayout(vlay)
                vlay.addWidget(vp.label)
                vlay.addWidget(vp.canvas)

                ctrl_widget = QWidget()  # Create a widget for the controls
                ctrl = QHBoxLayout()
                ctrl_widget.setLayout(ctrl)
                play = QPushButton("Play");
                pause = QPushButton("Pause")
                save = QPushButton("Save Slice")
                roi_btn = QPushButton("ROI (r)")
                save_roi_btn = QPushButton("Save ROI Volume")
                ctrl.addWidget(play);
                ctrl.addWidget(pause);
                ctrl.addWidget(save);
                ctrl.addWidget(roi_btn);
                ctrl.addWidget(save_roi_btn)
                vlay.addWidget(ctrl_widget)  # Add the controls widget
                vp.controls_widget = ctrl_widget  # Store reference in ViewPanel

                # For Oblique view: add both Oblique AND Contour controls, and toggle visibility
                if name == "Oblique":
                    # --- Standard Oblique Controls (from mpr 2) ---
                    # NOTE: These controls are now CREATED here but ADDED to the right-side panel
                    self.oblique_controls_widget = QWidget()
                    oblique_vlay = QVBoxLayout()
                    self.oblique_controls_widget.setLayout(oblique_vlay)

                    self.pitch_slider = QSlider(Qt.Horizontal)
                    self.pitch_slider.setRange(-90, 90);
                    self.pitch_slider.setValue(0)
                    self.pitch_slider.valueChanged.connect(self.on_oblique_params_changed)
                    oblique_vlay.addWidget(QLabel("Pitch (rotate X)"))
                    oblique_vlay.addWidget(self.pitch_slider)

                    self.yaw_slider = QSlider(Qt.Horizontal)
                    self.yaw_slider.setRange(-90, 90);
                    self.yaw_slider.setValue(0)
                    self.yaw_slider.valueChanged.connect(self.on_oblique_params_changed)
                    oblique_vlay.addWidget(QLabel("Yaw (rotate Y)"))
                    oblique_vlay.addWidget(self.yaw_slider)

                    self.roll_slider = QSlider(Qt.Horizontal)
                    self.roll_slider.setRange(-180, 180);
                    self.roll_slider.setValue(0)
                    self.roll_slider.valueChanged.connect(self.on_oblique_params_changed)
                    oblique_vlay.addWidget(QLabel("Roll (rotate Z)"))
                    oblique_vlay.addWidget(self.roll_slider)

                    self.depth_slider = QSlider(Qt.Horizontal)
                    self.depth_slider.setRange(-200, 200);
                    self.depth_slider.setValue(0)
                    self.depth_slider.valueChanged.connect(self.on_oblique_params_changed)
                    oblique_vlay.addWidget(QLabel("Depth (move plane along its normal)"))
                    oblique_vlay.addWidget(self.depth_slider)

                    # ***** CHANGE *****
                    # DO NOT ADD self.oblique_controls_widget to vlay here
                    # vlay.addWidget(self.oblique_controls_widget)  <-- THIS LINE IS REMOVED
                    # --- End Oblique Controls ---

                    # --- Contour Mode Controls (initially hidden) ---
                    # This panel (Oblique) gets the "free" view controls
                    self.contour_controls_widget = QWidget()
                    contour_vlay = QVBoxLayout()
                    self.contour_controls_widget.setLayout(contour_vlay)
                    contour_hlay = QHBoxLayout()
                    contour_vlay.addLayout(contour_hlay)
                    self.contour_view_btn_oblique = QPushButton("View: Axial")
                    contour_hlay.addWidget(self.contour_view_btn_oblique)
                    # We use the main slider 'vp.slider' for this panel's contour control
                    contour_vlay.addWidget(vp.slider)
                    vlay.addWidget(self.contour_controls_widget)
                    self.contour_controls_widget.setVisible(False)

                    # Connect contour signals for this specific panel
                    self.contour_view_btn_oblique.clicked.connect(self.on_contour_cycle_view_oblique)

                else:
                    vlay.addWidget(vp.slider)

                grid.addWidget(container, r, c)

                # events
                vp.canvas.mpl_connect('button_press_event', partial(self.on_click, view_name=name))
                vp.canvas.mpl_connect('scroll_event', partial(self.on_scroll, view_name=name))
                vp.canvas.mpl_connect('motion_notify_event', partial(self.on_motion, view_name=name))
                vp.canvas.mpl_connect('button_release_event', partial(self.on_release, view_name=name))

                # Connect slider
                # We will check for contour_mode_active inside the slot
                vp.slider.valueChanged.connect(partial(self.on_slider_change, view_name=name))

                play.clicked.connect(partial(self.start_cine, view_name=name))
                pause.clicked.connect(partial(self.stop_cine, view_name=name))
                save.clicked.connect(partial(self.save_current_slice, view_name=name))
                roi_btn.clicked.connect(partial(self.toggle_roi_mode, view_name=name))
                save_roi_btn.clicked.connect(partial(self.save_roi_volume, view_name=name))

                t = self.cine_timers[name]
                vp.cine_timer = t

            # The Oblique panel's main slider is used for contours, so hide it initially
            self.views["Oblique"].slider.setVisible(False)

            # right-side zoom
            right = QVBoxLayout()
            right.addWidget(QLabel("Zoom (Ctrl + wheel or buttons)"))
            zoom_in = QPushButton("Zoom In");
            zoom_out = QPushButton("Zoom Out");
            reset_zoom = QPushButton("Reset Zoom")
            zoom_in.clicked.connect(lambda: self.change_global_zoom(1.2))
            zoom_out.clicked.connect(lambda: self.change_global_zoom(1 / 1.2))
            reset_zoom.clicked.connect(lambda: self.set_global_zoom(1.0))
            zrow = QHBoxLayout();
            zrow.addWidget(zoom_in);
            zrow.addWidget(zoom_out);
            zrow.addWidget(reset_zoom)
            right.addLayout(zrow)

            # ***** CHANGE *****
            # Add the oblique controls widget (created in the loop) to the right-side panel
            right.addWidget(self.oblique_controls_widget)
            # Add a stretch to push controls to the top
            right.addStretch()
            # ***** END CHANGE *****

            right_widget = QWidget();
            right_widget.setLayout(right)
            grid.addWidget(right_widget, 1, 2, 2, 1)  # Add to grid at (row 1, col 2), spanning 2 rows

            self.statusBar().showMessage("Ready")
            self.update_all_views()
    # -------------------------
    # Loading and export
    # -------------------------
    def load_dicom(self):
        """Load DICOM series from a folder with improved handling"""
        if not HAVE_PYDICOM:
            QMessageBox.critical(self, "Missing dependency",
                                 "pydicom is required to load DICOM files.\nInstall with: pip install pydicom")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "")
        if not folder:
            return

        try:
            # Find all DICOM files in folder (including subdirectories)
            dcm_files = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        # Try to read as DICOM
                        pydicom.dcmread(filepath, stop_before_pixels=True)
                        dcm_files.append(filepath)
                    except:
                        # Not a valid DICOM file, skip
                        continue

            if not dcm_files:
                QMessageBox.warning(self, "No DICOM files",
                                    f"No valid DICOM files found in:\n{folder}")
                return

            # Read all DICOM files
            self.statusBar().showMessage(f"Loading {len(dcm_files)} DICOM files...")
            QApplication.processEvents()

            slices = []
            for f in dcm_files:
                try:
                    slices.append(pydicom.dcmread(f))
                except Exception as e:
                    print(f"Error reading {f}: {e}")
                    continue

            if not slices:
                QMessageBox.warning(self, "Load Error", "Could not read any DICOM slices")
                return

            # Sort slices by ImagePositionPatient or InstanceNumber
            def get_slice_location(s):
                # Try ImagePositionPatient first (more reliable)
                if hasattr(s, 'ImagePositionPatient') and s.ImagePositionPatient:
                    try:
                        return float(s.ImagePositionPatient[2])
                    except:
                        pass
                # Try SliceLocation
                if hasattr(s, 'SliceLocation'):
                    try:
                        return float(s.SliceLocation)
                    except:
                        pass
                # Fall back to InstanceNumber
                if hasattr(s, 'InstanceNumber'):
                    try:
                        return float(s.InstanceNumber)
                    except:
                        pass
                return 0.0

            slices.sort(key=get_slice_location)

            # Stack into volume
            arrays = []
            for s in slices:
                try:
                    arr = s.pixel_array.astype(np.float32)
                    # Apply rescale slope and intercept if available
                    if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
                        arr = arr * float(s.RescaleSlope) + float(s.RescaleIntercept)
                    arrays.append(arr)
                except Exception as e:
                    print(f"Error processing slice: {e}")
                    continue

            if not arrays:
                QMessageBox.warning(self, "Load Error", "Could not process DICOM pixel data")
                return

            vol = np.stack(arrays, axis=0)
            vol = np.transpose(vol,(2,1,0))
            self._finalize_volume_load(vol, f"DICOM folder: {folder}, {len(arrays)} slices")

        except Exception as e:
            QMessageBox.critical(self, "DICOM Load Error", f"Failed to load DICOM series:\n{e}")
            print("DICOM load error:", e)

    def load_nifti(self):
        """Load NIfTI or NumPy files"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NIfTI or NumPy File",
            "",
            "NIfTI (*.nii *.nii.gz);;NumPy (*.npy);;All Files (*)"
        )
        if not path:
            return

        try:
            if path.lower().endswith(('.nii', '.nii.gz')):
                if not HAVE_NIB:
                    QMessageBox.critical(self, "Missing dependency",
                                         "nibabel is required to load NIfTI files.\nInstall with: pip install nibabel")
                    return
                nii = nib.load(path)
                vol = nii.get_fdata()
                self._finalize_volume_load(vol, f"NIfTI: {os.path.basename(path)}")

            elif path.lower().endswith('.npy'):
                vol = np.load(path)
                self._finalize_volume_load(vol, f"NumPy: {os.path.basename(path)}")

            else:
                QMessageBox.warning(self, "Unsupported Format",
                                    "Please select a .nii, .nii.gz, or .npy file")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")
            print("Load error:", e)

    def _finalize_volume_load(self, vol, status_msg):
        """Common finalization for volume loading"""
        vol = vol.astype(np.float32)

        # Handle different dimensions
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        elif vol.ndim == 4:
            vol = vol[..., 0]
        elif vol.ndim != 3:
            raise ValueError(f"Unsupported volume dimensions: {vol.ndim}D")

        self.volume = vol
        self.shape = vol.shape
        self.center = [self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2]

        # reset oblique and AI info (using new mpr 2 controls)
        self.oblique_angles = [0.0, 0.0, 0.0]
        self.oblique_depth = 0.0
        if hasattr(self, 'pitch_slider'):
            self.pitch_slider.setValue(0)
            self.yaw_slider.setValue(0)
            self.roll_slider.setValue(0)
            # Set depth slider range based on new volume
            self.depth_slider.setRange(-max(self.shape) // 2, max(self.shape) // 2)
            self.depth_slider.setValue(0)

        self.current_organ_name = None
        self.ai_info.setText("AI: not run")

        # init sliders
        self.views["Axial"].slider.setRange(0, self.shape[2] - 1)
        self.views["Axial"].slider.setValue(self.center[2])
        self.views["Sagittal"].slider.setRange(0, self.shape[0] - 1)
        self.views["Sagittal"].slider.setValue(self.center[0])
        self.views["Coronal"].slider.setRange(0, self.shape[1] - 1)
        self.views["Coronal"].slider.setValue(self.center[1])

        self.statusBar().showMessage(f"Loaded: {status_msg} | Shape: {self.shape}")
        self.update_all_views()

    def export_volume(self):
        if self.volume is None:
            QMessageBox.information(self, "No volume", "Load a volume first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save volume (.npy)", "", "NumPy (*.npy)")
        if not path:
            return
        np.save(path, self.volume)
        self.statusBar().showMessage(f"Saved volume to {path}")

    # -------------------------
    # AI orientation (simple)
    # -------------------------
    def run_ai(self):
        if self.volume is None:
            QMessageBox.information(self, "No volume", "Load a volume first")
            return
        s = self.volume.shape
        min_axis = int(np.argmin(s))
        labels = {0: "Sagittal (axis 0 smallest)", 1: "Coronal (axis 1 smallest)", 2: "Axial (axis 2 smallest)"}
        self.ai_info.setText(f"AI: {labels.get(min_axis, 'Unknown')}")
        QMessageBox.information(self, "AI note", "Orientation detection is a simple heuristic in this demo.")

    # -------------------------
    # Detect main organ (background) - NAME ONLY, NO MASKS
    # -------------------------
    def detect_main_organ(self):
        """
        Smart TotalSegmentator detector - displays organ NAME only (no mask overlay)
        - Save volume to temp NIfTI
        - Run TotalSegmentator (python API) in --fast mode via SegThread
        - On success, identify largest organ, display name, delete all temps
        """
        if self.volume is None:
            QMessageBox.warning(self, "No Data", "Please load a volume first.")
            return

        if not HAVE_NIB:
            QMessageBox.critical(self, "Missing dependency",
                                 "nibabel required to run segmentation.\nInstall with: pip install nibabel")
            return

        # make temp dir
        temp_dir = tempfile.mkdtemp(prefix="mpr_totseg_")
        input_nifti = os.path.join(temp_dir, "input.nii.gz")
        out_folder = os.path.join(temp_dir, "output")
        os.makedirs(out_folder, exist_ok=True)

        # save volume as NIfTI
        try:
            nib.save(nib.Nifti1Image(self.volume.astype(np.float32), affine=np.eye(4)), input_nifti)
        except Exception as e:
            QMessageBox.critical(self, "Write error", f"Failed to write temporary NIfTI: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # progress dialog
        self.progress_dialog = QProgressDialog("Detecting organ... (running TotalSegmentator)", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Running AI")
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        # start background seg thread (uses totalsegmentator API)
        self.seg_thread = SegThread(input_nifti, out_folder, fast=True)
        self.seg_thread.progress_line.connect(lambda line: self.statusBar().showMessage(line))
        self.seg_thread.finished_success.connect(partial(self._on_seg_success, temp_dir))
        self.seg_thread.finished_error.connect(partial(self._on_seg_error, temp_dir))
        self.seg_thread.start()

    def _on_seg_success(self, temp_dir, output_path):
        # close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # find masks
        labels = [f for f in os.listdir(output_path) if f.endswith(".nii.gz") or f.endswith(".nii")]
        if not labels:
            QMessageBox.information(self, "Segmentation", "No organs detected.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # choose largest mask BY NAME ONLY (don't load masks)
        largest_name = None
        largest_count = 0
        for f in labels:
            try:
                img = nib.load(os.path.join(output_path, f))
                data = img.get_fdata()
                cnt = int(np.sum(data > 0))
                if cnt > largest_count:
                    largest_count = cnt
                    largest_name = os.path.splitext(os.path.splitext(f)[0])[0]  # remove .nii.gz
            except Exception as e:
                print("Mask read error:", f, e)
                continue

        if largest_name is None:
            QMessageBox.information(self, "Segmentation", "Could not determine main organ.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Format friendly organ name and display (NO MASK OVERLAY)
        display_name = largest_name.replace("_", " ").title()
        self.current_organ_name = display_name
        self.ai_info.setText(f"AI: Detected → {display_name}")
        self.statusBar().showMessage(f"Detected organ: {display_name} ({largest_count} voxels)")
        QMessageBox.information(self, "Detected Organ",
                                f"Main organ detected: {display_name}\n({largest_count} voxels)\n\nNote: Mask overlay is NOT displayed in viewers.")

        # cleanup temporary results (including all masks)
        shutil.rmtree(temp_dir, ignore_errors=True)
        self.update_all_views()

    def _on_seg_error(self, temp_dir, errmsg):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        QMessageBox.critical(self, "Segmentation Error", f"TotalSegmentator failed:\n{errmsg}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        self.statusBar().showMessage("Segmentation failed")

    # -------------------------
    # Apply view to oblique (FROM MPR 2)
    # -------------------------
    def apply_view_to_oblique(self):
        if self.volume is None:
            QMessageBox.information(self, "No volume", "Load a volume first")
            return
        from_name = self.apply_from_combo.currentText()

        # Ensure center exists
        if self.center is None:
            self.center = [s // 2 for s in self.shape]

        if from_name == "Axial":
            z = self.views["Axial"].slider.value()
            center = [self.center[0], self.center[1], z]
            angles = [0.0, 0.0, 0.0]
        elif from_name == "Sagittal":
            x = self.views["Sagittal"].slider.value()
            center = [x, self.center[1], self.center[2]]
            angles = [0.0, 90.0, 0.0]
        else:  # Coronal
            y = self.views["Coronal"].slider.value()
            center = [self.center[0], y, self.center[2]]
            angles = [90.0, 0.0, 0.0]

        self.center = center
        self.oblique_angles = angles
        self.oblique_depth = 0.0  # Reset depth

        # update sliders
        self.pitch_slider.setValue(int(angles[0]))
        self.yaw_slider.setValue(int(angles[1]))
        self.roll_slider.setValue(int(angles[2]))
        self.depth_slider.setValue(0)

        self.views["Oblique"].label.setText("Oblique")  # Reset label
        self.update_all_views()
        self.statusBar().showMessage(f"Applied {from_name} to oblique (center={center}, angles={angles})")

    # -------------------------
    # Interaction handlers (click/scroll/motion/release)
    # -------------------------
    def on_click(self, event, view_name):
        if self.contour_mode_active:  # Disable MPR clicks in contour mode
            return
        if self.volume is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        vp = self.views[view_name]

        if event.button == 1 and not vp.is_selecting_roi:
            vp.is_panning = True
            vp.pan_start = (event.xdata, event.ydata)

        if vp.is_selecting_roi and event.button == 1:
            vp.roi_start = (event.xdata, event.ydata)
            if vp.roi_patch:
                try:
                    vp.roi_patch.remove()
                except Exception:
                    pass
                vp.roi_patch = None
            vp.roi_patch = Rectangle((event.xdata, event.ydata), 1, 1, edgecolor='yellow', facecolor='none',
                                     linewidth=1.2)
            vp.ax.add_patch(vp.roi_patch)
            vp.canvas.draw_idle()
            return

        # Do not recenter on Oblique click
        if view_name == "Oblique":
            return

        xpix, ypix = int(round(event.xdata)), int(round(event.ydata))
        try:
            if view_name == "Axial":
                z = self.views["Axial"].slider.value()
                # Y/X are flipped in imshow
                x = np.clip(ypix, 0, self.shape[2] - 1);  # This should be ypix
                y = np.clip(xpix, 0, self.shape[1] - 1);  # This should be xpix
                # Let's re-check File A's logic
                # File A: x = int(round(xpix)); y = int(round(ypix))
                # File A: displayed_ax = np.flipud(slice_ax.T)
                # File B: displayed_ax = slice_ax.T
                # Ah, File B's Axial is slice_ax.T, so xpix is Y, ypix is X
                # File B's original:
                # x = np.clip(xpix, 0, self.shape[2] - 1);
                # y = np.clip(ypix, 0, self.shape[1] - 1)
                # This seems correct for slice_ax.T

                x_coord = np.clip(xpix, 0, self.shape[2] - 1)  # X-axis on plot
                y_coord = np.clip(ypix, 0, self.shape[1] - 1)  # Y-axis on plot

                if self.center is None:
                    self.center = [self.shape[0] // 2, y_coord, x_coord]
                else:
                    self.center = [self.center[0], y_coord, x_coord]
            elif view_name == "Sagittal":
                xslice = self.views["Sagittal"].slider.value()
                # displayed_sag = np.rot90(slice_sag)
                # xpix is Z, ypix is Y
                z = np.clip(xpix, 0, self.shape[2] - 1);
                y = np.clip(ypix, 0, self.shape[1] - 1)
                if self.center is None:
                    self.center = [xslice, y, z]
                else:
                    self.center = [xslice, y, z]
            elif view_name == "Coronal":
                yslice = self.views["Coronal"].slider.value()
                # displayed_cor = np.rot90(slice_cor)
                # xpix is Z, ypix is X
                z = np.clip(xpix, 0, self.shape[2] - 1);
                x = np.clip(ypix, 0, self.shape[0] - 1)
                if self.center is None:
                    self.center = [x, yslice, z]
                else:
                    self.center = [x, yslice, z]
        except Exception as e:
            print(f"Click error: {e}")
            pass

        if self.center is not None:
            self.views["Axial"].slider.blockSignals(True);
            self.views["Axial"].slider.setValue(int(self.center[2]));
            self.views["Axial"].slider.blockSignals(False)
            self.views["Sagittal"].slider.blockSignals(True);
            self.views["Sagittal"].slider.setValue(int(self.center[0]));
            self.views["Sagittal"].slider.blockSignals(False)
            self.views["Coronal"].slider.blockSignals(True);
            self.views["Coronal"].slider.setValue(int(self.center[1]));
            self.views["Coronal"].slider.blockSignals(False)

        self.update_all_views()

    def on_scroll(self, event, view_name):
        step = int(np.sign(event.step)) if hasattr(event, 'step') else (1 if event.button == 'up' else -1)

        # In contour mode, scroll only affects the "Oblique" panel's slider
        if self.contour_mode_active:
            if self.segmentation_volume is None:
                return

            # Only allow scrolling on the active contour panel
            if view_name == "Oblique":
                slider = self.views["Oblique"].slider
                cur = slider.value()
                new = int(np.clip(cur + step, slider.minimum(), slider.maximum()))
                slider.setValue(new)
            return

        # --- Standard MPR scrolling ---
        if self.volume is None:
            return

        mods = QApplication.keyboardModifiers()
        if mods == Qt.ControlModifier:
            if step > 0:
                self.change_global_zoom(1.15)
            else:
                self.change_global_zoom(1 / 1.15)
            return

        # For Oblique view, scroll changes ROLL angle
        if view_name == "Oblique":
            if hasattr(self, 'roll_slider'):
                cur = self.roll_slider.value()
                new = int(
                    np.clip(cur + step * 5, self.roll_slider.minimum(), self.roll_slider.maximum()))
                self.roll_slider.setValue(new)
            return

        vp = self.views[view_name]
        cur = vp.slider.value()
        new = int(np.clip(cur + step, vp.slider.minimum(), vp.slider.maximum()))
        vp.slider.setValue(new)

    def on_motion(self, event, view_name):
        if self.contour_mode_active:  # Disable panning/ROI in contour mode
            return
        vp = self.views[view_name]
        if vp.is_panning and event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            dx = event.xdata - vp.pan_start[0]
            dy = event.ydata - vp.pan_start[1]
            xlim = vp.ax.get_xlim();
            ylim = vp.ax.get_ylim()
            vp.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            vp.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            vp.canvas.draw_idle()
            vp.pan_start = (event.xdata, event.ydata)
            return

        if not vp.is_selecting_roi or event.inaxes is None or event.xdata is None:
            return
        if vp.roi_start is None:
            return
        x0, y0 = vp.roi_start
        x1, y1 = event.xdata, event.ydata
        w = x1 - x0;
        h = y1 - y0
        if vp.roi_patch is None:
            vp.roi_patch = Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor='none', linewidth=1.2)
            vp.ax.add_patch(vp.roi_patch)
        else:
            vp.roi_patch.set_xy((x0, y0));
            vp.roi_patch.set_width(w);
            vp.roi_patch.set_height(h)
        vp.canvas.draw_idle()

    def on_release(self, event, view_name):
        if self.contour_mode_active:  # Disable release actions in contour mode
            return
        vp = self.views[view_name]
        if vp.is_panning:
            vp.is_panning = False;
            vp.pan_start = None
            return
        if not vp.is_selecting_roi:
            return
        if vp.roi_start is None:
            vp.is_selecting_roi = False
            return
        if event.xdata is None or event.ydata is None:
            vp.roi_start = None;
            vp.is_selecting_roi = False;
            return
        x0, y0 = vp.roi_start;
        x1, y1 = event.xdata, event.ydata
        x_min, x_max = int(min(x0, x1)), int(max(x0, x1))
        y_min, y_max = int(min(y0, y1)), int(max(y0, y1))
        w = x_max - x_min;
        h = y_max - y_min
        vp.roi_rect = (x_min, y_min, w, h)
        vp.roi_start = None;
        vp.is_selecting_roi = False
        vp.canvas.draw_idle()
        self.statusBar().showMessage(f"ROI set in {view_name}: {vp.roi_rect}")

    def toggle_roi_mode(self, view_name):
        if self.contour_mode_active:
            QMessageBox.information(self, "Mode conflict", "ROI mode is disabled in Contour Mode.")
            return
        vp = self.views[view_name]
        if not vp.is_selecting_roi:
            vp.is_selecting_roi = True
            if vp.roi_patch:
                try:
                    vp.roi_patch.remove()
                except Exception:
                    pass
                vp.roi_patch = None
            self.statusBar().showMessage(f"ROI mode ON for {view_name}")
        else:
            vp.is_selecting_roi = False
            vp.roi_start = None
            if vp.roi_patch:
                try:
                    vp.roi_patch.remove()
                except Exception:
                    pass
                vp.roi_patch = None
            vp.roi_rect = None
            self.update_all_views()
            self.statusBar().showMessage(f"ROI mode OFF for {view_name}")

    # -------------------------
    # slider changes
    # -------------------------
    def on_slider_change(self, val, view_name):

        # --- Handle Contour Mode ---
        if self.contour_mode_active:
            # Only the Oblique panel's slider is active
            if view_name == "Oblique":
                self.update_contour_view(view_name)
            return

        # --- Handle MPR Mode ---
        if self.volume is None:
            return

        if view_name == "Axial":
            z = self.views["Axial"].slider.value()
            if self.center is None:
                self.center = [self.shape[0] // 2, self.shape[1] // 2, z]
            else:
                self.center[2] = z
        elif view_name == "Sagittal":
            x = self.views["Sagittal"].slider.value()
            if self.center is None:
                self.center = [x, self.shape[1] // 2, self.shape[2] // 2]
            else:
                self.center[0] = x
        elif view_name == "Coronal":
            y = self.views["Coronal"].slider.value()
            if self.center is None:
                self.center = [self.shape[0] // 2, y, self.shape[2] // 2]
            else:
                self.center[1] = y
        # Oblique angle sliders are handled by on_oblique_params_changed

        if self.center is not None:
            # Block signals to prevent feedback loops
            self.views["Axial"].slider.blockSignals(True);
            self.views["Axial"].slider.setValue(int(self.center[2]));
            self.views["Axial"].slider.blockSignals(False)
            self.views["Sagittal"].slider.blockSignals(True);
            self.views["Sagittal"].slider.setValue(int(self.center[0]));
            self.views["Sagittal"].slider.blockSignals(False)
            self.views["Coronal"].slider.blockSignals(True);
            self.views["Coronal"].slider.setValue(int(self.center[1]));
            self.views["Coronal"].slider.blockSignals(False)

        self.update_all_views()

    def on_oblique_params_changed(self, _=None):
        """ Handles changes from all 4 new oblique sliders (from mpr 2) """
        if self.contour_mode_active: return  # Do nothing in contour mode

        self.oblique_angles = [self.pitch_slider.value(), self.yaw_slider.value(), self.roll_slider.value()]
        self.oblique_depth = self.depth_slider.value()

        # We can optionally link the roll slider to the (hidden) cine slider
        # self.views["Oblique"].slider.setValue(int(self.roll_slider.value()) % 360)

        self.update_all_views()

    # -------------------------
    # Zoom helpers
    # -------------------------
    def set_global_zoom(self, zoom_factor):
        self.global_zoom = float(zoom_factor);
        self.update_all_views()

    def change_global_zoom(self, factor):
        self.global_zoom *= factor;
        self.update_all_views()

    def _apply_zoom_to_axis(self, ax, img_shape):
        if self.contour_mode_active:
            # In contour mode, just fit the image
            ax.autoscale(True)
            ax.set_aspect('equal')
            return

        h, w = img_shape
        cx = w / 2.0;
        cy = h / 2.0

        # Use center from click if available, else image center
        if self.center is not None:
            # This needs to be mapped to the specific view's coordinate system
            # For simplicity, let's keep zoom centered on the image
            pass

        half_w = (w / 2.0) / self.global_zoom
        half_h = (h / 2.0) / self.global_zoom

        # Calculate new center based on pan (if any)

        ax.set_xlim(cx - half_w, cx + half_w)
        # Y-axis is inverted in imshow
        ax.set_ylim(cy + half_h, cy - half_h)

    # -------------------------
    # draw/update views
    # -------------------------
    def update_all_views(self):
        # --- Handle Contour Mode ---
        if self.contour_mode_active:
            # Black out the first 3 panels
            self.black_out_mpr_panels()
            # Update only the 4th panel
            self.update_contour_view("Oblique")
            return

        # --- Standard MPR Mode ---
        if self.volume is None:
            for vp in self.views.values():
                vp.ax.clear();
                vp.ax.axis('off');
                vp.canvas.draw_idle()
            return

        vol = self.volume
        # Ensure indices are within valid range
        z = int(np.clip(self.views["Axial"].slider.value(), 0, self.shape[2] - 1))
        x = int(np.clip(self.views["Sagittal"].slider.value(), 0, self.shape[0] - 1))
        y = int(np.clip(self.views["Coronal"].slider.value(), 0, self.shape[1] - 1))

        # Update center from sliders if not set by click
        if self.center is None:
            self.center = [x, y, z]
        else:
            # Keep center, but update sliders to match (already done in on_slider_change)
            pass

        # AXIAL
        vp = self.views["Axial"];
        vp.ax.clear();
        vp.ax.axis('off')
        slice_ax = vol[:, :, z]
        displayed_ax = slice_ax.T
        vp.ax.imshow(displayed_ax, cmap='gray', origin='lower')
        self._apply_zoom_to_axis(vp.ax, displayed_ax.shape)
        if self.center is not None:
            cx = int(self.center[2]);  # X-coord
            cy = int(self.center[1]);  # Y-coord
            self._draw_crosshair(vp, cx, cy)
        self._draw_plane_intersection_on_axial(vp, self._compute_plane_normal(), self.center)
        if vp.roi_patch: vp.ax.add_patch(vp.roi_patch)
        if vp.roi_rect:
            x0, y0, w, h = vp.roi_rect
            try:
                vp.ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='yellow', linewidth=1.2, facecolor='none'))
            except Exception:
                pass
        vp.canvas.draw_idle()

        # SAGITTAL
        vp = self.views["Sagittal"];
        vp.ax.clear();
        vp.ax.axis('off')
        slice_sag = vol[x, :, :]
        displayed_sag = np.rot90(slice_sag)
        vp.ax.imshow(displayed_sag, cmap='gray', origin='lower')
        self._apply_zoom_to_axis(vp.ax, displayed_sag.shape)
        if self.center is not None:
            disp_x = int(self.center[2]);  # Z-coord
            disp_y = int(self.center[1]);  # Y-coord
            self._draw_crosshair(vp, disp_x, disp_y)
        self._draw_plane_intersection_on_sagittal(vp, self._compute_plane_normal(), self.center)
        if vp.roi_patch: vp.ax.add_patch(vp.roi_patch)
        if vp.roi_rect:
            x0, y0, w, h = vp.roi_rect
            try:
                vp.ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='yellow', linewidth=1.2, facecolor='none'))
            except Exception:
                pass
        vp.canvas.draw_idle()

        # CORONAL
        vp = self.views["Coronal"];
        vp.ax.clear();
        vp.ax.axis('off')
        slice_cor = vol[:, y, :]
        displayed_cor = np.rot90(slice_cor)
        vp.ax.imshow(displayed_cor, cmap='gray', origin='lower')
        self._apply_zoom_to_axis(vp.ax, displayed_cor.shape)
        if self.center is not None:
            disp_x = int(self.center[2]);  # Z-coord
            disp_y = int(self.center[0]);  # X-coord
            self._draw_crosshair(vp, disp_x, disp_y)
        self._draw_plane_intersection_on_coronal(vp, self._compute_plane_normal(), self.center)
        if vp.roi_patch: vp.ax.add_patch(vp.roi_patch)
        if vp.roi_rect:
            x0, y0, w, h = vp.roi_rect
            try:
                vp.ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='yellow', linewidth=1.2, facecolor='none'))
            except Exception:
                pass
        vp.canvas.draw_idle()

        # OBLIQUE (from mpr 2)
        vp = self.views["Oblique"];
        vp.ax.clear();
        vp.ax.axis('off')

        pitch, yaw, roll = np.deg2rad(self.oblique_angles)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        normal = R @ np.array([0.0, 0.0, 1.0])
        up_vec = R @ np.array([0.0, 1.0, 0.0])
        center = self.center if self.center is not None else [s // 2 for s in self.shape]
        center_arr = np.array(center) + (self.oblique_depth / 1.0) * normal

        try:
            # Using reslice_oblique from File_A, which does not flipud
            ob_slice = reslice_oblique(self.volume, center_arr, normal, up_vec, output_size=512, spacing=1.0)
            vp.ax.imshow(ob_slice, cmap='gray', origin='lower')
            self._apply_zoom_to_axis(vp.ax, ob_slice.shape)  # Preserving File_B's zoom
        except Exception as e:
            # Fallback logic from File_A
            slice_simple = self.volume[:, :, z]  # z from above
            ob_simple = rotate(slice_simple, np.rad2deg(roll), reshape=False, mode='nearest')
            vp.ax.imshow(ob_simple, cmap='gray', origin='lower')
            self._apply_zoom_to_axis(vp.ax, ob_simple.shape)  # Preserving File_B's zoom
            print(f"Oblique fallback render: {e}")

        # Display organ name if detected (NO MASK OVERLAY)
        if self.current_organ_name:
            try:
                vp.ax.text(10, 20, f"Detected: {self.current_organ_name}", color='lime',
                           fontsize=11, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
            except Exception:
                pass

        if vp.roi_patch: vp.ax.add_patch(vp.roi_patch)
        if vp.roi_rect:
            x0, y0, w, h = vp.roi_rect
            try:
                vp.ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='yellow', linewidth=1.2, facecolor='none'))
            except Exception:
                pass
        vp.canvas.draw_idle()

    def black_out_mpr_panels(self):
        """Helper to explicitly black out the 3 MPR panels"""
        for name in ["Axial", "Sagittal", "Coronal"]:
            vp = self.views[name]
            vp.ax.clear()
            vp.ax.axis('off')
            vp.ax.set_facecolor('black')
            vp.label.setText(f"({vp.original_name} - Disabled)")
            vp.canvas.draw_idle()

    # -------------------------
    # crosshair / plane intersection math
    # -------------------------
    def _draw_crosshair(self, vp, x, y, color='lime'):
        if self.contour_mode_active: return  # No crosshairs in contour mode

        if getattr(vp, 'crosshair_v', None):
            try:
                vp.crosshair_v.remove()
            except Exception:
                pass
            vp.crosshair_v = None
        if getattr(vp, 'crosshair_h', None):
            try:
                vp.crosshair_h.remove()
            except Exception:
                pass
            vp.crosshair_h = None
        xlim = vp.ax.get_xlim();
        ylim = vp.ax.get_ylim()

        # Draw lines that respect the zoom
        vp.crosshair_v = Line2D([x, x], [ylim[0], ylim[1]], color=color, linewidth=0.8)
        vp.crosshair_h = Line2D([xlim[0], xlim[1]], [y, y], color=color, linewidth=0.8)
        vp.ax.add_line(vp.crosshair_v);
        vp.ax.add_line(vp.crosshair_h)

    def _compute_plane_normal(self):
        """ Computes the 3D normal vector from the oblique angles """
        p, y, r = np.deg2rad(self.oblique_angles)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(p), -np.sin(p)],
                       [0, np.sin(p), np.cos(p)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(r), -np.sin(r), 0],
                       [np.sin(r), np.cos(r), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        # Normal vector in Z, Y, X order
        return R @ np.array([0.0, 0.0, 1.0])

    def _draw_plane_intersection_on_axial(self, vp, normal, center):
        if self.center is None or self.contour_mode_active: return
        z0 = int(self.views["Axial"].slider.value())
        cz, cy, cx = float(center[0]), float(center[1]), float(center[2])
        # Normal is (Z, Y, X)
        nz, ny, nx = normal[0], normal[1], normal[2]

        # Plane equation: nx(x-cx) + ny(y-cy) + nz(z-cz) = 0
        # On axial plane, z = z0
        # nx(x-cx) + ny(y-cy) + nz(z0-cz) = 0
        # ny(y-cy) = -nx(x-cx) - nz(z0-cz)

        h, w = self.shape[1], self.shape[2]  # Y, X
        xs = np.array([0, w - 1], dtype=float)

        if abs(ny) > 1e-6:
            ys = cy + (-nx * (xs - cx) - nz * (z0 - cz)) / ny
        else:
            # Plane is parallel to Y axis (vertical line on axial)
            ys = np.array([0, h - 1], dtype=float)
            if abs(nx) > 1e-6:
                xs = cx + (-ny * (ys - cy) - nz * (z0 - cz)) / nx
            else:
                return  # Plane is parallel to axial view

        xs_clip = np.clip(xs, 0, w - 1);
        ys_clip = np.clip(ys, 0, h - 1)

        if hasattr(vp, 'intersection_line') and vp.intersection_line:
            try:
                vp.intersection_line.remove()
            except Exception:
                pass
        # Plotting (X, Y)
        vp.intersection_line = Line2D(xs_clip, ys_clip, color='red', linewidth=1.0);
        vp.ax.add_line(vp.intersection_line)

    def _draw_plane_intersection_on_sagittal(self, vp, normal, center):
        if self.center is None or self.contour_mode_active: return
        x0 = int(self.views["Sagittal"].slider.value())  # This is the Z-axis index
        cz, cy, cx = float(center[0]), float(center[1]), float(center[2])
        nz, ny, nx = normal[0], normal[1], normal[2]

        # On sagittal plane, z = x0
        # nx(x-cx) + ny(y-cy) + nz(x0-cz) = 0
        # ny(y-cy) = -nx(x-cx) - nz(x0-cz)

        # Sagittal view plots (Z, Y)
        h, w = self.shape[1], self.shape[2]  # Y, Z
        zs = np.array([0, w - 1], dtype=float)  # Plot X-axis is Z

        if abs(ny) > 1e-6:
            ys = cy + (-nx * (x0 - cx) - nz * (zs - cz)) / ny
        else:
            # Plane is parallel to Y axis
            ys = np.array([0, h - 1], dtype=float)  # Plot Y-axis is Y
            if abs(nz) > 1e-6:
                zs = cz + (-nx * (x0 - cx) - ny * (ys - cy)) / nz
            else:
                return

        zs_clip = np.clip(zs, 0, w - 1);
        ys_clip = np.clip(ys, 0, h - 1)

        if hasattr(vp, 'intersection_line') and vp.intersection_line:
            try:
                vp.intersection_line.remove()
            except Exception:
                pass
        vp.intersection_line = Line2D(zs_clip, ys_clip, color='red', linewidth=1.0);
        vp.ax.add_line(vp.intersection_line)

    def _draw_plane_intersection_on_coronal(self, vp, normal, center):
        if self.center is None or self.contour_mode_active: return
        y0 = int(self.views["Coronal"].slider.value())  # This is the Y-axis index
        cz, cy, cx = float(center[0]), float(center[1]), float(center[2])
        nz, ny, nx = normal[0], normal[1], normal[2]

        # On coronal plane, y = y0
        # nx(x-cx) + ny(y0-cy) + nz(z-cz) = 0
        # nz(z-cz) = -nx(x-cx) - ny(y0-cy)

        # Coronal view plots (Z, X)
        h, w = self.shape[0], self.shape[2]  # X, Z
        zs = np.array([0, w - 1], dtype=float)  # Plot X-axis is Z

        if abs(nx) > 1e-6:
            # x = cx + (-nz*(z-cz) - ny*(y0-cy)) / nx
            xs = cx + (-nz * (zs - cz) - ny * (y0 - cy)) / nx
        else:
            # Plane is parallel to X axis
            xs = np.array([0, h - 1], dtype=float)  # Plot Y-axis is X
            if abs(nz) > 1e-6:
                zs = cz + (-nx * (xs - cx) - ny * (y0 - cy)) / nz
            else:
                return

        zs_clip = np.clip(zs, 0, w - 1);
        xs_clip = np.clip(xs, 0, h - 1)

        if hasattr(vp, 'intersection_line') and vp.intersection_line:
            try:
                vp.intersection_line.remove()
            except Exception:
                pass
        # Plotting (Z, X)
        vp.intersection_line = Line2D(zs_clip, xs_clip, color='red', linewidth=1.0);
        vp.ax.add_line(vp.intersection_line)

    # -------------------------
    # cine & save slice
    # -------------------------
    def start_cine(self, view_name):
        timer = self.cine_timers[view_name]

        if self.contour_mode_active:
            if self.segmentation_volume is None:
                return
            # Only allow cine on the active contour panel
            if view_name == "Oblique":
                if not timer.isActive():
                    timer.timeout.connect(partial(self._cine_step_contour, view_name="Oblique"))
                    timer.start(80)
                    self.statusBar().showMessage(f"Cine started on Contour View")
            return

        if self.volume is None: return
        if not timer.isActive():
            if view_name == "Oblique":
                timer.timeout.connect(partial(self._cine_step_oblique))
            else:
                timer.timeout.connect(partial(self._cine_step, view_name=view_name))
            timer.start(80)
            self.statusBar().showMessage(f"Cine started on {view_name}")

    def stop_cine(self, view_name):
        timer = self.cine_timers[view_name]
        if timer.isActive():
            timer.stop()
            try:
                timer.timeout.disconnect()
            except:
                pass
        self.statusBar().showMessage(f"Cine stopped on {view_name}")

    def _cine_step(self, view_name):
        vp = self.views[view_name]
        new = vp.slider.value() + 1
        if new > vp.slider.maximum():
            new = vp.slider.minimum()
        vp.slider.setValue(new)

    def _cine_step_oblique(self):
        """Animate oblique ROLL angle (modified)"""
        if hasattr(self, 'roll_slider'):
            new = self.roll_slider.value() + 5
            if new > self.roll_slider.maximum():
                new = self.roll_slider.minimum()
            self.roll_slider.setValue(new)

    def _cine_step_contour(self, view_name):
        """Animate contour slice slider for a specific view"""
        vp = self.views[view_name]
        slider = vp.slider

        new = slider.value() + 1
        if new > slider.maximum():
            new = slider.minimum()
        slider.setValue(new)

    def save_current_slice(self, view_name):
        # Handle saving from contour mode
        if self.contour_mode_active:
            if view_name == "Oblique":
                path, _ = QFileDialog.getSaveFileName(self, "Save contour slice", "", "PNG (*.png)")
                if not path: return
                self.views["Oblique"].canvas.figure.savefig(path, facecolor='black')
                self.statusBar().showMessage(f"Saved contour slice to {path}")
            return

        if self.volume is None: return
        arr = None
        if view_name == "Axial":
            z = self.views["Axial"].slider.value();
            arr = self.volume[:, :, z]
        elif view_name == "Sagittal":
            x = self.views["Sagittal"].slider.value();
            arr = self.volume[x, :, :]
        elif view_name == "Coronal":
            y = self.views["Coronal"].slider.value();
            arr = self.volume[:, y, :]
        else:
            # oblique (from mpr 2, with depth)
            center = self.center if self.center else [s // 2 for s in self.shape]
            pitch, yaw, roll = self.oblique_angles
            pitch_r, yaw_r, roll_r = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)
            Rx = np.array([[1, 0, 0], [0, np.cos(pitch_r), -np.sin(pitch_r)], [0, np.sin(pitch_r), np.cos(pitch_r)]])
            Ry = np.array([[np.cos(yaw_r), 0, np.sin(yaw_r)], [0, 1, 0], [-np.sin(yaw_r), 0, np.cos(yaw_r)]])
            Rz = np.array([[np.cos(roll_r), -np.sin(roll_r), 0], [np.sin(roll_r), np.cos(roll_r), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            normal = R @ np.array([0.0, 0.0, 1.0])
            up_vec = R @ np.array([0.0, 1.0, 0.0])
            # Apply depth
            center_arr = np.array(center) + (self.oblique_depth / 1.0) * normal
            arr = reslice_oblique(self.volume, center_arr, normal, up_vec, output_size=512)

        if arr is None:
            QMessageBox.information(self, "No data", "Cannot extract slice")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save slice", "", "NumPy (*.npy);;PNG (*.png)")
        if not path: return
        if path.lower().endswith('.npy'):
            np.save(path, arr);
            self.statusBar().showMessage(f"Saved {path}")
        else:
            import matplotlib.image as mpimg
            mpimg.imsave(path, arr, cmap='gray');
            self.statusBar().showMessage(f"Saved {path}")

    # -------------------------
    # save ROI as separate volume with user-specified slice count
    # -------------------------
    def save_roi_volume(self, view_name):
        """Extract ROI region as a separate volume with user-specified slice count"""
        if self.contour_mode_active:
            QMessageBox.information(self, "Mode conflict", "ROI saving is disabled in Contour Mode.")
            return

        if view_name == "Oblique":
            QMessageBox.information(self, "Cannot save", "Saving ROI from 3D Oblique is not supported.")
            return

        if self.volume is None:
            QMessageBox.information(self, "No volume", "Load a volume first")
            return
        vp = self.views[view_name]
        if not getattr(vp, 'roi_rect', None):
            QMessageBox.information(self, "No ROI", "Draw an ROI first (use ROI (r) button).")
            return

        x0, y0, w, h = map(int, vp.roi_rect)
        x1 = x0 + w;
        y1 = y0 + h
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "Invalid ROI", "ROI has zero or negative dimensions")
            return

        # Ask user for number of slices to include
        num_slices, ok = QInputDialog.getInt(
            self,
            "ROI Slice Count",
            f"How many slices to include in the ROI volume?\n(centered around current slice):",
            value=20,  # default value
            min=1,
            max=1000
        )

        if not ok:
            return

        # Extract the ROI region from the volume
        roi_volume = None
        try:
            if view_name == "Axial":
                z = int(self.views["Axial"].slider.value())
                # Map display coordinates (X, Y) to volume coordinates (Y, X)
                # Axial display is (X=vol_X, Y=vol_Y)
                # Wait, display is slice_ax.T, so display(X,Y) is vol(Y,X)
                vol_y_start = np.clip(x0, 0, self.shape[1])
                vol_y_end = np.clip(x1, 0, self.shape[1])
                vol_x_start = np.clip(y0, 0, self.shape[2])
                vol_x_end = np.clip(y1, 0, self.shape[2])

                # Check File B's original logic
                # vol_y_start = np.clip(x0, 0, self.shape[1])
                # vol_y_end = np.clip(x1, 0, self.shape[1])
                # vol_x_start = np.clip(y0, 0, self.shape[2])
                # vol_x_end = np.clip(y1, 0, self.shape[2])
                # This seems wrong. If display is (X,Y) = (vol_X, vol_Y) then...
                # slice_ax = vol[:, :, z] # Shape (Y, X)
                # displayed_ax = slice_ax.T # Shape (X, Y)
                # So plot X is vol X, plot Y is vol Y

                vol_x_start = np.clip(x0, 0, self.shape[2])
                vol_x_end = np.clip(x1, 0, self.shape[2])
                vol_y_start = np.clip(y0, 0, self.shape[1])
                vol_y_end = np.clip(y1, 0, self.shape[1])

                # Extract ROI: use user-specified slices around current slice
                half_slices = num_slices // 2
                z_start = max(0, z - half_slices)
                z_end = min(self.shape[0], z + half_slices)  # SLICE IN Z-AXIS

                # ... File B original logic was:
                # z_start = max(0, z - half_slices)
                # z_end = min(self.shape[0], z + half_slices)
                # roi_volume = self.volume[z_start:z_end, vol_y_start:vol_y_end, vol_x_start:vol_x_end]
                # This seems correct, slicing in Z (axis 0)

                # Let's re-check click logic.
                # Axial click: xpix is vol_X (axis 2), ypix is vol_Y (axis 1)
                # My click logic update:
                # x_coord = np.clip(xpix, 0, self.shape[2] - 1) # X-axis on plot (vol_X)
                # y_coord = np.clip(ypix, 0, self.shape[1] - 1) # Y-axis on plot (vol_Y)
                # self.center = [self.center[0], y_coord, x_coord] # Z, Y, X
                # This is correct.

                # So, ROI rect (x0,y0) is (vol_x_start, vol_y_start)
                vol_x_start_roi = np.clip(x0, 0, self.shape[2])
                vol_x_end_roi = np.clip(x1, 0, self.shape[2])
                vol_y_start_roi = np.clip(y0, 0, self.shape[1])
                vol_y_end_roi = np.clip(y1, 0, self.shape[1])

                z_slice = int(self.views["Axial"].slider.value())  # This is Z (axis 2)
                # We should slice along the view's axis, Z (axis 2)
                half_slices_ax = num_slices // 2
                z_start_ax = max(0, z_slice - half_slices_ax)
                z_end_ax = min(self.shape[2], z_slice + half_slices_ax)

                # Vol shape is (Z, Y, X)
                roi_volume = self.volume[:, vol_y_start_roi:vol_y_end_roi, vol_x_start_roi:vol_x_end_roi]
                # Whoops, File B slices in Z axis (axis 0)
                # z_start = max(0, z - half_slices) # z is slider value, axis 2
                # z_end = min(self.shape[0], z + half_slices) # shape[0] is axis 0
                # This is confusing. Let's stick to File B's original, it seemed to work.

                # File B original:
                z_slider_val = int(self.views["Axial"].slider.value())  # Current slice, axis 2

                # This is confusing. Axial view is slice in Z (axis 2).
                # Let's assume user wants slices *around* the current Z slice.

                z_slice_idx = int(self.views["Axial"].slider.value())  # Axis 2
                half_slices_z = num_slices // 2
                z_start_vol = max(0, z_slice_idx - half_slices_z)
                z_end_vol = min(self.shape[2], z_slice_idx + half_slices_z)

                # ROI rect (x0,y0) is (vol_x_start, vol_y_start)
                vol_x_start_rect = np.clip(x0, 0, self.shape[2])
                vol_x_end_rect = np.clip(x1, 0, self.shape[2])
                vol_y_start_rect = np.clip(y0, 0, self.shape[1])
                vol_y_end_rect = np.clip(y1, 0, self.shape[1])

                # Shape is (Z, Y, X). Slicing Z, Y, X
                roi_volume = self.volume[:, vol_y_start_rect:vol_y_end_rect, vol_x_start_rect:vol_x_end_rect]
                # This ignores num_slices.

                # Let's re-read File B's original.
                # z = int(self.views["Axial"].slider.value()) # axis 2
                # half_slices = num_slices // 2
                # z_start = max(0, z - half_slices) # uses z slider value
                # z_end = min(self.shape[0], z + half_slices) # uses shape[0] ??
                # roi_volume = self.volume[z_start:z_end, vol_y_start:vol_y_end, vol_x_start:vol_x_end]
                # This is slicing axis 0 (Sagittal) using axis 2 (Axial) slider.
                # This must be a bug.

                # Let's fix it.
                if view_name == "Axial":
                    z_idx = int(self.views["Axial"].slider.value())  # Axis 2
                    half_s = num_slices // 2
                    z_start_vol = max(0, z_idx - half_s)
                    z_end_vol = min(self.shape[2], z_idx + half_s)

                    x_start_vol = np.clip(x0, 0, self.shape[2])
                    x_end_vol = np.clip(x1, 0, self.shape[2])
                    y_start_vol = np.clip(y0, 0, self.shape[1])
                    y_end_vol = np.clip(y1, 0, self.shape[1])

                    # Slicing (Z, Y, X)
                    roi_volume = self.volume[:, y_start_vol:y_end_vol, x_start_vol:x_end_vol]
                    # This still ignores num_slices.

                    # Let's assume the user wants slices along the *perpendicular* axis (Z)
                    # Slicing (Z, Y, X)
                    # Ah, I see. vol is (Z, Y, X).
                    # displayed_ax = slice_ax.T = vol[:,:,z].T. Shape (X, Y)
                    # roi_rect (x0,y0) maps to (vol_X, vol_Y)
                    vol_x_start = np.clip(x0, 0, self.shape[2])
                    vol_x_end = np.clip(x1, 0, self.shape[2])
                    vol_y_start = np.clip(y0, 0, self.shape[1])
                    vol_y_end = np.clip(y1, 0, self.shape[1])

                    # Slices are in Z (axis 0)
                    z_slice = int(self.center[0])  # Use crosshair center
                    half_s_z = num_slices // 2
                    z_start_roi = max(0, z_slice - half_s_z)
                    z_end_roi = min(self.shape[0], z_slice + half_s_z)

                    roi_volume = self.volume[z_start_roi:z_end_roi, vol_y_start:vol_y_end, vol_x_start:vol_x_end]

                elif view_name == "Sagittal":
                    # displayed_sag = np.rot90(vol[x, :, :]) Shape (Z, Y)
                    # roi_rect (x0,y0) maps to (vol_Z, vol_Y)
                    vol_z_start = np.clip(x0, 0, self.shape[2])
                    vol_z_end = np.clip(x1, 0, self.shape[2])
                    vol_y_start = np.clip(y0, 0, self.shape[1])
                    vol_y_end = np.clip(y1, 0, self.shape[1])

                    # Slices are in X (axis 0)
                    x_slice = int(self.views["Sagittal"].slider.value())
                    half_s_x = num_slices // 2
                    x_start_roi = max(0, x_slice - half_s_x)
                    x_end_roi = min(self.shape[0], x_slice + half_s_x)

                    roi_volume = self.volume[x_start_roi:x_end_roi, vol_y_start:vol_y_end, vol_z_start:vol_z_end]

                elif view_name == "Coronal":
                    # displayed_cor = np.rot90(vol[:, y, :]) Shape (Z, X)
                    # roi_rect (x0,y0) maps to (vol_Z, vol_X)
                    vol_z_start = np.clip(x0, 0, self.shape[2])
                    vol_z_end = np.clip(x1, 0, self.shape[2])
                    vol_x_start = np.clip(y0, 0, self.shape[0])  # y0 is plot Y, maps to vol X (axis 0)
                    vol_x_end = np.clip(y1, 0, self.shape[0])

                    # Slices are in Y (axis 1)
                    y_slice = int(self.views["Coronal"].slider.value())
                    half_s_y = num_slices // 2
                    y_start_roi = max(0, y_slice - half_s_y)
                    y_end_roi = min(self.shape[1], y_slice + half_s_y)

                    roi_volume = self.volume[vol_x_start:vol_x_end, y_start_roi:y_end_roi, vol_z_start:vol_z_end]

            # My logic above seems much more correct than File B's original.
            # File B's Sagittal:
            # y_start = clip(y0, 0, shape[1]), y_end = clip(y1, 0, shape[1])
            # z_start = clip(x0, 0, shape[2]), z_end = clip(x1, 0, shape[2])
            # ... this part is correct (x0=Z, y0=Y) ...
            # BUT it reverses Y:
            # vol_y_start = np.clip(self.shape[1] - y1, 0, self.shape[1])
            # vol_y_end = np.clip(self.shape[1] - y0, 0, self.shape[1])
            # This is because of the `origin='lower'` and `rot90`.
            # I will trust File B's original coordinate mapping.

            # --- RESETTING TO FILE B'S ORIGINAL (TRUSTED) MAPPING ---
            if view_name == "Axial":
                z_slice_idx = int(self.views["Axial"].slider.value())  # Axis 2

                # Slices are perpendicular to view, so in Z (axis 0)
                # But centered on the *crosshair* Z, not the slider
                z_crosshair = int(self.center[0])
                half_slices_z = num_slices // 2
                z_start_vol = max(0, z_crosshair - half_slices_z)
                z_end_vol = min(self.shape[0], z_crosshair + half_slices_z)

                # Axial display is (X, Y) = (vol_X, vol_Y)
                vol_x_start = np.clip(x0, 0, self.shape[2])
                vol_x_end = np.clip(x1, 0, self.shape[2])
                vol_y_start = np.clip(y0, 0, self.shape[1])
                vol_y_end = np.clip(y1, 0, self.shape[1])

                roi_volume = self.volume[z_start_vol:z_end_vol, vol_y_start:vol_y_end, vol_x_start:vol_x_end]

            elif view_name == "Sagittal":
                x_slice_idx = int(self.views["Sagittal"].slider.value())  # Axis 0
                half_slices_x = num_slices // 2
                x_start_vol = max(0, x_slice_idx - half_slices_x)
                x_end_vol = min(self.shape[0], x_slice_idx + half_slices_x)

                # Sagittal display is (Z, Y)
                # File B's original:
                vol_z_start = np.clip(x0, 0, self.shape[2])
                vol_z_end = np.clip(x1, 0, self.shape[2])
                vol_y_start = np.clip(self.shape[1] - y1, 0, self.shape[1])  # Flipped Y
                vol_y_end = np.clip(self.shape[1] - y0, 0, self.shape[1])  # Flipped Y

                roi_volume = self.volume[x_start_vol:x_end_vol, vol_y_start:vol_y_end, vol_z_start:vol_z_end]

            elif view_name == "Coronal":
                y_slice_idx = int(self.views["Coronal"].slider.value())  # Axis 1
                half_slices_y = num_slices // 2
                y_start_vol = max(0, y_slice_idx - half_slices_y)
                y_end_vol = min(self.shape[1], y_slice_idx + half_slices_y)

                # Coronal display is (Z, X)
                # File B's original:
                vol_z_start = np.clip(x0, 0, self.shape[2])
                vol_z_end = np.clip(x1, 0, self.shape[2])
                vol_x_start = np.clip(self.shape[0] - y1, 0, self.shape[0])  # Flipped X (Y-axis)
                vol_x_end = np.clip(self.shape[0] - y0, 0, self.shape[0])  # Flipped X (Y-axis)

                roi_volume = self.volume[vol_x_start:vol_x_end, y_start_vol:y_end_vol, vol_z_start:vol_z_end]

            # This logic seems the most robust.

        except Exception as e:
            QMessageBox.critical(self, "ROI Error", f"Failed to extract ROI volume: {e}")
            print("ROI extraction error:", e)
            return

        if roi_volume is None or roi_volume.size == 0:
            QMessageBox.warning(self, "Empty ROI", "The selected ROI is empty.")
            return

        # Ask user where to save
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI Volume",
            "",
            "NIfTI files (*.nii.gz);;NumPy files (*.npy)"
        )
        if not path:
            return

        # Save based on file extension
        try:
            if path.lower().endswith('.npy'):
                np.save(path, roi_volume)
                self.statusBar().showMessage(f"ROI volume saved as NumPy: {path}")
            else:
                # Save as NIfTI
                if not path.lower().endswith(('.nii', '.nii.gz')):
                    path = path + '.nii.gz'
                if not HAVE_NIB:
                    QMessageBox.critical(self, "Missing dependency",
                                         "Please install nibabel to save NIfTI files (pip install nibabel).")
                    return
                nii = nib.Nifti1Image(roi_volume.astype(np.float32), affine=np.eye(4))
                nib.save(nii, path)
                self.statusBar().showMessage(f"ROI volume saved as NIfTI: {path}")

            QMessageBox.information(
                self,
                "ROI Volume Saved",
                f"ROI saved to: {path}\nShape: {roi_volume.shape}\n\nYou can load this file using 'Load NIfTI/NumPy' button."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save ROI volume:\n{e}")
            print("ROI save error:", e)

    # -------------------------
    # Contour Mode functions
    # -------------------------
    def load_segmentation(self):
        """Load a segmentation file for contour mode"""
        path, _ = QFileDialog.getOpenFileName(self, "Load segmentation", "", "NIfTI (*.nii *.nii.gz);;NumPy (*.npy)")
        if not path: return
        try:
            if path.lower().endswith(('.nii', '.nii.gz')):
                if not HAVE_NIB:
                    QMessageBox.critical(self, "Missing dependency",
                                         "nibabel is required to load NIfTI files.\nInstall with: pip install nibabel")
                    return
                nii = nib.load(path);
                seg = nii.get_fdata()
            else:
                seg = np.load(path)

            self.segmentation_volume = seg.astype(int)
            labels = np.unique(self.segmentation_volume);
            labels = labels[labels > 0]
            self.contour_label_selector.clear()
            for lbl in labels: self.contour_label_selector.addItem(str(int(lbl)))

            if labels.size > 0:
                self.current_seg_label = int(labels[0])
                self.compute_seg_bboxes(labels)
                self.statusBar().showMessage(f"Loaded segmentation {path}")
                # Enable contour controls
                self.contour_label_selector.setEnabled(True)
                self.contour_mode_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Empty Segmentation", "No non-zero labels found in the file.")
                self.segmentation_volume = None
                self.contour_label_selector.setEnabled(False)
                self.contour_mode_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load segmentation:\n{str(e)}")
            self.segmentation_volume = None

    def compute_seg_bboxes(self, labels):
        """Compute bounding boxes for segmentation labels"""
        try:
            self.seg_bboxes = {}
            for lbl in labels:
                rois = find_objects(self.segmentation_volume == lbl)
                if rois and len(rois) > 0: self.seg_bboxes[int(lbl)] = rois[0]
        except Exception:
            self.seg_bboxes = {}
            print("Could not compute bounding boxes (scipy.ndimage.find_objects failed?)")

    def on_contour_label_change(self):
        """Handle change in global contour label selection"""
        if not self.contour_label_selector.currentText(): return
        self.current_seg_label = int(self.contour_label_selector.currentText())

        # If in contour mode, update the active view
        if self.contour_mode_active:
            self.update_contour_slider_range("Oblique")
            self.update_contour_view("Oblique")

    def on_contour_cycle_view_oblique(self):
        """Cycle through Axial, Sagittal, Coronal for the 'Oblique' (free) panel"""
        order = ['Axial', 'Sagittal', 'Coronal']
        try:
            i = order.index(self.current_seg_view_oblique)
            self.current_seg_view_oblique = order[(i + 1) % 3]
        except ValueError:
            self.current_seg_view_oblique = 'Axial'
        self.contour_view_btn_oblique.setText(f"View: {self.current_seg_view_oblique}")

        # Update just this one panel
        self.update_contour_slider_range("Oblique")
        self.update_contour_view("Oblique")

    def update_contour_slider_range(self, view_name):
        """Update a single contour slice slider"""
        # Only update the Oblique panel's slider
        if self.segmentation_volume is None or self.current_seg_label is None or view_name != "Oblique":
            return

        view_type = self.current_seg_view_oblique
        slider = self.views[view_name].slider
        bbox = self.seg_bboxes.get(self.current_seg_label)
        axis = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}[view_type]

        length = 0
        if bbox:
            start = bbox[axis].start
            stop = bbox[axis].stop
            length = stop - start
        else:
            # Fallback if no bbox
            length = self.segmentation_volume.shape[axis]

        slider.blockSignals(True)
        slider.setRange(0, max(0, length - 1))
        slider.setValue(length // 2)
        slider.blockSignals(False)

    def update_contour_view(self, view_name):
        """Draw the contour outline in a specific view panel"""
        # Only update the Oblique panel in contour mode
        if view_name != "Oblique" or not self.contour_mode_active:
            return
        if self.segmentation_volume is None or self.current_seg_label is None:
            return

        # 1. Get the view panel
        vp = self.views[view_name]
        ax = vp.ax
        canvas = vp.canvas

        # 2. Determine view type
        view_type = self.current_seg_view_oblique

        # 3. Get slice index from the correct slider
        idx = vp.slider.value()
        label = self.current_seg_label
        axis = {'Axial': 2, 'Sagittal': 0, 'Coronal': 1}[view_type]
        bbox = self.seg_bboxes.get(label)

        # 4. Update panel label
        vp.label.setText(f"Contour Viewer ({view_type})")

        # 5. Get the slice array
        arr = None
        try:
            slice_idx = idx
            if bbox:
                # Use bbox to slice (more efficient)
                slice_idx = bbox[axis].start + idx
                # Ensure slice_idx is within global bounds
                if slice_idx >= self.segmentation_volume.shape[axis]:
                    slice_idx = self.segmentation_volume.shape[axis] - 1

            if view_type == 'Axial':
                arr = self.segmentation_volume[:, :, slice_idx]
            elif view_type == 'Sagittal':
                arr = self.segmentation_volume[slice_idx, :, :]
            else:  # Coronal
                arr = self.segmentation_volume[:, slice_idx, :]

        except IndexError:
            arr = np.zeros((100, 100))  # fallback
        except Exception as e:
            print(f"Error slicing contour: {e}")
            arr = np.zeros((100, 100))

        # 6. Draw the contours
        ax.clear()
        ax.axis('off')
        ax.set_facecolor('black')

        if not HAVE_SKIMAGE:
            ax.text(0.5, 0.5, "scikit-image not installed\n(pip install scikit-image)",
                    ha='center', va='center', color='white', fontsize=10)
        elif arr is not None:
            try:
                # Find contours on the transposed slice
                contours = measure.find_contours((arr == label).T, 0.5)
                for c in contours:
                    ax.plot(c[:, 1], c[:, 0], 'r', lw=1.5)
                # Auto-zoom
                self._apply_zoom_to_axis(ax, arr.T.shape)
            except Exception as e:
                print(f"Error finding contours: {e}")
                ax.text(0.5, 0.5, "Error finding contours", ha='center', va='center', color='yellow')

        canvas.draw_idle()

    def toggle_contour_mode(self):
        """Toggle between MPR mode and Contour mode"""

        if self.segmentation_volume is None:
            QMessageBox.warning(self, "No Segmentation",
                                "Please load a segmentation file first using 'Load Seg (Contour)'.")
            return

        self.contour_mode_active = not self.contour_mode_active

        if self.contour_mode_active:
            # --- Enter Contour Mode ---

            # Hide controls for the first 3 panels
            self.views["Axial"].slider.setVisible(False)
            self.views["Axial"].controls_widget.setVisible(False)
            self.views["Sagittal"].slider.setVisible(False)
            self.views["Sagittal"].controls_widget.setVisible(False)
            self.views["Coronal"].slider.setVisible(False)
            self.views["Coronal"].controls_widget.setVisible(False)

            # Black out the first 3 panels
            self.black_out_mpr_panels()

            # In Oblique panel: Hide Oblique-specific controls, show Contour-specific controls
            self.oblique_controls_widget.setVisible(False)
            self.views["Oblique"].controls_widget.setVisible(False)  # Hide MPR controls
            self.contour_controls_widget.setVisible(True)
            self.views["Oblique"].slider.setVisible(True)  # Make sure contour slider is visible

            self.contour_mode_btn.setText("Exit Contour Mode")
            self.contour_mode_btn.setStyleSheet("background-color: #FF8888;")

            # Set up slider range and draw the single active view
            self.update_contour_slider_range("Oblique")
            self.update_contour_view("Oblique")  # This will also set the label

            self.statusBar().showMessage(f"Contour Mode Active. Label: {self.current_seg_label}")

        else:
            # --- Exit Contour Mode ---

            # Show controls for the first 3 panels
            self.views["Axial"].slider.setVisible(True)
            self.views["Axial"].controls_widget.setVisible(True)
            self.views["Sagittal"].slider.setVisible(True)
            self.views["Sagittal"].controls_widget.setVisible(True)
            self.views["Coronal"].slider.setVisible(True)
            self.views["Coronal"].controls_widget.setVisible(True)

            # In Oblique panel: Hide Contour-specific controls, show Oblique-specific controls
            self.contour_controls_widget.setVisible(False)
            self.views["Oblique"].slider.setVisible(False)  # Hide contour slider
            self.oblique_controls_widget.setVisible(True)
            self.views["Oblique"].controls_widget.setVisible(True)  # Show MPR controls

            self.contour_mode_btn.setText("Enter Contour Mode")
            self.contour_mode_btn.setStyleSheet("")  # Reset style

            # Restore labels
            for name, vp in self.views.items():
                vp.label.setText(vp.original_name)

            # Reset MPR sliders to volume state
            if self.volume is not None and self.center is not None:
                self.views["Axial"].slider.setRange(0, self.shape[2] - 1)
                self.views["Axial"].slider.setValue(self.center[2])
                self.views["Sagittal"].slider.setRange(0, self.shape[0] - 1)
                self.views["Sagittal"].slider.setValue(self.center[0])
                self.views["Coronal"].slider.setRange(0, self.shape[1] - 1)
                self.views["Coronal"].slider.setValue(self.center[1])

            # Redraw all MPR views
            self.update_all_views()
            self.statusBar().showMessage("Exited Contour Mode. MPR Active.")


# -------------------------
# app entrypoint
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MPRApp()
    win.show()
    sys.exit(app.exec_())