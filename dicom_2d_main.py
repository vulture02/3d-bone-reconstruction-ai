#!/usr/bin/env python3
"""
Enhanced 2D DICOM Analysis Tool with 3D Connection
This file handles 2D visualization and connects to the second file for 3D reconstruction
"""

import os
import sys
import numpy as np
import pydicom
from skimage import measure, filters, morphology, exposure
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import traceback
from pathlib import Path
import subprocess
import importlib.util


# Import the second file for 3D functionality
def import_3d_module():
    """Import the 3D reconstruction module"""
    try:
        # Try to import the second file (assuming it's in the same directory)
        spec = importlib.util.spec_from_file_location("dicom_3d", "paste-2.py")
        if spec is None:
            # If paste-2.py doesn't exist, try other common names
            for filename in ["dicom_3d.py", "3d_reconstruction.py", "paste-2.txt"]:
                if os.path.exists(filename):
                    spec = importlib.util.spec_from_file_location("dicom_3d", filename)
                    break

        if spec is not None:
            dicom_3d_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dicom_3d_module)
            return dicom_3d_module
        else:
            print("Warning: 3D reconstruction module not found")
            return None
    except Exception as e:
        print(f"Warning: Could not import 3D module: {e}")
        return None


# Import 3D module at startup
DICOM_3D_MODULE = import_3d_module()


class EnhancedDicom2D:
    def __init__(self, dicom_dir=None):
        """Initialize with the directory containing DICOM files"""
        self.dicom_dir = dicom_dir
        self.slices = []
        self.original_volume = None
        self.volume = None
        self.spacing = [1.0, 1.0, 1.0]

        # Enhanced HU thresholds for different bone types
        self.cortical_bone_lower = 400  # Dense cortical bone
        self.cortical_bone_upper = 3000
        self.trabecular_bone_lower = 150  # Spongy/trabecular bone
        self.trabecular_bone_upper = 400
        self.soft_tissue_upper = 100

        # Bone identification data
        self.bone_regions = []
        self.identified_bones = {}

        # GUI components
        self.root = None
        self.info_frame = None
        self.canvas_frame = None
        self.progress_var = None
        self.status_var = None

        # 3D module connection
        self.dicom_3d_converter = None

    def select_dicom_directory(self):
        """GUI method to select DICOM directory"""
        directory = filedialog.askdirectory(
            title="Select DICOM Directory",
            initialdir=os.getcwd()
        )
        if directory:
            self.dicom_dir = directory
            return True
        return False

    def load_dicom_series(self):
        """Load all DICOM files from the specified directory with better error handling"""
        if not self.dicom_dir or not os.path.exists(self.dicom_dir):
            raise ValueError(f"Invalid DICOM directory: {self.dicom_dir}")

        print(f"Loading DICOM files from {self.dicom_dir}...")

        # Get list of all potential DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(self.dicom_dir):
            for f in files:
                file_path = os.path.join(root, f)
                # Try to identify DICOM files by extension or by reading
                if (f.lower().endswith(('.dcm', '.dic', '.dicom', '.ima')) or
                        self.is_dicom_file(file_path)):
                    dicom_files.append(file_path)

        if not dicom_files:
            # Try all files in directory
            for f in os.listdir(self.dicom_dir):
                file_path = os.path.join(self.dicom_dir, f)
                if os.path.isfile(file_path) and self.is_dicom_file(file_path):
                    dicom_files.append(file_path)

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {self.dicom_dir}")

        print(f"Found {len(dicom_files)} potential DICOM files.")

        # Load each DICOM file
        valid_slices = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                    # Basic validation
                    if ds.pixel_array.size > 0:
                        valid_slices.append(ds)
                        print(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error reading {os.path.basename(file_path)}: {e}")
                continue

        if not valid_slices:
            raise ValueError("No valid DICOM slices were loaded.")

        # Sort slices by instance number or slice location
        try:
            valid_slices.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        except:
            try:
                valid_slices.sort(key=lambda x: float(x.SliceLocation) if hasattr(x, 'SliceLocation') else 0)
            except:
                print("Warning: Could not sort slices by position")

        self.slices = valid_slices
        print(f"Successfully loaded {len(self.slices)} valid DICOM slices.")

    def is_dicom_file(self, filepath):
        """Check if a file is a DICOM file"""
        try:
            with open(filepath, 'rb') as f:
                # Check for DICOM magic number
                f.seek(128)
                magic = f.read(4)
                return magic == b'DICM'
        except:
            return False

    def process_volume(self):
        """Convert DICOM slices to a 3D volume with proper Hounsfield Units"""
        print("Processing DICOM slices into 3D volume...")

        if not self.slices:
            raise ValueError("No DICOM slices available")

        # Get slice dimensions
        first_slice = self.slices[0]
        img_shape = first_slice.pixel_array.shape

        print(f"Slice dimensions: {img_shape}")
        print(f"Number of slices: {len(self.slices)}")

        # Verify all slices have same dimensions
        for i, slice_data in enumerate(self.slices):
            if slice_data.pixel_array.shape != img_shape:
                print(f"Warning: Slice {i} has different dimensions: {slice_data.pixel_array.shape}")

        # Create 3D volume array
        self.original_volume = np.zeros((img_shape[0], img_shape[1], len(self.slices)), dtype=np.float32)

        # Get pixel spacing information
        try:
            pixel_spacing = first_slice.PixelSpacing if hasattr(first_slice, 'PixelSpacing') else [1.0, 1.0]
            slice_thickness = first_slice.SliceThickness if hasattr(first_slice, 'SliceThickness') else 1.0
            self.spacing = [float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)]
        except:
            self.spacing = [1.0, 1.0, 1.0]
            print("Warning: Using default spacing [1.0, 1.0, 1.0]")

        print(f"Volume spacing: {self.spacing}")

        # Fill the 3D volume with slice data and convert to Hounsfield Units
        for i, slice_data in enumerate(self.slices):
            try:
                pixel_array = slice_data.pixel_array.astype(np.float32)

                # Ensure consistent dimensions
                if pixel_array.shape != img_shape:
                    print(f"Resizing slice {i} from {pixel_array.shape} to {img_shape}")
                    pixel_array = cv2.resize(pixel_array, (img_shape[1], img_shape[0]))

                # Convert to Hounsfield Units (HU)
                if hasattr(slice_data, 'RescaleIntercept') and hasattr(slice_data, 'RescaleSlope'):
                    intercept = float(slice_data.RescaleIntercept)
                    slope = float(slice_data.RescaleSlope)
                    hu_image = pixel_array * slope + intercept
                else:
                    hu_image = pixel_array

                self.original_volume[:, :, i] = hu_image

            except Exception as e:
                print(f"Error processing slice {i}: {e}")
                # Use zeros for problematic slices
                self.original_volume[:, :, i] = np.zeros(img_shape, dtype=np.float32)

        print(f"Volume dimensions: {self.original_volume.shape}")
        print(f"Volume value range: [{np.min(self.original_volume):.1f}, {np.max(self.original_volume):.1f}]")

        # Apply enhanced filtering
        print("Applying image enhancement...")
        self.original_volume = self.enhance_image_quality(self.original_volume)

    def enhance_image_quality(self, volume):
        """Apply advanced image enhancement techniques with error handling"""
        try:
            enhanced_volume = np.zeros_like(volume)

            for i in range(volume.shape[2]):
                slice_img = volume[:, :, i]

                # Skip empty slices
                if np.max(slice_img) - np.min(slice_img) < 1:
                    enhanced_volume[:, :, i] = slice_img
                    continue

                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                try:
                    # Normalize to 0-255 range
                    slice_normalized = exposure.rescale_intensity(
                        slice_img, out_range=(0, 255)
                    ).astype(np.uint8)

                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced_slice = clahe.apply(slice_normalized)

                    # Convert back to original intensity range
                    enhanced_volume[:, :, i] = exposure.rescale_intensity(
                        enhanced_slice, out_range=(slice_img.min(), slice_img.max())
                    )
                except Exception as e:
                    print(f"CLAHE failed for slice {i}: {e}")
                    enhanced_volume[:, :, i] = slice_img

            # Apply 3D Gaussian filter for overall smoothing
            try:
                enhanced_volume = ndimage.gaussian_filter(enhanced_volume, sigma=0.5)
            except Exception as e:
                print(f"Gaussian filter failed: {e}")

            return enhanced_volume

        except Exception as e:
            print(f"Image enhancement failed: {e}")
            return volume

    def identify_bone_structures(self):
        """Enhanced bone structure identification with better segmentation"""
        print("Identifying bone structures with advanced segmentation...")

        try:
            # Multi-level bone segmentation for better accuracy
            print("Creating multi-level bone masks...")

            # Primary bone mask (conservative threshold for clear bone)
            primary_bone_mask = self.original_volume >= 300

            # Secondary bone mask (includes trabecular bone)
            secondary_bone_mask = np.logical_and(
                self.original_volume >= 150,
                self.original_volume < 300
            )

            # Apply morphological operations to clean up masks
            print("Applying morphological operations...")

            # Clean primary bone mask
            structure_3d = ndimage.generate_binary_structure(3, 2)
            primary_bone_mask = ndimage.binary_closing(primary_bone_mask, structure=structure_3d, iterations=2)
            primary_bone_mask = ndimage.binary_fill_holes(primary_bone_mask)
            primary_bone_mask = morphology.remove_small_objects(primary_bone_mask, min_size=200)

            # Clean secondary bone mask
            secondary_bone_mask = ndimage.binary_closing(secondary_bone_mask, structure=structure_3d, iterations=1)
            secondary_bone_mask = morphology.remove_small_objects(secondary_bone_mask, min_size=100)

            # Combine masks intelligently
            print("Combining bone masks...")
            combined_mask = np.logical_or(primary_bone_mask, secondary_bone_mask)

            # Apply 3D opening to separate connected components
            combined_mask = ndimage.binary_opening(combined_mask, structure=structure_3d, iterations=1)

            # Final cleanup
            combined_mask = ndimage.binary_fill_holes(combined_mask)
            combined_mask = morphology.remove_small_objects(combined_mask, min_size=500)

            # Apply 3D Gaussian smoothing to the volume for better surface extraction
            print("Smoothing volume for surface extraction...")
            smoothed_volume = ndimage.gaussian_filter(self.original_volume.astype(np.float32), sigma=0.8)

            # Create high-quality bone volume for 2D visualization
            bone_volume = np.zeros_like(smoothed_volume)
            bone_volume[combined_mask] = smoothed_volume[combined_mask]

            # Normalize and enhance the bone volume
            bone_volume = np.clip(bone_volume, 150, 2000)  # Clamp to bone HU range
            bone_volume = ((bone_volume - 150) / (2000 - 150) * 255).astype(np.uint8)

            # Label connected components
            labels, num_labels = ndimage.label(combined_mask)
            print(f"Found {num_labels} bone components after cleaning")

            # Analyze each bone region
            self.bone_regions = []
            for i in range(1, min(num_labels + 1, 15)):  # Limit for performance
                region_mask = (labels == i)
                region_size = np.sum(region_mask)

                if region_size > 1000:  # Only significant regions
                    try:
                        z_indices, y_indices, x_indices = np.where(region_mask)

                        centroid = [np.mean(z_indices), np.mean(y_indices), np.mean(x_indices)]
                        bbox = [
                            np.min(z_indices), np.max(z_indices),
                            np.min(y_indices), np.max(y_indices),
                            np.min(x_indices), np.max(x_indices)
                        ]

                        # Analyze bone density distribution
                        region_hu_values = self.original_volume[region_mask]
                        cortical_percentage = np.sum(region_hu_values >= 400) / len(region_hu_values) * 100

                        bone_info = {
                            'id': i,
                            'size': region_size,
                            'centroid': centroid,
                            'bbox': bbox,
                            'cortical_percentage': cortical_percentage,
                            'mean_hu': np.mean(region_hu_values),
                            'anatomical_region': self.determine_anatomical_region(centroid, bbox),
                            'bone_type': self.classify_bone_type(region_size, cortical_percentage, bbox)
                        }

                        self.bone_regions.append(bone_info)
                    except Exception as e:
                        print(f"Error analyzing region {i}: {e}")

            # Sort by size (largest first)
            self.bone_regions.sort(key=lambda x: x['size'], reverse=True)

            # Store the high-quality volume for 2D visualization
            self.volume = bone_volume

            print(f"Successfully identified {len(self.bone_regions)} significant bone structures")
            for i, bone in enumerate(self.bone_regions[:5]):
                print(f"  {i + 1}. {bone['bone_type']} in {bone['anatomical_region']}")
                print(f"     Size: {bone['size']} voxels, Cortical: {bone['cortical_percentage']:.1f}%")

        except Exception as e:
            print(f"Error in bone identification: {e}")
            # Enhanced fallback
            try:
                simple_mask = self.original_volume > 250
                simple_mask = ndimage.binary_fill_holes(simple_mask)
                simple_mask = morphology.remove_small_objects(simple_mask, min_size=1000)
                self.volume = simple_mask.astype(np.uint8) * 255
            except:
                self.volume = (self.original_volume > 200).astype(np.uint8) * 255
            self.bone_regions = []

    def determine_anatomical_region(self, centroid, bbox):
        """Determine anatomical region based on position and shape"""
        try:
            z, y, x = centroid
            height = bbox[1] - bbox[0]
            width = max(bbox[3] - bbox[2], bbox[5] - bbox[4])

            # Simple anatomical classification based on position and size
            if z < self.original_volume.shape[2] * 0.3:
                if height > width * 1.5:
                    return "Upper Extremity (Long Bone)"
                else:
                    return "Upper Extremity (Short/Irregular Bone)"
            elif z > self.original_volume.shape[2] * 0.7:
                if height > width * 1.5:
                    return "Lower Extremity (Long Bone)"
                else:
                    return "Lower Extremity (Short/Irregular Bone)"
            else:
                if width > height:
                    return "Axial Skeleton (Vertebra/Rib)"
                else:
                    return "Central Region (Pelvis/Spine)"
        except:
            return "Unclassified Region"

    def classify_bone_type(self, size, cortical_percentage, bbox):
        """Classify bone type based on characteristics"""
        try:
            height = bbox[1] - bbox[0]
            width = max(bbox[3] - bbox[2], bbox[5] - bbox[4])
            aspect_ratio = height / width if width > 0 else 1

            if size > 50000:  # Large bone
                if aspect_ratio > 3:
                    return "Long Bone (Femur/Tibia/Humerus)"
                elif cortical_percentage > 60:
                    return "Flat Bone (Skull/Pelvis/Scapula)"
                else:
                    return "Irregular Bone (Vertebra/Pelvis)"
            elif size > 10000:  # Medium bone
                if aspect_ratio > 2:
                    return "Long Bone (Radius/Ulna/Fibula)"
                else:
                    return "Short Bone (Carpal/Tarsal)"
            else:  # Small bone
                return "Small Bone (Phalanx/Metacarpal)"
        except:
            return "Unclassified Bone"

    def create_main_window(self):
        """Create the main application window"""
        self.root = tk.Tk()
        self.root.title("Enhanced DICOM 2D Analysis with 3D Connection")
        self.root.geometry("1400x900")

        # Status variables
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")

        # Create menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select DICOM Directory", command=self.select_and_process)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(5, 0))

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Information panel
        self.info_frame = ttk.LabelFrame(content_frame, text="Analysis Results", padding=10)
        self.info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Right side - Visualization
        self.canvas_frame = ttk.LabelFrame(content_frame, text="2D Views & Controls", padding=10)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Initial content
        ttk.Label(self.info_frame, text="Select a DICOM directory to begin analysis",
                  font=('Arial', 12)).pack(pady=20)

        ttk.Button(self.canvas_frame, text="Select DICOM Directory",
                   command=self.select_and_process).pack(pady=20)

    def select_and_process(self):
        """Select directory and process in background thread"""
        if self.select_dicom_directory():
            # Disable UI during processing
            self.progress_var.set(0)
            self.status_var.set("Processing DICOM files...")

            # Start processing in background thread
            processing_thread = threading.Thread(target=self.process_dicom_data)
            processing_thread.daemon = True
            processing_thread.start()

    def process_dicom_data(self):
        """Process DICOM data in background thread"""
        try:
            self.progress_var.set(20)
            self.status_var.set("Loading DICOM files...")
            self.load_dicom_series()

            self.progress_var.set(40)
            self.status_var.set("Processing volume...")
            self.process_volume()

            self.progress_var.set(60)
            self.status_var.set("Identifying bone structures...")
            self.identify_bone_structures()

            self.progress_var.set(80)
            self.status_var.set("Creating visualizations...")

            # Initialize 3D converter connection
            self.initialize_3d_connection()

            # Update GUI in main thread
            self.root.after(100, self.update_gui_after_processing)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.root.after(100, lambda: self.show_error(error_msg))

    def initialize_3d_connection(self):
        """Initialize connection to 3D module"""
        try:
            if DICOM_3D_MODULE is not None:
                print("Initializing 3D converter connection...")
                self.dicom_3d_converter = DICOM_3D_MODULE.DicomTo3D(self.dicom_dir)
                print("3D connection established successfully")
            else:
                print("3D module not available")
                self.dicom_3d_converter = None
        except Exception as e:
            print(f"Failed to initialize 3D connection: {e}")
            self.dicom_3d_converter = None

    def update_gui_after_processing(self):
        """Update GUI after processing is complete"""
        try:
            # Clear existing content
            for widget in self.info_frame.winfo_children():
                widget.destroy()
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            # Update information panel
            self.populate_bone_info()

            # Create visualization
            self.create_visualization_panel()

            self.progress_var.set(100)
            self.status_var.set("Analysis complete")

        except Exception as e:
            self.show_error(f"GUI update error: {str(e)}")

    def show_error(self, message):
        """Show error message"""
        self.status_var.set("Error occurred")
        self.progress_var.set(0)
        messagebox.showerror("Error", message)

    def populate_bone_info(self):
        """Populate the bone information panel with improved layout"""
        # Title
        title_label = ttk.Label(self.info_frame, text="BONE ANALYSIS RESULTS",
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))

        # Summary frame
        summary_frame = ttk.LabelFrame(self.info_frame, text="Summary", padding=5)
        summary_frame.pack(fill=tk.X, pady=(0, 10))

        total_volume = sum(bone['size'] for bone in self.bone_regions) if self.bone_regions else 0

        summary_info = [
            f"Volume Dimensions: {self.original_volume.shape}",
            f"Voxel Spacing: {[f'{s:.2f}' for s in self.spacing]}",
            f"Bone Structures Found: {len(self.bone_regions)}",
            f"Total Bone Volume: {total_volume:,} voxels"
        ]

        for info in summary_info:
            ttk.Label(summary_frame, text=info, font=('Courier', 9)).pack(anchor=tk.W)

        # 3D Connection Status
        connection_frame = ttk.LabelFrame(self.info_frame, text="3D Module Status", padding=5)
        connection_frame.pack(fill=tk.X, pady=(0, 10))

        if self.dicom_3d_converter is not None:
            ttk.Label(connection_frame, text="✓ 3D Module Connected",
                      font=('Courier', 9), foreground='green').pack(anchor=tk.W)
            ttk.Label(connection_frame, text="Ready for 3D visualization",
                      font=('Courier', 9)).pack(anchor=tk.W)
        else:
            ttk.Label(connection_frame, text="✗ 3D Module Not Available",
                      font=('Courier', 9), foreground='red').pack(anchor=tk.W)
            ttk.Label(connection_frame, text="3D visualization disabled",
                      font=('Courier', 9)).pack(anchor=tk.W)

        # Patient info (if available)
        if self.slices:
            patient_frame = ttk.LabelFrame(self.info_frame, text="DICOM Information", padding=5)
            patient_frame.pack(fill=tk.X, pady=(0, 10))

            ds = self.slices[0]
            patient_info = []

            if hasattr(ds, 'PatientName') and ds.PatientName:
                patient_info.append(f"Patient: {ds.PatientName}")
            if hasattr(ds, 'StudyDate') and ds.StudyDate:
                patient_info.append(f"Study Date: {ds.StudyDate}")
            if hasattr(ds, 'Modality') and ds.Modality:
                patient_info.append(f"Modality: {ds.Modality}")
            if hasattr(ds, 'BodyPartExamined') and ds.BodyPartExamined:
                patient_info.append(f"Body Part: {ds.BodyPartExamined}")

            for info in patient_info:
                ttk.Label(patient_frame, text=info, font=('Courier', 9)).pack(anchor=tk.W)

        # Bone structures information
        if self.bone_regions:
            bones_frame = ttk.LabelFrame(self.info_frame, text="Detected Structures", padding=5)
            bones_frame.pack(fill=tk.BOTH, expand=True)

            # Create scrollable text widget
            text_frame = ttk.Frame(bones_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                  font=('Courier', 9), height=20, width=45)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=text_widget.yview)

            # Add bone information
            for i, bone in enumerate(self.bone_regions[:10]):  # Show top 10
                text_widget.insert(tk.END, f"STRUCTURE #{i + 1}\n")
                text_widget.insert(tk.END, f"{'-' * 25}\n")
                text_widget.insert(tk.END, f"Type: {bone['bone_type']}\n")
                text_widget.insert(tk.END, f"Location: {bone['anatomical_region']}\n")
                text_widget.insert(tk.END, f"Volume: {bone['size']:,} voxels\n")
                text_widget.insert(tk.END, f"Cortical: {bone['cortical_percentage']:.1f}%\n")
                text_widget.insert(tk.END, f"Mean HU: {bone['mean_hu']:.1f}\n\n")

            text_widget.config(state=tk.DISABLED)

    def create_visualization_panel(self):
        """Create visualization panel with 2D views and 3D button"""
        try:
            # Control buttons frame
            control_frame = ttk.Frame(self.canvas_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))

            # 3D button - connects to second file
            if self.dicom_3d_converter is not None:
                ttk.Button(control_frame, text="Show 3D Model (Connected Module)",
                           command=self.launch_3d_viewer).pack(side=tk.LEFT, padx=(0, 10))
            else:
                btn_3d = ttk.Button(control_frame, text="3D Not Available",
                                    state='disabled')
                btn_3d.pack(side=tk.LEFT, padx=(0, 10))

            ttk.Button(control_frame, text="Export Results",
                       command=self.export_results).pack(side=tk.LEFT)

            # Create 2D views
            self.create_2d_views()

        except Exception as e:
            print(f"Error creating visualization panel: {e}")
            ttk.Label(self.canvas_frame, text=f"Visualization error: {str(e)}").pack()

    def create_2d_views(self):
        """Create 2D projection views with better error handling"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('2D Projection Views', fontsize=14, fontweight='bold')

            # Create different projection views
            projections = {}

            if self.original_volume is not None and self.original_volume.size > 0:
                projections['Sagittal (Side)'] = np.max(self.original_volume, axis=2)
                projections['Coronal (Front)'] = np.max(self.original_volume, axis=1)
                projections['Axial (Top)'] = np.max(self.original_volume, axis=0)

                if self.volume is not None:
                    projections['Bone Mask'] = np.max(self.volume, axis=2)
                else:
                    projections['Bone Mask'] = np.zeros_like(projections['Sagittal (Side)'])

                axes_flat = axes.flatten()

                for i, (title, projection) in enumerate(projections.items()):
                    ax = axes_flat[i]

                    if projection.size > 0:
                        if 'Mask' in title:
                            im = ax.imshow(projection, cmap='bone', aspect='equal')
                        else:
                            # Apply windowing for better visualization
                            windowed = np.clip(projection, -200, 1000)
                            im = ax.imshow(windowed, cmap='bone', aspect='equal')

                        ax.set_title(title, fontweight='bold', fontsize=10)
                        ax.axis('off')

                        # Add colorbar
                        plt.colorbar(im, ax=ax, shrink=0.6)
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(title, fontweight='bold', fontsize=10)
                        ax.axis('off')

            else:
                # Show placeholder if no data
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No Image Data', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')

            plt.tight_layout()

            # Embed matplotlib figure in tkinter
            canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Error creating 2D views: {e}")
            ttk.Label(self.canvas_frame, text=f"2D visualization error: {str(e)}").pack()

    def launch_3d_viewer(self):
        """Launch 3D viewer using connected module in separate thread"""
        if self.dicom_3d_converter is None:
            messagebox.showwarning("Warning", "3D module not available")
            return

        if self.dicom_dir is None:
            messagebox.showwarning("Warning", "No DICOM directory selected")
            return

        # Show loading message
        self.status_var.set("Loading 3D visualization...")

        # Launch 3D viewer in separate thread
        viewer_thread = threading.Thread(target=self.run_3d_reconstruction)
        viewer_thread.daemon = True
        viewer_thread.start()

    def run_3d_reconstruction(self):
        """Run 3D reconstruction using the connected module"""
        try:
            print("Starting 3D reconstruction using connected module...")

            # Update status
            self.root.after(0, lambda: self.status_var.set("Running 3D reconstruction..."))

            # Use the connected 3D module to run reconstruction
            self.dicom_3d_converter.run()

            # Update status back to ready
            self.root.after(0, lambda: self.status_var.set("3D visualization complete"))

        except Exception as e:
            error_msg = f"3D reconstruction error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.root.after(0, lambda: self.show_error(error_msg))
            self.root.after(0, lambda: self.status_var.set("3D reconstruction failed"))

    def export_results(self):
        """Export analysis results to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Analysis Results"
            )

            if filename:
                with open(filename, 'w') as f:
                    f.write("DICOM 2D Analysis Results with 3D Connection\n")
                    f.write("=" * 50 + "\n\n")

                    f.write(f"DICOM Directory: {self.dicom_dir}\n")
                    f.write(f"Number of Slices: {len(self.slices)}\n")
                    f.write(f"Volume Dimensions: {self.original_volume.shape}\n")
                    f.write(f"Voxel Spacing: {self.spacing}\n")
                    f.write(f"3D Module Connected: {'Yes' if self.dicom_3d_converter else 'No'}\n\n")

                    f.write(f"Identified Bone Structures: {len(self.bone_regions)}\n")
                    f.write("-" * 40 + "\n")

                    for i, bone in enumerate(self.bone_regions):
                        f.write(f"\nStructure #{i + 1}:\n")
                        f.write(f"  Type: {bone['bone_type']}\n")
                        f.write(f"  Location: {bone['anatomical_region']}\n")
                        f.write(f"  Volume: {bone['size']} voxels\n")
                        f.write(f"  Cortical Percentage: {bone['cortical_percentage']:.1f}%\n")
                        f.write(f"  Mean HU: {bone['mean_hu']:.1f}\n")

                messagebox.showinfo("Success", f"Results exported to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

    def run_gui(self):
        """Run the GUI application"""
        self.create_main_window()
        self.root.mainloop()

    def run_command_line(self):
        """Run command line version"""
        if not self.dicom_dir:
            print("No DICOM directory specified")
            return

        try:
            print("Starting DICOM 2D Analysis Process...")
            self.load_dicom_series()
            self.process_volume()
            self.identify_bone_structures()

            # Initialize 3D connection
            self.initialize_3d_connection()

            # Show results
            print("\n" + "=" * 50)
            print("2D ANALYSIS RESULTS")
            print("=" * 50)
            print(f"DICOM Directory: {self.dicom_dir}")
            print(f"Number of Slices: {len(self.slices)}")
            print(f"Volume Dimensions: {self.original_volume.shape}")
            print(f"Voxel Spacing: {self.spacing}")
            print(f"3D Module Connected: {'Yes' if self.dicom_3d_converter else 'No'}")
            print(f"Identified Bone Structures: {len(self.bone_regions)}")

            for i, bone in enumerate(self.bone_regions[:5]):
                print(f"\nStructure #{i + 1}:")
                print(f"  Type: {bone['bone_type']}")
                print(f"  Location: {bone['anatomical_region']}")
                print(f"  Volume: {bone['size']} voxels")
                print(f"  Cortical: {bone['cortical_percentage']:.1f}%")

            # Offer 3D visualization if available
            if self.dicom_3d_converter is not None:
                response = input("\nWould you like to view the 3D model? (y/n): ")
                if response.lower().startswith('y'):
                    print("Launching 3D visualization...")
                    self.run_3d_reconstruction()

        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()


def main():
    """Main function with improved argument handling"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced DICOM 2D Analysis with 3D Connection')
    parser.add_argument('--dir', '-d', type=str, help='DICOM directory path')
    parser.add_argument('--gui', '-g', action='store_true', help='Run with GUI (default)')
    parser.add_argument('--cli', '-c', action='store_true', help='Run command line version')

    args = parser.parse_args()

    # Default DICOM directory (update this path)
    default_dicom_dir = r"D:\5th sem notes\MINORPROJECTSENGINERRING\3dconstructions\dicom5\J040"

    dicom_dir = args.dir if args.dir else (default_dicom_dir if os.path.exists(default_dicom_dir) else None)

    # Create converter
    converter = EnhancedDicom2D(dicom_dir)

    # Adjust parameters for better results (optional)
    converter.cortical_bone_lower = 300
    converter.trabecular_bone_lower = 100

    try:
        if args.cli:
            # Command line version
            if not dicom_dir:
                print("Error: No DICOM directory specified. Use --dir option.")
                return
            converter.run_command_line()
        else:
            # GUI version (default)
            converter.run_gui()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()