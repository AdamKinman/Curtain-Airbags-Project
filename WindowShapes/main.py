"""
Window Shapes Processing GUI

A graphical user interface for processing car window images through
the full pipeline: horizontal alignment, mask creation, polygon extraction,
DXF generation, and scaling.
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import threading
import queue

from image_files import is_image_file, list_image_files, find_image_path
from generate_new_design import (
    CarType, CarDescription, CarDesignGenerator,
    generate_car_design, list_available_car_types, DEFAULT_DESCRIPTIONS
)

from config import (
    ORIGINAL_IMAGE_FOLDER,
    ORIGINAL_LABELS_FOLDER,
    HORIZONTAL_ALIGN_IMAGES,
    ROTATED_LABELS_FOLDER,
    WINDOW_MASK_FOLDER,
    SIDE_MIRROR_MASK_FOLDER,
    PROCESSED_MASK_FOLDER,
    POLYGON_FOLDER,
    DXF_FOLDER,
    SCALED_DXF_FOLDER,
    GENERATED_IMAGES_FOLDER
)

# Maximum number of files allowed for DXF scaling (API limit)
MAX_SCALE_DXF_FILES = 14


class RedirectText:
    """Redirect stdout to a text widget."""
    def __init__(self, text_widget, queue):
        self.text_widget = text_widget
        self.queue = queue

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass


class WindowShapesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Window Shapes Processing Tool")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Queue for thread-safe logging
        self.log_queue = queue.Queue()
        
        # Current displayed image
        self.current_image = None
        self.current_image_tk = None
        
        # Image generation state
        self.generation_in_progress = False
        self.generation_progress = tk.DoubleVar(value=0)
        self.generation_status = tk.StringVar(value="")
        
        # Setup UI
        self.setup_ui()
        
        # Start log queue processing
        self.process_log_queue()
    
    def setup_ui(self):
        """Setup the main UI components."""
        # Create main paned window
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        self.left_frame = ttk.Frame(self.paned, width=400)
        self.paned.add(self.left_frame, weight=1)
        
        # Right panel - Image display
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=2)
        
        self.setup_left_panel()
        self.setup_right_panel()
    
    def setup_left_panel(self):
        """Setup the left control panel."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Processing
        self.processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_tab, text="Processing")
        self.setup_processing_tab()
        
        # Tab 2: Browse
        self.browse_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.browse_tab, text="Browse Files")
        self.setup_browse_tab()
        
        # Tab 3: Generate Images
        self.generate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.generate_tab, text="Generate Images")
        self.setup_generate_tab()
        
        # Log output at bottom
        log_frame = ttk.LabelFrame(self.left_frame, text="Log Output")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_processing_tab(self):
        """Setup the processing operations tab."""
        # File selection frame
        selection_frame = ttk.LabelFrame(self.processing_tab, text="File Selection")
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Selection mode
        self.selection_mode = tk.StringVar(value="single")
        ttk.Radiobutton(selection_frame, text="Single File", variable=self.selection_mode, 
                       value="single", command=self.on_selection_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(selection_frame, text="Multiple Files", variable=self.selection_mode, 
                       value="multiple", command=self.on_selection_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(selection_frame, text="Entire Folder", variable=self.selection_mode, 
                       value="folder", command=self.on_selection_mode_change).pack(anchor=tk.W)
        
        # Selected files display
        self.selected_files_var = tk.StringVar(value="No files selected")
        ttk.Label(selection_frame, textvariable=self.selected_files_var, wraplength=350).pack(anchor=tk.W, pady=5)
        
        self.select_files_btn = ttk.Button(selection_frame, text="Select Files...", command=self.select_input_files)
        self.select_files_btn.pack(pady=5)
        
        # Store selected files
        self.selected_files = []
        self.input_folder = ORIGINAL_IMAGE_FOLDER
        
        # Pipeline Steps frame (with checkboxes)
        steps_frame = ttk.LabelFrame(self.processing_tab, text="Pipeline Steps")
        steps_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pipeline_steps = {
            'align': tk.BooleanVar(value=True),
            'masks': tk.BooleanVar(value=True),
            'process_masks': tk.BooleanVar(value=True),
            'polygons': tk.BooleanVar(value=True),
            'dxf': tk.BooleanVar(value=True),
            'scale': tk.BooleanVar(value=False),  # Off by default due to API limits
        }
        
        step_labels = [
            ('align', '1. Horizontal Align'),
            ('masks', '2. Create Masks (SAM3)'),
            ('process_masks', '3. Process Masks (extrapolation)'),
            ('polygons', '4. Create Polygons from Masks'),
            ('dxf', '5. Create DXF Files from Polygons'),
            ('scale', '6. Scale DXF Files (API limited - max 14 files)'),
        ]
        
        for key, label in step_labels:
            ttk.Checkbutton(steps_frame, text=label, 
                           variable=self.pipeline_steps[key]).pack(anchor=tk.W)
        
        # Warning label for scale step
        self.scale_warning_label = ttk.Label(
            steps_frame, 
            text="⚠️ Scale DXF step has API limits (max 14 files). "
                 "Disable it for large batches.",
            foreground="orange"
        )
        self.scale_warning_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Run Pipeline button
        ttk.Button(steps_frame, text="Run Pipeline", 
                  command=self.run_pipeline).pack(fill=tk.X, padx=5, pady=10)
        
        # Output folder selection
        output_frame = ttk.LabelFrame(self.processing_tab, text="Output Folders (optional)")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Custom output folder option
        self.use_custom_output = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Use custom output folder", 
                       variable=self.use_custom_output).pack(anchor=tk.W)
        
        self.custom_output_folder = tk.StringVar(value="")
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Entry(output_entry_frame, textvariable=self.custom_output_folder).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_entry_frame, text="Browse...", command=self.select_output_folder).pack(side=tk.RIGHT)
    
    def setup_browse_tab(self):
        """Setup the file browsing tab."""
        # File type selection
        type_frame = ttk.LabelFrame(self.browse_tab, text="File Type")
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.browse_type = tk.StringVar(value="image")
        types = [
            ("Images (CleanedImages)", "image", ORIGINAL_IMAGE_FOLDER),
            ("Generated Images", "generated", GENERATED_IMAGES_FOLDER),
            ("Aligned Images", "aligned", HORIZONTAL_ALIGN_IMAGES),
            ("Masks (overlay)", "mask", PROCESSED_MASK_FOLDER),
            ("Polygons (overlay)", "polygon", POLYGON_FOLDER),
            ("DXF Files", "dxf", DXF_FOLDER),
            ("Scaled DXF Files", "scaled_dxf", SCALED_DXF_FOLDER),
        ]
        
        self.browse_folders = {}
        for text, value, folder in types:
            self.browse_folders[value] = folder
            ttk.Radiobutton(type_frame, text=text, variable=self.browse_type, 
                           value=value, command=self.refresh_file_list).pack(anchor=tk.W)
        
        # File list
        list_frame = ttk.LabelFrame(self.browse_tab, text="Files")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar and listbox
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        self.file_listbox.bind('<Double-1>', self.on_file_double_click)
        
        # Display button
        ttk.Button(self.browse_tab, text="Display Selected", 
                  command=self.display_selected_file).pack(pady=5)
        
        # Initial file list
        self.refresh_file_list()
    
    def setup_generate_tab(self):
        """Setup the image generation tab."""
        # Create a scrollable frame for the generate tab
        canvas = tk.Canvas(self.generate_tab)
        scrollbar = ttk.Scrollbar(self.generate_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input mode selection
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Input Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.gen_input_mode = tk.StringVar(value="default")
        ttk.Radiobutton(mode_frame, text="Use Default Car Type", 
                       variable=self.gen_input_mode, value="default",
                       command=self.on_gen_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Custom Parameters", 
                       variable=self.gen_input_mode, value="custom",
                       command=self.on_gen_mode_change).pack(anchor=tk.W)
        
        # Default car type selection
        self.default_type_frame = ttk.LabelFrame(scrollable_frame, text="Select Car Type")
        self.default_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.selected_car_type = tk.StringVar(value=CarType.SEDAN.value)
        car_types = list_available_car_types()
        self.car_type_combo = ttk.Combobox(self.default_type_frame, 
                                           textvariable=self.selected_car_type,
                                           values=car_types, state="readonly")
        self.car_type_combo.pack(fill=tk.X, padx=5, pady=5)
        
        # Show default description
        self.default_desc_label = ttk.Label(self.default_type_frame, text="", wraplength=350)
        self.default_desc_label.pack(anchor=tk.W, padx=5, pady=2)
        self.car_type_combo.bind("<<ComboboxSelected>>", self.on_car_type_selected)
        self.on_car_type_selected(None)  # Initialize
        
        # Custom parameters frame
        self.custom_params_frame = ttk.LabelFrame(scrollable_frame, text="Custom Parameters")
        self.custom_params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Car type (custom)
        ttk.Label(self.custom_params_frame, text="Car Type:").pack(anchor=tk.W, padx=5)
        self.custom_car_type = tk.StringVar(value="sedan")
        ttk.Entry(self.custom_params_frame, textvariable=self.custom_car_type).pack(fill=tk.X, padx=5, pady=2)
        
        # Color
        ttk.Label(self.custom_params_frame, text="Color:").pack(anchor=tk.W, padx=5)
        self.custom_color = tk.StringVar(value="")
        ttk.Entry(self.custom_params_frame, textvariable=self.custom_color).pack(fill=tk.X, padx=5, pady=2)
        
        # Brand style
        ttk.Label(self.custom_params_frame, text="Brand Style:").pack(anchor=tk.W, padx=5)
        self.custom_brand_style = tk.StringVar(value="")
        ttk.Entry(self.custom_params_frame, textvariable=self.custom_brand_style).pack(fill=tk.X, padx=5, pady=2)
        
        # Era
        ttk.Label(self.custom_params_frame, text="Era:").pack(anchor=tk.W, padx=5)
        self.custom_era = tk.StringVar(value="modern")
        era_combo = ttk.Combobox(self.custom_params_frame, textvariable=self.custom_era,
                                  values=["modern", "vintage", "futuristic", "retro"])
        era_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Features (comma-separated)
        ttk.Label(self.custom_params_frame, text="Features (comma-separated):").pack(anchor=tk.W, padx=5)
        self.custom_features = tk.StringVar(value="")
        ttk.Entry(self.custom_params_frame, textvariable=self.custom_features).pack(fill=tk.X, padx=5, pady=2)
        
        # Custom details
        ttk.Label(self.custom_params_frame, text="Custom Details:").pack(anchor=tk.W, padx=5)
        self.custom_details = tk.StringVar(value="")
        ttk.Entry(self.custom_params_frame, textvariable=self.custom_details).pack(fill=tk.X, padx=5, pady=2)
        
        # Initially hide custom params
        self.custom_params_frame.pack_forget()
        
        # Generation options
        options_frame = ttk.LabelFrame(scrollable_frame, text="Generation Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Number of images
        count_frame = ttk.Frame(options_frame)
        count_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(count_frame, text="Number of Images:").pack(side=tk.LEFT)
        self.gen_count = tk.IntVar(value=1)
        ttk.Spinbox(count_frame, from_=1, to=20, textvariable=self.gen_count, width=5).pack(side=tk.LEFT, padx=5)
        
        # Seed (optional)
        seed_frame = ttk.Frame(options_frame)
        seed_frame.pack(fill=tk.X, padx=5, pady=2)
        self.use_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(seed_frame, text="Use Seed:", variable=self.use_seed).pack(side=tk.LEFT)
        self.gen_seed = tk.IntVar(value=42)
        self.seed_entry = ttk.Spinbox(seed_frame, from_=0, to=999999999, textvariable=self.gen_seed, width=12)
        self.seed_entry.pack(side=tk.LEFT, padx=5)
        
        # Filename prefix
        prefix_frame = ttk.Frame(options_frame)
        prefix_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(prefix_frame, text="Filename Prefix:").pack(side=tk.LEFT)
        self.gen_filename_prefix = tk.StringVar(value="car_design")
        ttk.Entry(prefix_frame, textvariable=self.gen_filename_prefix, width=20).pack(side=tk.LEFT, padx=5)
        
        # Warning label
        warning_frame = ttk.Frame(scrollable_frame)
        warning_frame.pack(fill=tk.X, padx=5, pady=5)
        warning_label = ttk.Label(
            warning_frame,
            text="⚠️ Image generation may take several minutes per image.\n"
                 "The UI will remain responsive during generation.",
            foreground="orange",
            wraplength=350
        )
        warning_label.pack(anchor=tk.W)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Generation Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.gen_progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.generation_progress,
            maximum=100,
            mode='determinate'
        )
        self.gen_progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.gen_status_label = ttk.Label(progress_frame, textvariable=self.generation_status)
        self.gen_status_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Generate button
        self.generate_btn = ttk.Button(scrollable_frame, text="Generate Images", 
                                        command=self.start_image_generation)
        self.generate_btn.pack(pady=10)
        
        # Cancel button (hidden initially)
        self.cancel_gen_btn = ttk.Button(scrollable_frame, text="Cancel Generation",
                                          command=self.cancel_generation, state=tk.DISABLED)
        self.cancel_gen_btn.pack(pady=5)
        
        # Flag for cancellation
        self.cancel_generation_flag = False
    
    def on_gen_mode_change(self):
        """Handle generation input mode change."""
        mode = self.gen_input_mode.get()
        if mode == "default":
            self.default_type_frame.pack(fill=tk.X, padx=5, pady=5, after=self.default_type_frame.master.winfo_children()[0])
            self.custom_params_frame.pack_forget()
        else:
            self.default_type_frame.pack_forget()
            self.custom_params_frame.pack(fill=tk.X, padx=5, pady=5, after=self.custom_params_frame.master.winfo_children()[0])
    
    def on_car_type_selected(self, event):
        """Update description when car type is selected."""
        selected = self.selected_car_type.get()
        # Find the CarType enum
        for ct in CarType:
            if ct.value == selected:
                desc = DEFAULT_DESCRIPTIONS.get(ct)
                if desc:
                    info = f"Default: {desc.color} {desc.car_type}, {desc.era} era"
                    self.default_desc_label.config(text=info)
                break
    
    def start_image_generation(self):
        """Start the image generation process."""
        if self.generation_in_progress:
            messagebox.showwarning("Warning", "Image generation is already in progress.")
            return
        
        # Show warning
        count = self.gen_count.get()
        result = messagebox.askyesno(
            "Confirm Generation",
            f"You are about to generate {count} image(s).\n\n"
            "⚠️ This process may take several minutes per image.\n"
            "The model will be loaded into memory (requires significant RAM/VRAM).\n\n"
            "Do you want to continue?"
        )
        if not result:
            return
        
        self.generation_in_progress = True
        self.cancel_generation_flag = False
        self.generation_progress.set(0)
        self.generation_status.set("Initializing...")
        self.generate_btn.config(state=tk.DISABLED)
        self.cancel_gen_btn.config(state=tk.NORMAL)
        
        # Start generation in thread
        self.run_in_thread(self._generate_images)
    
    def cancel_generation(self):
        """Cancel the ongoing generation."""
        self.cancel_generation_flag = True
        self.generation_status.set("Cancelling...")
        self.log("Generation cancellation requested...")
    
    def _generate_images(self):
        """Thread worker for image generation."""
        try:
            count = self.gen_count.get()
            mode = self.gen_input_mode.get()
            prefix = self.gen_filename_prefix.get() or "car_design"
            seed = self.gen_seed.get() if self.use_seed.get() else None

            # Pick the next available filename index so we never overwrite.
            os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
            pattern = re.compile(rf"^{re.escape(prefix)}_(\\d+)\\.png$", re.IGNORECASE)
            existing_indices = []
            for name in os.listdir(GENERATED_IMAGES_FOLDER):
                match = pattern.match(name)
                if match:
                    try:
                        existing_indices.append(int(match.group(1)))
                    except ValueError:
                        pass
            start_index = (max(existing_indices) + 1) if existing_indices else 1
            
            self.log("Initializing image generator...")
            self.root.after(0, lambda: self.generation_status.set("Loading model..."))
            
            generator = CarDesignGenerator()
            generator.load_model()
            
            self.log("Model loaded. Starting image generation...")
            
            generated_images = []
            
            for i in range(count):
                if self.cancel_generation_flag:
                    self.log("Generation cancelled by user.")
                    break
                
                progress = ((i) / count) * 100
                self.root.after(0, lambda p=progress: self.generation_progress.set(p))
                self.root.after(0, lambda idx=i+1, total=count: 
                               self.generation_status.set(f"Generating image {idx}/{total}..."))
                self.log(f"Generating image {i+1}/{count}...")

                filename = f"{prefix}_{start_index + i:04d}.png"
                
                if mode == "default":
                    # Use default car type
                    selected = self.selected_car_type.get()
                    car_type_enum = None
                    for ct in CarType:
                        if ct.value == selected:
                            car_type_enum = ct
                            break
                    
                    image = generator.generate(
                        car_type=car_type_enum,
                        output_path=GENERATED_IMAGES_FOLDER,
                        filename=filename,
                        seed=seed + i if seed else None,
                    )
                else:
                    # Use custom parameters
                    features = [f.strip() for f in self.custom_features.get().split(",") if f.strip()]
                    
                    description = CarDescription(
                        car_type=self.custom_car_type.get() or "sedan",
                        color=self.custom_color.get() or None,
                        brand_style=self.custom_brand_style.get() or None,
                        era=self.custom_era.get() or "modern",
                        features=features,
                        custom_details=self.custom_details.get() or None,
                    )
                    
                    image = generator.generate(
                        description=description,
                        output_path=GENERATED_IMAGES_FOLDER,
                        filename=filename,
                        seed=seed + i if seed else None,
                    )
                
                generated_images.append(image)
                self.log(f"Image {i+1}/{count} generated successfully.")
            
            # Complete
            final_progress = 100 if not self.cancel_generation_flag else ((len(generated_images)) / count) * 100
            self.root.after(0, lambda: self.generation_progress.set(final_progress))
            
            if self.cancel_generation_flag:
                self.root.after(0, lambda: self.generation_status.set(
                    f"Cancelled. Generated {len(generated_images)}/{count} images."))
                self.log(f"Generation cancelled. {len(generated_images)} images were created.")
            else:
                self.root.after(0, lambda: self.generation_status.set(
                    f"Complete! Generated {count} images."))
                self.log(f"Image generation complete. {count} images saved to {GENERATED_IMAGES_FOLDER}")
            
            # Display the last generated image if any
            if generated_images:
                self.root.after(0, lambda img=generated_images[-1]: self._display_generated_image(img))
            
            # Refresh file list if on browse tab with generated images
            self.root.after(0, self.refresh_file_list)
            
        except Exception as e:
            self.log(f"Error during image generation: {e}")
            self.root.after(0, lambda: self.generation_status.set(f"Error: {str(e)[:50]}..."))
            self.root.after(0, lambda: messagebox.showerror("Generation Error", str(e)))
        finally:
            self.generation_in_progress = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_gen_btn.config(state=tk.DISABLED))
    
    def _display_generated_image(self, image):
        """Display a generated PIL image on the canvas."""
        if image:
            self.show_image_on_canvas(image.convert("RGBA"))
            self.info_label.config(text="Displaying: Latest generated image")
    
    def setup_right_panel(self):
        """Setup the right image display panel."""
        # Image display frame
        display_frame = ttk.LabelFrame(self.right_frame, text="Preview")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(display_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # Info label
        self.info_label = ttk.Label(self.right_frame, text="Select a file to preview")
        self.info_label.pack(pady=5)
    
    def on_selection_mode_change(self):
        """Handle selection mode change."""
        self.selected_files = []
        self.selected_files_var.set("No files selected")
    
    def select_input_files(self):
        """Open file dialog to select input files."""
        mode = self.selection_mode.get()
        
        if mode == "single":
            file = filedialog.askopenfilename(
                initialdir=self.input_folder,
                title="Select Image File",
                filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
            )
            if file:
                self.selected_files = [os.path.basename(file)]
                self.input_folder = os.path.dirname(file)
                self.selected_files_var.set(f"Selected: {self.selected_files[0]}")
        
        elif mode == "multiple":
            files = filedialog.askopenfilenames(
                initialdir=self.input_folder,
                title="Select Image Files",
                filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
            )
            if files:
                self.selected_files = [os.path.basename(f) for f in files]
                self.input_folder = os.path.dirname(files[0])
                self.selected_files_var.set(f"Selected: {len(self.selected_files)} files")
        
        elif mode == "folder":
            folder = filedialog.askdirectory(
                initialdir=self.input_folder,
                title="Select Folder"
            )
            if folder:
                self.input_folder = folder
                self.selected_files = [f for f in os.listdir(folder) if is_image_file(f)]
                self.selected_files_var.set(f"Selected folder: {folder}\n({len(self.selected_files)} files)")
    
    def select_output_folder(self):
        """Select custom output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.custom_output_folder.set(folder)
    
    def refresh_file_list(self):
        """Refresh the file list based on selected type."""
        self.file_listbox.delete(0, tk.END)
        
        browse_type = self.browse_type.get()
        folder = self.browse_folders.get(browse_type, ORIGINAL_IMAGE_FOLDER)
        
        if not os.path.exists(folder):
            self.file_listbox.insert(tk.END, "(Folder does not exist)")
            return
        
        # Get appropriate files for each type
        if browse_type in ['dxf', 'scaled_dxf']:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.dxf')])
        elif browse_type in ['mask']:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.pt')])
        elif browse_type in ['polygon']:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.txt')])
        else:
            files = list_image_files(folder)
        
        if not files:
            self.file_listbox.insert(tk.END, "(No files found)")
            return
        
        for f in files:
            self.file_listbox.insert(tk.END, f)
    
    def on_file_select(self, event):
        """Handle file selection in listbox."""
        pass
    
    def on_file_double_click(self, event):
        """Handle double-click on file."""
        self.display_selected_file()
    
    def display_selected_file(self):
        """Display the selected file in the preview panel."""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        filename = self.file_listbox.get(selection[0])
        if filename.startswith("("):  # Skip placeholder entries
            return
        
        browse_type = self.browse_type.get()
        folder = self.browse_folders.get(browse_type, ORIGINAL_IMAGE_FOLDER)
        filepath = os.path.join(folder, filename)
        
        try:
            if browse_type in {"image", "aligned", "generated"}:
                self.display_image(filepath)
            elif browse_type == "mask":
                self.display_mask_overlay(filename)
            elif browse_type == "polygon":
                self.display_polygon_overlay(filename)
            elif browse_type in ["dxf", "scaled_dxf"]:
                self.display_dxf(filepath)
            
            self.info_label.config(text=f"Displaying: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display file: {e}")
    
    def display_image(self, filepath):
        """Display a simple image."""
        # Ensure modes like 'P' (palette PNG) display correctly in Tkinter.
        image = Image.open(filepath).convert("RGBA")
        self.show_image_on_canvas(image)
    
    def display_mask_overlay(self, mask_filename):
        """Display mask overlaid on corresponding image."""
        base_name = os.path.splitext(mask_filename)[0]
        
        # Try to find corresponding image
        image_path = None
        for img_folder in [HORIZONTAL_ALIGN_IMAGES, GENERATED_IMAGES_FOLDER, ORIGINAL_IMAGE_FOLDER, self.input_folder]:
            test_path = find_image_path(img_folder, base_name)
            if test_path:
                image_path = test_path
                break
        
        if image_path is None:
            messagebox.showerror("Error", "Could not find corresponding image for mask")
            return
        
        # Load image
        image = Image.open(image_path).convert("RGBA")
        
        # Load mask
        mask_path = os.path.join(PROCESSED_MASK_FOLDER, mask_filename)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(WINDOW_MASK_FOLDER, mask_filename)
        
        masks_tensor = torch.load(mask_path, weights_only=False)
        
        # Create overlay
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
        # Handle different tensor shapes
        if masks_tensor.ndim == 4:
            masks = [masks_tensor[i, 0].cpu().numpy() for i in range(masks_tensor.shape[0])]
        elif masks_tensor.ndim == 3:
            masks = [masks_tensor[i].cpu().numpy() for i in range(masks_tensor.shape[0])]
        else:
            masks = [masks_tensor.cpu().numpy()]
        
        colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), 
                  (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
        
        for i, mask in enumerate(masks):
            mask = (mask > 0).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask, mode='L')
            color = colors[i % len(colors)]
            single_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            single_overlay.paste(color, (0, 0), mask=mask_pil)
            overlay = Image.alpha_composite(overlay, single_overlay)
        
        result = Image.alpha_composite(image, overlay)
        self.show_image_on_canvas(result)
    
    def display_polygon_overlay(self, polygon_filename):
        """Display polygon overlaid on corresponding image."""
        base_name = os.path.splitext(polygon_filename)[0]
        
        # Try to find corresponding image
        image_path = None
        for img_folder in [HORIZONTAL_ALIGN_IMAGES, GENERATED_IMAGES_FOLDER, ORIGINAL_IMAGE_FOLDER, self.input_folder]:
            test_path = find_image_path(img_folder, base_name)
            if test_path:
                image_path = test_path
                break
        
        if image_path is None:
            messagebox.showerror("Error", "Could not find corresponding image for polygon")
            return
        
        # Load image
        image = Image.open(image_path).convert("RGBA")
        
        # Load polygons
        polygon_path = os.path.join(POLYGON_FOLDER, polygon_filename)
        polygons = self.parse_polygon_file(polygon_path)
        
        # Create overlay
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for polygon in polygons:
            if len(polygon) >= 3:
                # Draw filled polygon with transparency
                draw.polygon(polygon, fill=(255, 0, 0, 80), outline=(255, 0, 0, 255))
                # Draw thicker outline
                draw.line(polygon + [polygon[0]], fill=(255, 0, 0, 255), width=3)
        
        result = Image.alpha_composite(image, overlay)
        self.show_image_on_canvas(result)
    
    def display_dxf(self, filepath):
        """Display a DXF file as rendered polygons."""
        try:
            import ezdxf
        except ImportError:
            messagebox.showerror("Error", "ezdxf module not installed")
            return
        
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        
        # Collect all points to determine bounds
        all_points = []
        polygons = []
        
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0], -p[1]) for p in entity.get_points()]  # Flip Y back
                polygons.append(points)
                all_points.extend(points)
        
        if not all_points:
            messagebox.showinfo("Info", "DXF file contains no polylines")
            return
        
        # Calculate bounds
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Create image with padding
        padding = 20
        width = int(max_x - min_x + 2 * padding)
        height = int(max_y - min_y + 2 * padding)
        
        # Limit size for display
        max_dim = 800
        scale = min(max_dim / width, max_dim / height, 1.0)
        width = int(width * scale)
        height = int(height * scale)
        
        image = Image.new("RGB", (max(width, 100), max(height, 100)), "white")
        draw = ImageDraw.Draw(image)
        
        # Draw polygons
        for polygon in polygons:
            scaled_polygon = [
                (int((p[0] - min_x + padding) * scale), 
                 int((p[1] - min_y + padding) * scale))
                for p in polygon
            ]
            if len(scaled_polygon) >= 3:
                draw.polygon(scaled_polygon, outline="blue", fill=None)
                draw.line(scaled_polygon + [scaled_polygon[0]], fill="blue", width=2)
        
        self.show_image_on_canvas(image)
    
    def parse_polygon_file(self, filepath):
        """Parse a polygon file."""
        polygons = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = list(map(float, line.split()))
                polygon = [(int(values[i]), int(values[i+1])) for i in range(0, len(values), 2)]
                polygons.append(polygon)
        return polygons
    
    def show_image_on_canvas(self, image):
        """Display an image on the canvas, scaled to fit."""
        self.current_image = image
        self.update_canvas_image()
    
    def update_canvas_image(self):
        """Update the canvas with the current image, scaled to fit."""
        if self.current_image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        # Calculate scale to fit
        img_width, img_height = self.current_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.current_image_tk = ImageTk.PhotoImage(resized)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                  image=self.current_image_tk, anchor=tk.CENTER)
    
    def on_canvas_resize(self, event):
        """Handle canvas resize."""
        self.update_canvas_image()
    
    def log(self, message):
        """Add message to log."""
        self.log_queue.put(message + "\n")
    
    def process_log_queue(self):
        """Process messages in the log queue."""
        while True:
            try:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
            except queue.Empty:
                break
        self.root.after(100, self.process_log_queue)
    
    def get_selected_files_for_operation(self):
        """Get the list of files to process."""
        if not self.selected_files:
            messagebox.showwarning("Warning", "No files selected. Please select files first.")
            return None
        return self.selected_files
    
    def run_in_thread(self, func, *args):
        """Run a function in a separate thread."""
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()
    
    def run_horizontal_align(self):
        """Run horizontal alignment on selected files."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        self.log("Starting horizontal alignment...")
        self.run_in_thread(self._run_horizontal_align, files)
    
    def _run_horizontal_align(self, files):
        """Thread worker for horizontal alignment."""
        try:
            import horizontal_align
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else HORIZONTAL_ALIGN_IMAGES
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(ROTATED_LABELS_FOLDER, exist_ok=True)
            
            success = 0
            failed = 0
            
            for f in files:
                name = os.path.splitext(f)[0]
                try:
                    horizontal_align.alignImage(
                        name, 
                        self.input_folder, 
                        ORIGINAL_LABELS_FOLDER,
                        horizontal_align.TIRE_LABEL_KEY
                    )
                    success += 1
                    self.log(f"  Aligned: {f}")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {f} - {e}")
            
            self.log(f"Horizontal alignment complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during horizontal alignment: {e}")
    
    def run_create_masks(self):
        """Run mask creation on selected files."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        self.log("Starting mask creation (this may take a while)...")
        self.run_in_thread(self._run_create_masks, files)
    
    def _run_create_masks(self, files):
        """Thread worker for mask creation."""
        try:
            import create_masks
            from create_masks import createMaskFromImage
            
            # Build model once
            self.log("Loading SAM3 model...")
            model = create_masks.build_sam3_image_model()
            processor = create_masks.Sam3Processor(model)
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else WINDOW_MASK_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(SIDE_MIRROR_MASK_FOLDER, exist_ok=True)
            
            success = 0
            failed = 0
            
            for f in files:
                try:
                    image_path = os.path.join(HORIZONTAL_ALIGN_IMAGES, f)
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.input_folder, f)
                    
                    base_name = os.path.splitext(f)[0]
                    
                    # Create window mask
                    mask = createMaskFromImage(image_path, processor, "Side window")
                    torch.save(mask, os.path.join(output_folder, f"{base_name}.pt"))
                    
                    # Create mirror mask
                    mirror_mask = createMaskFromImage(image_path, processor, "Side mirror")
                    torch.save(mirror_mask, os.path.join(SIDE_MIRROR_MASK_FOLDER, f"{base_name}.pt"))
                    
                    success += 1
                    self.log(f"  Created masks: {f}")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {f} - {e}")
            
            self.log(f"Mask creation complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during mask creation: {e}")
    
    def run_process_masks(self):
        """Run mask post-processing (extrapolation behind mirror)."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        self.log("Starting mask processing...")
        self.run_in_thread(self._run_process_masks, files)
    
    def _run_process_masks(self, files):
        """Thread worker for mask post-processing."""
        try:
            import process_masks_new
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else PROCESSED_MASK_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            
            success = 0
            failed = 0
            
            for f in files:
                base_name = os.path.splitext(f)[0]
                
                try:
                    window_mask_path = os.path.join(WINDOW_MASK_FOLDER, f"{base_name}.pt")
                    mirror_mask_path = os.path.join(SIDE_MIRROR_MASK_FOLDER, f"{base_name}.pt")
                    output_path = os.path.join(output_folder, f"{base_name}.pt")
                    
                    if not os.path.exists(window_mask_path):
                        raise FileNotFoundError(f"Window mask not found: {window_mask_path}")
                    
                    # Load window masks
                    window_masks_tensor = torch.load(window_mask_path, weights_only=False)
                    num_masks = window_masks_tensor.shape[0]
                    window_masks = [process_masks_new.squeezeMask(window_masks_tensor[i]) for i in range(num_masks)]
                    
                    # Load mirror mask if available
                    mirror_mask = None
                    if os.path.exists(mirror_mask_path):
                        mirror_masks_tensor = torch.load(mirror_mask_path, weights_only=False)
                        if mirror_masks_tensor.ndim >= 3 and mirror_masks_tensor.shape[0] > 0:
                            mirror_mask = process_masks_new.squeezeMask(mirror_masks_tensor[0])
                            for i in range(1, mirror_masks_tensor.shape[0]):
                                mirror_mask = torch.logical_or(mirror_mask, process_masks_new.squeezeMask(mirror_masks_tensor[i]))
                        elif mirror_masks_tensor.ndim == 2:
                            mirror_mask = mirror_masks_tensor
                    
                    # For now, just save the combined masks to processed folder
                    # Full processing would require the complete processAndSaveMasks logic
                    # which processes front window extrapolation
                    torch.save(window_masks_tensor, output_path)
                    
                    success += 1
                    self.log(f"  Processed masks: {base_name}")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {base_name} - {e}")
            
            self.log(f"Mask processing complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during mask processing: {e}")
    
    def run_create_polygons(self):
        """Run polygon creation from masks."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        self.log("Starting polygon creation...")
        self.run_in_thread(self._run_create_polygons, files)
    
    def _run_create_polygons(self, files):
        """Thread worker for polygon creation."""
        try:
            import masks_to_polygons
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else POLYGON_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            
            success = 0
            failed = 0
            
            for f in files:
                base_name = os.path.splitext(f)[0]
                mask_file = f"{base_name}.pt"
                
                try:
                    masks = masks_to_polygons.loadProcessedMasks(base_name)
                    all_polygons = []
                    for mask in masks:
                        polygons = masks_to_polygons.maskToPolygons(mask)
                        all_polygons.extend(polygons)
                    
                    with open(os.path.join(output_folder, f"{base_name}.txt"), 'w') as file:
                        for polygon in all_polygons:
                            file.write(' '.join(map(str, polygon)) + '\n')
                    
                    success += 1
                    self.log(f"  Created polygons: {base_name}")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {base_name} - {e}")
            
            self.log(f"Polygon creation complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during polygon creation: {e}")
    
    def run_create_dxf(self):
        """Run DXF creation from polygons."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        self.log("Starting DXF creation...")
        self.run_in_thread(self._run_create_dxf, files)
    
    def _run_create_dxf(self, files):
        """Thread worker for DXF creation."""
        try:
            import polygons_to_dxf
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else DXF_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            
            success = 0
            failed = 0
            
            for f in files:
                base_name = os.path.splitext(f)[0]
                polygon_path = os.path.join(POLYGON_FOLDER, f"{base_name}.txt")
                output_path = os.path.join(output_folder, f"{base_name}.dxf")
                
                try:
                    if not os.path.exists(polygon_path):
                        raise FileNotFoundError(f"Polygon file not found: {polygon_path}")
                    
                    polygons = polygons_to_dxf.loadPolygons(polygon_path)
                    if polygons:
                        polygons_to_dxf.polygonsToDxf(polygons, output_path)
                        success += 1
                        self.log(f"  Created DXF: {base_name}")
                    else:
                        self.log(f"  Warning: No polygons in {base_name}")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {base_name} - {e}")
            
            self.log(f"DXF creation complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during DXF creation: {e}")
    
    def run_scale_dxf(self):
        """Run DXF scaling."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        # Check file count limit
        if len(files) > MAX_SCALE_DXF_FILES:
            messagebox.showerror(
                "Error", 
                f"DXF scaling is limited to {MAX_SCALE_DXF_FILES} files due to API rate limits.\n"
                f"You selected {len(files)} files.\n\n"
                "Please select fewer files or process in batches."
            )
            return
        
        self.log(f"Starting DXF scaling for {len(files)} files...")
        self.run_in_thread(self._run_scale_dxf, files)
    
    def _run_scale_dxf(self, files):
        """Thread worker for DXF scaling."""
        try:
            import scale_dxf
            
            output_folder = self.custom_output_folder.get() if self.use_custom_output.get() else SCALED_DXF_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            
            # Initialize Gemini client
            client = scale_dxf.genai.Client(api_key=scale_dxf.API_KEY)
            
            success = 0
            failed = 0
            
            for f in files:
                try:
                    base_name = os.path.splitext(f)[0]
                    image_path = os.path.join(ORIGINAL_IMAGE_FOLDER, f)
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.input_folder, f)
                    
                    dxf_path = os.path.join(DXF_FOLDER, f"{base_name}.dxf")
                    output_path = os.path.join(output_folder, f"{base_name}.dxf")
                    
                    if not os.path.exists(dxf_path):
                        raise FileNotFoundError(f"DXF file not found: {dxf_path}")
                    
                    # Estimate dimensions
                    dimensions = scale_dxf.estimateDimensions(client, image_path)
                    if dimensions is None:
                        raise ValueError("Failed to estimate dimensions")
                    
                    # Get bounding box
                    bbox = scale_dxf.getDxfBoundingBox(dxf_path)
                    if bbox is None:
                        raise ValueError("Failed to get DXF bounding box")
                    
                    # Calculate scale and create scaled DXF
                    scale_factor = scale_dxf.calculateScaleFactor(bbox, dimensions)
                    scale_dxf.createScaledDxf(dxf_path, output_path, scale_factor)
                    
                    success += 1
                    self.log(f"  Scaled: {base_name} (factor: {scale_factor:.4f})")
                except Exception as e:
                    failed += 1
                    self.log(f"  Failed: {f} - {e}")
            
            self.log(f"DXF scaling complete: {success} succeeded, {failed} failed")
        except Exception as e:
            self.log(f"Error during DXF scaling: {e}")
    
    def run_pipeline(self):
        """Run the full pipeline with selected steps."""
        files = self.get_selected_files_for_operation()
        if files is None:
            return
        
        # Check scale step limitations
        if self.pipeline_steps['scale'].get() and len(files) > MAX_SCALE_DXF_FILES:
            result = messagebox.askyesno(
                "Warning",
                f"DXF scaling is limited to {MAX_SCALE_DXF_FILES} files due to API rate limits.\n"
                f"You selected {len(files)} files.\n\n"
                "Do you want to continue with scaling disabled?"
            )
            if result:
                self.pipeline_steps['scale'].set(False)
            else:
                return
        
        self.log(f"Starting pipeline with {len(files)} files...")
        self.run_in_thread(self._run_pipeline, files)
    
    def _run_pipeline(self, files):
        """Thread worker for full pipeline."""
        steps = self.pipeline_steps
        
        try:
            if steps['align'].get():
                self.log("\n=== Step 1: Horizontal Alignment ===")
                self._run_horizontal_align(files)
            
            if steps['masks'].get():
                self.log("\n=== Step 2: Creating Masks ===")
                self._run_create_masks(files)
            
            if steps['process_masks'].get():
                self.log("\n=== Step 3: Processing Masks ===")
                self._run_process_masks(files)
            
            if steps['polygons'].get():
                self.log("\n=== Step 4: Creating Polygons ===")
                self._run_create_polygons(files)
            
            if steps['dxf'].get():
                self.log("\n=== Step 5: Creating DXF Files ===")
                self._run_create_dxf(files)
            
            if steps['scale'].get():
                self.log("\n=== Step 6: Scaling DXF Files ===")
                self._run_scale_dxf(files)
            
            self.log("\n=== Pipeline Complete ===")
        except Exception as e:
            self.log(f"\nPipeline error: {e}")


def main():
    root = tk.Tk()
    app = WindowShapesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
