import sys
import os
import threading
import time
import random
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QCheckBox, QWidget, QFileDialog, QTextEdit, QSpinBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import QTimer, Qt, QRect
from psychopy import visual, core, monitors
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2  # Needed for image processing functions like applyColorMap

# Import everything from vmbpy
from vmbpy import *


# Helper Functions
def generate_random_string(length=6):
    return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=length))

# Custom QLabel to handle ROI selection
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rect = None
        self.roi_callback = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            self.rect = QRect(self.start_point, self.end_point).normalized()
            if self.roi_callback:
                self.roi_callback(self.rect)
            self.update()

    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        if self.drawing and self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect.normalized())

# GUI Application Class
class ExperimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize variables
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.acquisition_thread = None
        self.stop_acquisition = threading.Event()
        self.frames_list = []
        self.frames_data = []
        self.current_exp_data = None
        self.stimuli_list = []
        self.stop_experiment = False
        self.condition_data = {}  # To store cumulative data per stimulus condition
        self.global_clock = core.MonotonicClock()  # Initialize a global clock
        self.roi = None  # Region of Interest
        self.full_frame_size = None  # Full frame size
        self.current_stimulus_frame = None  # Stores the current stimulus frame for the pseudo camera
        self.stimulus_window = self.create_stimulus_window()  # Create a persistent stimulus window

        # Start frame acquisition thread
        self.start_acquisition_thread()

        # Start timer for updating the GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def initUI(self):
        # Main Horizontal Layout
        main_layout = QHBoxLayout()

        # Left Side Layout (Controls and Camera Feed)
        left_layout = QVBoxLayout()

        # Camera Feed
        self.camera_label = ImageLabel(self)  # Use custom ImageLabel
        self.camera_label.setFixedSize(400, 300)  # Set fixed size
        self.camera_label.setAlignment(Qt.AlignCenter)  # Center alignment
        left_layout.addWidget(self.camera_label)

        # Animal ID
        self.animal_id_label = QLabel("Animal ID:")
        self.animal_id_text = QLineEdit(self)
        self.animal_id_text.setText("adam")
        left_layout.addWidget(self.animal_id_label)
        left_layout.addWidget(self.animal_id_text)

        # Heat/Gray Switch
        self.heat_gray_checkbox = QCheckBox("Heat/Gray", self)
        left_layout.addWidget(self.heat_gray_checkbox)

        # Experiment ID
        self.exp_id_label = QLabel("Experiment ID:")
        self.exp_id_value = QLabel(generate_random_string())
        left_layout.addWidget(self.exp_id_label)
        left_layout.addWidget(self.exp_id_value)

        # Load Stimulus Configuration Button
        self.load_button = QPushButton("LOAD Stimulus Configuration", self)
        self.load_button.clicked.connect(self.load_stimulus_config)
        left_layout.addWidget(self.load_button)

        # Number of Repeats
        self.repeats_label = QLabel("Number of Repeats:")
        self.repeats_spinbox = QSpinBox()
        self.repeats_spinbox.setValue(1)
        left_layout.addWidget(self.repeats_label)
        left_layout.addWidget(self.repeats_spinbox)

        # Randomize Order Checkbox
        self.randomize_checkbox = QCheckBox("Randomize Order")
        left_layout.addWidget(self.randomize_checkbox)

        # Inter-trial Interval
        self.iti_label = QLabel("Inter-Trial Interval (seconds):")
        self.iti_text = QLineEdit(self)
        self.iti_text.setText("1")
        left_layout.addWidget(self.iti_label)
        left_layout.addWidget(self.iti_text)

        # Button Layout
        button_layout = QHBoxLayout()

        # RUN Button
        self.run_button = QPushButton("RUN", self)
        self.run_button.clicked.connect(self.run_experiment)
        button_layout.addWidget(self.run_button)

        # SET ROI Button
        self.set_roi_button = QPushButton("SET ROI", self)
        self.set_roi_button.clicked.connect(self.set_roi)
        button_layout.addWidget(self.set_roi_button)

        # Save Current View Button
        self.save_view_button = QPushButton("Save Current View", self)
        self.save_view_button.clicked.connect(self.save_current_view)
        button_layout.addWidget(self.save_view_button)

        # Add button layout to left layout
        left_layout.addLayout(button_layout)

        # STOP Button
        self.stop_button = QPushButton("STOP", self)
        self.stop_button.clicked.connect(self.stop_experiment_func)
        left_layout.addWidget(self.stop_button)

        # Load Data Button
        self.load_data_button = QPushButton("LOAD DATA", self)
        self.load_data_button.clicked.connect(self.load_data)
        left_layout.addWidget(self.load_data_button)

        # Feedback Listbox
        self.feedback_textbox = QTextEdit(self)
        self.feedback_textbox.setReadOnly(True)
        self.feedback_textbox.setMaximumHeight(150)
        left_layout.addWidget(self.feedback_textbox)

        # Left Layout Stretch to take up space
        left_layout.addStretch()

        # Right Side Layout (Analysis Plots)
        right_layout = QVBoxLayout()

        self.analysis_label = QLabel("Analysis Plots:")
        right_layout.addWidget(self.analysis_label)

        # Analysis Plots Layout
        self.analysis_layout = QVBoxLayout()
        right_layout.addLayout(self.analysis_layout)

        # Right Layout Stretch to take up space
        right_layout.addStretch()

        # Add left and right layouts to the main layout
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)

        self.setLayout(main_layout)
        self.setWindowTitle("Experiment Control GUI")
        self.setGeometry(100, 100, 1200, 800)

    def create_stimulus_window(self):
        """Create a PsychoPy stimulus window and keep it open throughout the program."""
        mon = monitors.Monitor('testMonitor')
        mon.setWidth(53.0)
        mon.setDistance(60.0)
        mon.setSizePix((1920, 1080))
        win = visual.Window(size=mon.getSizePix(), fullscr=True, monitor=mon, units="deg", screen=0, color=[0, 0, 0])
        return win

    def start_acquisition_thread(self):
        self.stop_acquisition.clear()
        self.acquisition_thread = threading.Thread(target=self.acquire_frames)
        self.acquisition_thread.start()

    def acquire_frames(self):
        # Simulate the case where no camera is available
        simulated_camera = False
        
        with VmbSystem.get_instance() as vmb:
            cams = vmb.get_all_cameras()
            if not cams:
                self.feedback_textbox.append("No Allied Vision cameras found. Switching to simulated camera.")
                simulated_camera = True
            else:
                cam = cams[0]
            
            if simulated_camera:
                # Start the simulated camera loop
                while not self.stop_acquisition.is_set():
                    try:
                        # If no current stimulus, start with a blank 2D grayscale image
                        if self.current_stimulus_frame is None:
                            last_display_frame = np.zeros((300, 400), dtype=np.uint8)  # Blank grayscale image
                        else:
                            last_display_frame = self.current_stimulus_frame.copy()  # 2D array

                        # Add noise (10% of the max pixel value)
                        noise = np.random.normal(0, 0.1 * 255, last_display_frame.shape).astype(np.int16)
                        simulated_frame = np.clip(last_display_frame + noise, 0, 255).astype(np.uint8)
                        
                        # Save the frame as the latest frame (2D)
                        timestamp = self.global_clock.getTime()
                        with self.frame_lock:
                            self.latest_frame = (simulated_frame.copy(), timestamp)
                        
                        # Store the frames for the experiment if running
                        if not self.stop_experiment:
                            self.frames_list.append((simulated_frame.copy(), timestamp))
                        
                        # Add a small delay to simulate frame rate
                        time.sleep(1/30.0)  # 30 fps equivalent delay
                    except Exception as e:
                        self.feedback_textbox.append(f"Simulated frame generation failed. Error: {e}")
                        break
            else:
                # Actual camera loop if the camera is available
                with cam:
                    settings_file = 'C:/Users/ranso/OneDrive - UAB/Code/repos/ISI_Standalone/cam_settings.xml'
                    cam.load_settings(settings_file, PersistType.All)

                    while not self.stop_acquisition.is_set():
                        try:
                            frame = cam.get_frame()
                            frame_np = frame.as_numpy_ndarray().squeeze()
                            image_array = frame_np  # Already 2D since it's a grayscale image

                            # Apply ROI if set
                            if self.roi:
                                x, y, w, h = self.roi
                                image_array = image_array[y:y+h, x:x+w]
                            else:
                                # Store full frame size
                                if self.full_frame_size is None:
                                    self.full_frame_size = image_array.shape[::-1]  # (width, height)

                            timestamp = self.global_clock.getTime()

                            # Store the latest frame (2D)
                            with self.frame_lock:
                                self.latest_frame = (image_array.copy(), timestamp)

                            # If experiment is running, store frames
                            if not self.stop_experiment:
                                self.frames_list.append((image_array.copy(), timestamp))

                        except Exception as e:
                            self.feedback_textbox.append(f"Frame capture failed. Error: {e}")
                            break

    def update_frame(self):
        # Retrieve the latest frame for display
        with self.frame_lock:
            if self.latest_frame is not None:
                image_array, _ = self.latest_frame

                # Convert image to uint8 if necessary (ensure it's 2D)
                if image_array.dtype != np.uint8:
                    image_array_display = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
                    image_array_display = image_array_display.astype(np.uint8)
                else:
                    image_array_display = image_array.copy()

                if self.heat_gray_checkbox.isChecked():
                    image_array_display = cv2.applyColorMap(image_array_display, cv2.COLORMAP_JET)
                    height, width = image_array_display.shape[:2]
                    bytes_per_line = 3 * width
                    qimg = QImage(image_array_display.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    height, width = image_array_display.shape  # 2D array
                    bytes_per_line = width
                    qimg = QImage(image_array_display.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(qimg)

                # Scale the pixmap to fit within 400x300 while maintaining aspect ratio
                pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Create a new pixmap with size 400x300 for padding
                final_pixmap = QPixmap(400, 300)
                final_pixmap.fill(Qt.black)  # Fill background with black

                # Draw the scaled pixmap onto the final pixmap centered
                painter = QPainter(final_pixmap)
                x = (400 - pixmap.width()) // 2
                y = (300 - pixmap.height()) // 2
                painter.drawPixmap(x, y, pixmap)
                painter.end()

                # Set the final pixmap to the label
                self.camera_label.setPixmap(final_pixmap)

    def set_roi(self):
        # Reset ROI to full frame
        self.roi = None
        self.full_frame_size = None
        self.camera_label.setEnabled(True)
        self.feedback_textbox.append("Please select ROI by dragging over the image.")
        QApplication.processEvents()
        # Set the ROI callback
        self.camera_label.roi_callback = self.roi_selected

    def roi_selected(self, rect):
        # Disable further ROI selection
        self.camera_label.setEnabled(False)
        # Get ROI coordinates
        with self.frame_lock:
            if self.latest_frame is not None:
                image_array, _ = self.latest_frame
                frame_height, frame_width = image_array.shape
        label_width = self.camera_label.width()
        label_height = self.camera_label.height()

        # Calculate scaling factors
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height

        # Adjust ROI to image coordinates
        x = int(rect.left() * scale_x)
        y = int(rect.top() * scale_y)
        w = int(rect.width() * scale_x)
        h = int(rect.height() * scale_y)

        # Ensure ROI is within image bounds
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))

        self.roi = (x, y, w, h)
        self.feedback_textbox.append(f"ROI set to x:{x}, y:{y}, w:{w}, h:{h}")
        QApplication.processEvents()

    def save_current_view(self):
        # Save the current displayed frame
        animal_id = self.animal_id_text.text()
        save_dir = os.path.join("C:\\local_repository", animal_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prompt for filename in animal directory
        options = QFileDialog.Options()
        initial_file_name = os.path.join(save_dir, f"current_view_{generate_random_string()}.png")
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Current View", initial_file_name, "Images (*.png *.jpg *.bmp)", options=options)
        if file_name:
            # Copy the latest frame while holding the lock
            with self.frame_lock:
                if self.latest_frame is not None:
                    image_array, _ = self.latest_frame
                    image_array = image_array.copy()
                else:
                    image_array = None

            if image_array is not None:
                # Process the image outside the lock
                if self.heat_gray_checkbox.isChecked():
                    image_array = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)

                # Convert image to uint8 if necessary
                if image_array.dtype != np.uint8:
                    # Scale the image data to 0-255 and convert to uint8
                    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
                    image_array = image_array.astype(np.uint8)

                try:
                    cv2.imwrite(file_name, image_array)
                    self.feedback_textbox.append(f"Current view saved to {file_name}")
                except Exception as e:
                    self.feedback_textbox.append(f"Error saving image: {e}")
            else:
                self.feedback_textbox.append("No frame available to save.")
            QApplication.processEvents()

    def load_stimulus_config(self):
        self.feedback_textbox.append("Loading stimulus configuration...")
        QApplication.processEvents()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Stimulus Configuration File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                self.stimuli_list = []
                for idx, line in enumerate(lines):
                    tokens = line.strip().split('\t')
                    # Ensure tokens have all required parameters
                    while len(tokens) < 10:
                        tokens.append('default')  # Ensure all fields are loaded
                    stimulus = {
                        'index': idx,
                        'sf': tokens[0],
                        'tf': tokens[1],
                        'xpos': tokens[2],
                        'ypos': tokens[3],
                        'sizex': tokens[4],
                        'sizey': tokens[5],
                        'orientation': tokens[6] if tokens[6] != 'default' else '0',  # Default orientation is 0 degrees
                        'duty_cycle': tokens[7],
                        'on_time': tokens[8],
                        'stim_type': tokens[9]
                    }
                    self.stimuli_list.append(stimulus)
            self.feedback_textbox.append("Stimulus configuration loaded successfully.")
        QApplication.processEvents()

    def run_experiment(self):
        self.stop_experiment = False

        # Clear previous data
        self.frames_data.clear()
        self.condition_data.clear()
        self.frames_list.clear()

        # Clear analysis plots from GUI
        self.clear_analysis_plots()

        # Generate a new experiment ID
        exp_id = generate_random_string()
        self.exp_id_value.setText(exp_id)
        self.feedback_textbox.append(f"New Experiment ID set: {exp_id}")
        QApplication.processEvents()

        animal_id = self.animal_id_text.text()

        # Directory Setup
        save_dir = os.path.join("C:\\local_repository", animal_id, exp_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Wait for at least one frame
        while self.latest_frame is None:
            time.sleep(0.01)

        # Run Through Stimulus List
        repeats = self.repeats_spinbox.value()
        randomize = self.randomize_checkbox.isChecked()
        iti = float(self.iti_text.text())

        total_number_of_trials = repeats * len(self.stimuli_list)

        for _ in range(repeats):
            order = list(range(len(self.stimuli_list)))
            if randomize:
                random.shuffle(order)
            for i in order:
                if self.stop_experiment:
                    break
                stimulus = self.stimuli_list[i]
                sf = stimulus['sf']
                tf = stimulus['tf']
                xpos = stimulus['xpos']
                ypos = stimulus['ypos']
                sizex = stimulus['sizex']
                sizey = stimulus['sizey']
                orientation = stimulus['orientation']
                duty_cycle = stimulus['duty_cycle']
                on_time = stimulus['on_time']
                stim_type = stimulus['stim_type']
                stimulus_index = stimulus['index']
                self.run_single_trial(sf, tf, xpos, ypos, sizex, sizey, duty_cycle, on_time, stim_type,
                                      save_dir, exp_id, iti, repeats, stimulus_index, orientation)

        self.stop_experiment = True

    def clear_analysis_plots(self):
        # Clear previous plots
        for i in reversed(range(self.analysis_layout.count())):
            item = self.analysis_layout.itemAt(i)
            if item.layout():
                for j in reversed(range(item.layout().count())):
                    widget_to_remove = item.layout().itemAt(j).widget()
                    if widget_to_remove is not None:
                        widget_to_remove.setParent(None)
                self.analysis_layout.removeItem(item)
            else:
                widget_to_remove = item.widget()
                if widget_to_remove is not None:
                    widget_to_remove.setParent(None)
                self.analysis_layout.removeItem(item)
        self.analysis_layout.update()
        self.update()

    def stop_experiment_func(self):
        self.stop_experiment = True

    def load_data(self):
        animal_id = self.animal_id_text.text()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Experiment Data File", f"C:\\local_repository\\{animal_id}", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            self.current_exp_data = pd.read_csv(file_name, delimiter='\t')
            print("Loaded Data:", self.current_exp_data)

    def run_single_trial(self, sf, tf, xpos, ypos, sizex, sizey, duty_cycle, on_time, stim_type,
                         save_dir, exp_id, iti, repeats, stimulus_index, orientation):
        # Record the index in frames_list where this trial's frames start
        start_frame_index = len(self.frames_list)

        # PsychoPy Grating Setup (using the persistent window)
        position = (float(xpos), float(ypos))
        grating = self.create_grating(self.stimulus_window, float(sf), (float(sizex), float(sizey)), float(duty_cycle), float(tf), float(orientation), position)

        # Run Trial
        trial_start_time = None  # Will be set to the time of the first flip
        frames_in_trial = []
        first_flip = True
        while True:
            current_time = self.global_clock.getTime()
            if trial_start_time is not None and current_time - trial_start_time >= float(on_time):
                break
            grating.phase = (grating.tf * (current_time if trial_start_time is None else current_time - trial_start_time))
            grating.draw()

            # Capture the stimulus screen as an image array for the pseudo camera
            self.stimulus_window.getMovieFrame(buffer='back')  # Capture the current frame
            stimulus_frame = self.stimulus_window._getFrame(buffer='back')  # Get the actual pixel data of the frame

            # Convert stimulus_frame to a NumPy array
            stimulus_frame_array = np.array(stimulus_frame)

            # Convert to 2D grayscale if the captured frame is not already grayscale
            if stimulus_frame_array.ndim == 3 and stimulus_frame_array.shape[2] == 3:  # Check if it's 3D (i.e., colored)
                self.current_stimulus_frame = cv2.cvtColor(stimulus_frame_array, cv2.COLOR_BGR2GRAY)
            else:
                self.current_stimulus_frame = stimulus_frame_array.squeeze()  # Keep as 2D grayscale if already single-channel

            flip_time = self.stimulus_window.flip()  # Returns the precise time of the flip
            if first_flip:
                trial_start_time = flip_time  # Set trial_start_time to the time of the first flip
                first_flip = False
            frames_in_trial.append(flip_time - trial_start_time)  # Adjust flip times relative to trial_start_time

        # Get the frames corresponding to this trial
        trial_frames = self.frames_list[start_frame_index:]

        # Adjust timestamps to be relative to trial_start_time
        adjusted_trial_frames = [(frame, timestamp - trial_start_time) for frame, timestamp in trial_frames if timestamp >= trial_start_time]

        # Debugging: Output the number of frames and timestamps
        self.feedback_textbox.append(f"Adjusted trial frames: {len(adjusted_trial_frames)} frames")
        if adjusted_trial_frames:
            self.feedback_textbox.append(f"First frame timestamp: {adjusted_trial_frames[0][1]:.4f}s")
            self.feedback_textbox.append(f"Last frame timestamp: {adjusted_trial_frames[-1][1]:.4f}s")
        else:
            self.feedback_textbox.append("No frames captured during this trial.")

        QApplication.processEvents()

        # Store Trial Data
        trial_data = {
            'stimulus': {
                'sf': sf,
                'tf': tf,
                'xpos': xpos,
                'ypos': ypos,
                'sizex': sizex,
                'sizey': sizey,
                'duty_cycle': duty_cycle,
                'on_time': on_time,
                'stim_type': stim_type,
                'orientation': orientation,
                'index': stimulus_index
            },
            'frames': adjusted_trial_frames.copy(),
            'flip_times': frames_in_trial
        }
        self.frames_data.append(trial_data)

        # Analysis step after the trial if stim_type is '0'
        if stim_type == '0':
            start_time_analysis = time.time()
            # Perform the averaged analysis for this condition
            self.perform_averaged_analysis(adjusted_trial_frames, float(tf), save_dir, exp_id, stimulus_index)

            end_time_analysis = time.time()
            self.feedback_textbox.append(f"FFT analysis complete. Time taken: {end_time_analysis - start_time_analysis:.2f} seconds")
            QApplication.processEvents()

        # Save Trial Data
        save_path = os.path.join(save_dir, f'isi_{exp_id}.txt')
        with open(save_path, 'a') as file:
            file.write(f"{trial_data}\n")

        # Inter-trial Interval
        self.feedback_textbox.append(f"Completed trial {len(self.frames_data)} of {repeats * len(self.stimuli_list)}")
        QApplication.processEvents()
        core.wait(iti)

    def create_grating(self, win, sf, size, duty_cycle, tf, orientation, position):
        n_cycles = 100
        on_samples = int(n_cycles * duty_cycle)
        off_samples = n_cycles - on_samples
        texture = np.concatenate([np.ones(on_samples), -np.ones(off_samples)])
        texture = np.tile(texture, (1, 1))

        grating = visual.GratingStim(win=win,
                                     size=size,
                                     sf=(sf, 0),
                                     phase=0,
                                     tex=texture,
                                     units='deg',
                                     ori=orientation,
                                     pos=position)  # Set the position
        grating.tf = tf
        return grating

    def perform_averaged_analysis(self, trial_frames, tf, save_dir, exp_id, stimulus_index):
        """Perform the averaged analysis for stimulus type 0."""
        frames, timestamps = zip(*trial_frames)
        frames = np.array(frames)  # Convert frames to a NumPy array
        num_frames, height, width = frames.shape

        # Ensure that we accumulate frames for this stimulus condition
        if stimulus_index not in self.condition_data:
            self.condition_data[stimulus_index] = {
                'all_frames': [],  # To store frames across trials
                'trial_lengths': []  # To store the number of frames in each trial
            }

        # Add this trial's frames and record the number of frames
        self.condition_data[stimulus_index]['all_frames'].append(frames)
        self.condition_data[stimulus_index]['trial_lengths'].append(num_frames)

        # Determine the minimum length of trials for this stimulus condition
        min_length = min(self.condition_data[stimulus_index]['trial_lengths'])

        # Truncate all trials to the minimum length
        truncated_trials = [trial[:min_length] for trial in self.condition_data[stimulus_index]['all_frames']]

        # Calculate the average frame sequence
        accumulated_frames = np.stack(truncated_trials, axis=0)
        avg_frames = np.mean(accumulated_frames, axis=0)

        # Resample the averaged frames
        resampled_frames = self.resample_video(avg_frames, new_rate=10)

        # Perform FFT on the averaged frames
        power_map, phase_map = self.perform_fft(resampled_frames, tf)

        # Store the results in condition data
        self.condition_data[stimulus_index]['average_power_map'] = power_map
        self.condition_data[stimulus_index]['average_phase_map'] = phase_map

        # Update analysis plots
        self.update_analysis_plots()

        # Save analysis results
        plt.imshow(power_map, cmap='hot')
        plt.colorbar()
        plt.title('Power Map at Stimulus Frequency')
        plt.savefig(os.path.join(save_dir, f'power_map_{exp_id}_avg.png'))
        plt.close()

        plt.imshow(phase_map, cmap='hsv')
        plt.colorbar()
        plt.title('Phase Map at Stimulus Frequency')
        plt.savefig(os.path.join(save_dir, f'phase_map_{exp_id}_avg.png'))
        plt.close()

    def resample_video(self, frames, new_rate):
        """ Resample the frames to a new frame rate (e.g., 10 Hz) """
        num_samples, height, width = frames.shape

        # Original frame times assuming uniform spacing
        original_times = np.linspace(0, num_samples / new_rate, num_samples)

        # New frame times for resampling
        duration = original_times[-1]
        new_num_samples = int(duration * new_rate)
        new_times = np.linspace(0, duration, new_num_samples)

        frames_reshaped = frames.reshape(num_samples, -1)  # Shape: (num_samples, num_pixels)
        interp_func = interp1d(original_times, frames_reshaped, kind='linear', axis=0, fill_value='extrapolate')
        resampled_frames = interp_func(new_times).reshape(len(new_times), height, width)

        return resampled_frames

    def perform_fft(self, frames, tf):
        """ Perform FFT on the resampled frames and return the power and phase maps """
        from scipy.fft import fft, fftfreq
        num_samples, height, width = frames.shape
        fft_vals = fft(frames, axis=0)
        freqs = fftfreq(num_samples, 1 / 10)  # Assuming 10 Hz sampling rate
        idx = (np.abs(freqs - tf)).argmin()

        fft_at_tf = fft_vals[idx, :, :]
        fft_at_zero = fft_vals[0, :, :]  # DC component

        # Compute power and phase maps
        magnitude_zero = np.abs(fft_at_zero)
        magnitude_zero[magnitude_zero == 0] = np.finfo(float).eps

        power_map = np.abs(fft_at_tf) / magnitude_zero
        phase_map = np.angle(fft_at_tf)

        return power_map, phase_map

    def update_analysis_plots(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        # Clear previous plots
        for i in reversed(range(self.analysis_layout.count())):
            item = self.analysis_layout.itemAt(i)
            if item.layout():
                for j in reversed(range(item.layout().count())):
                    widget_to_remove = item.layout().itemAt(j).widget()
                    if widget_to_remove is not None:
                        widget_to_remove.setParent(None)
                self.analysis_layout.removeItem(item)
            else:
                widget_to_remove = item.widget()
                if widget_to_remove is not None:
                    widget_to_remove.setParent(None)
                self.analysis_layout.removeItem(item)

        # For each condition, create the average plots
        for condition_idx in sorted(self.condition_data.keys()):
            condition = self.condition_data[condition_idx]

            # Check if 'average_power_map' and 'average_phase_map' exist before accessing them
            if 'average_power_map' not in condition or 'average_phase_map' not in condition:
                self.feedback_textbox.append(f"Condition {condition_idx} does not have an average power or phase map yet.")
                continue

            avg_power_map = condition['average_power_map']
            avg_phase_map = condition['average_phase_map']
            trial_count = len(condition['all_frames'])

            # Create a horizontal layout for this condition
            condition_layout = QHBoxLayout()

            # Create power map plot
            fig_power, ax_power = plt.subplots(figsize=(4, 3))
            im_power = ax_power.imshow(avg_power_map, cmap='hot', aspect='auto')
            ax_power.set_title(f'Power Map - Condition {condition_idx} (n={trial_count})', fontsize=9)  # Reduced font size
            ax_power.axis('off')  # Remove axis ticks and labels
            fig_power.colorbar(im_power, ax=ax_power)
            fig_power.tight_layout()
            canvas_power = FigureCanvas(fig_power)
            canvas_power.setMinimumSize(200, 150)
            condition_layout.addWidget(canvas_power)

            # Create phase map plot
            fig_phase, ax_phase = plt.subplots(figsize=(4, 3))
            im_phase = ax_phase.imshow(avg_phase_map, cmap='hsv', aspect='auto')
            ax_phase.set_title(f'Phase Map - Condition {condition_idx} (n={trial_count})', fontsize=9)  # Reduced font size
            ax_phase.axis('off')  # Remove axis ticks and labels
            fig_phase.colorbar(im_phase, ax=ax_phase)
            fig_phase.tight_layout()
            canvas_phase = FigureCanvas(fig_phase)
            canvas_phase.setMinimumSize(200, 150)
            condition_layout.addWidget(canvas_phase)

            # Add the condition layout to the analysis layout
            self.analysis_layout.addLayout(condition_layout)

        # Refresh the GUI
        self.analysis_layout.update()
        self.update()
        self.feedback_textbox.append("Analysis plots updated.")
        QApplication.processEvents()

    def closeEvent(self, event):
        # Stop the acquisition thread
        self.stop_acquisition.set()
        if self.acquisition_thread is not None:
            self.acquisition_thread.join()
        # Close the stimulus window
        if self.stimulus_window:
            self.stimulus_window.close()
        event.accept()

# Main Application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExperimentApp()
    ex.show()
    sys.exit(app.exec_())
