import traceback
from typing import TYPE_CHECKING
import threading

import numpy as np

import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.engine.frame_engine import FrameEngine

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class FrameWorker(FrameEngine, threading.Thread):
    """Desktop adapter around the Qt-free FrameEngine.

    Reads live widget state off main_window, hands the engine a plain control
    dict, then delivers the processed frame back to the UI via Qt signals.
    All the heavy compute lives in FrameEngine."""

    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        threading.Thread.__init__(self)
        FrameEngine.__init__(self, main_window.models_processor, target_faces=main_window.target_faces)
        self.frame_queue = frame_queue
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.compare_images = []

    def run(self):
        try:
            # Update parameters from markers (if exists) without concurrent access from other threads
            with self.main_window.models_processor.model_lock:
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            self.parameters = self.main_window.parameters.copy()
            self.default_parameters = self.main_window.default_parameters
            self.target_faces = self.main_window.target_faces

            # Build the plain control dict the engine consumes, folding in the
            # button states the engine previously read straight off widgets.
            control = self.main_window.control.copy()
            control['SwapFacesEnabled'] = self.main_window.swapfacesButton.isChecked()
            control['EditFacesEnabled'] = self.main_window.editFacesButton.isChecked()
            self.control = control

            # Check if view mask or face compare checkboxes are checked
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked()
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked()

            # Process the frame with model inference
            if control['SwapFacesEnabled'] or control['EditFacesEnabled'] or control['FrameEnhancerEnableToggle']:
                self.frame = self.process_frame(self.frame)
            else:
                # Img must be in BGR format
                self.frame = self.frame[..., ::-1]  # Swap the channels from RGB to BGR
            self.frame = np.ascontiguousarray(self.frame)

            # Display the frame if processing is still active
            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            # Output processed Webcam frame
            if self.video_processor.file_type=='webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)

            #Output Video frame (while playing)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            # Output Image/Video frame (Single frame)
            else:
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)

            # Mark the frame as done in the queue
            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            # Check if playback is complete
            if self.video_processor.frame_queue.empty() and not self.video_processor.processing and self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error in FrameWorker: {e}")
            traceback.print_exc()
