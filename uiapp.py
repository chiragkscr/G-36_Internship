from ultralytics import YOLO
import cv2
import torch
import time
import supervision as sv
import numpy as np
import os
import tkinter as tk 
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import shutil
import threading
import queue

class TrackerUI:
    def __init__(self, root, comm_queue):
        self.root = root
        self.comm_queue = comm_queue
        self.current_image = None

        # UI Setup
        self.root.title("Object Tracker")
        self.root.geometry("800x600")

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_file)
        self.upload_button.pack(pady=5)

        self.remove_button = tk.Button(root, text="Remove Image", command=self.remove_reference)
        self.remove_button.pack(pady=5)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img.copy()
            
            # Send image through queue
            self.comm_queue.put(("NEW_REFERENCE", img))
            
            # Display preview
            display_img = Image.fromarray(img).resize((150, 150))
            img_tk = ImageTk.PhotoImage(display_img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def remove_reference(self):
        self.comm_queue.put(("REMOVE_REFERENCE", None))
        self.img_label.config(image="")
        self.current_image = None

class Tracker:
    def __init__(self, model_path, frame_size):
        self.model = YOLO(model_path, task='detect')
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=150, 
            minimum_matching_threshold=0.8
        )
        self.frame_size = frame_size
        self.target_class_id = None
        self.reference_image = None
        self.best_track_id = None
        self.lock = threading.Lock()

    def update_reference(self, new_image):
        with self.lock:
            self.reference_image = new_image
            
            self.best_track_id = None
            if new_image is not None:
                self._process_new_reference()

    def _process_new_reference(self):
        with torch.no_grad():
            resized_img = cv2.resize(self.reference_image, self.frame_size)
            results = self.model.predict(resized_img, conf=0.1)
        if len(results[0].boxes.cls) > 0:
            self.target_class_id = int(results[0].boxes.cls[0].cpu().numpy())
        else:
            self.target_class_id = None

    def process_frame(self, frame):
        with self.lock:
            if self.target_class_id is None or self.reference_image is None:
                return frame

            results = self.model.predict(frame, conf=0.5 , classes=self.target_class_id)
            detections = sv.Detections.from_ultralytics(results[0])
            tracked_objects = self.tracker.update_with_detections(detections)

            if self.best_track_id is not None and self.best_track_id not in tracked_objects.tracker_id:
                self.best_track_id = None
            
            if self.best_track_id is None and len(tracked_objects.xyxy) > 0:
                best_idx, _ = self.find_best_match(self.reference_image, tracked_objects.xyxy, frame)
                if best_idx is not None:
                    self.best_track_id = tracked_objects.tracker_id[best_idx]

            
            for idx, bbox in enumerate(tracked_objects.xyxy):
                track_id = tracked_objects.tracker_id[idx]
                color = (0, 255, 0) 
                if self.best_track_id == track_id:
                    color = (255, 0, 0)
                    x1, y1, x2, y2 = map(int, bbox)
                    r = 10
                    thickness =2
                    cv2.line(frame, (x1, y1), (x1 + r, y1), color, thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + r), color, thickness)
                    cv2.line(frame, (x2, y1), (x2 - r, y1), color, thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + r), color, thickness)
                    cv2.line(frame, (x1, y2), (x1 + r, y2), color, thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - r), color, thickness)
                    cv2.line(frame, (x2, y2), (x2 - r, y2), color, thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - r), color, thickness)
            return frame
    
    def find_best_match(self, reference_image, tracked_objects, frame):
        ref_hist = self.compute_histogram(reference_image)
        best_score = -1
        best_object = None

        for i, bbox in enumerate(tracked_objects):
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0,y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            hist2 = self.compute_histogram(crop)

            similarity = self.compare_histogram(ref_hist, hist2)
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_object = i

        if best_object is not None:
            return best_object, best_score
        return None, 0 
    
    def compute_histogram(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, (0,30,30),(180,255,255))
        hist_h = cv2.calcHist([image_hsv], [0], mask, [180], [0, 180])
        hist_s = cv2.calcHist([image_hsv], [1], mask, [256], [0, 256])
        hist_v = cv2.calcHist([image_hsv], [2], mask, [256], [0, 256])
        hist_h /= hist_h.sum() if hist_h.sum() > 0 else 1
        hist_s /= hist_s.sum() if hist_s.sum() > 0 else 1
        hist_v /= hist_v.sum() if hist_v.sum() > 0 else 1
        return [hist_h, hist_s, hist_v]
    
    def compare_histogram(self, hist1, hist2, method= cv2.HISTCMP_INTERSECT):
        total_score = 0
        for i in range(len(hist1)):
            score = cv2.compareHist(hist1[i], hist2[i], method)
            total_score += score
        return total_score / len(hist1)
            

    

class VideoApp:
    def __init__(self, model_path, video_path, frame_size, comm_queue):
        self.tracker = Tracker(model_path, frame_size)
        self.frame_size = frame_size
        self.cap = cv2.VideoCapture(video_path)
        self.comm_queue = comm_queue
        self.running = True

    def run(self):
        while self.running and self.cap.isOpened():
            # Handle communication from UI
            try:
                msg = self.comm_queue.get_nowait()
                if msg[0] == "NEW_REFERENCE":
                    self.tracker.update_reference(msg[1])
                elif msg[0] == "REMOVE_REFERENCE":
                    self.tracker.update_reference(None)
            except queue.Empty:
                pass

            # Process video frame
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, self.frame_size)
            processed_frame = self.tracker.process_frame(frame)
            
            cv2.imshow("Tracking", processed_frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    comm_queue = queue.Queue()
    
    root = TkinterDnD.Tk()
    ui = TrackerUI(root, comm_queue)
    
    video_path = r"128195-740906972_tiny.mp4"
    model_path = r"C:\Users\Chirag S\Desktop\G-36_Internship\visdronebest.pt"
    
    video_app = VideoApp(
        model_path=model_path,
        video_path=video_path,
        frame_size=(1280, 720),
        comm_queue=comm_queue
    )

    video_thread = threading.Thread(target=video_app.run, daemon=True)
    video_thread.start()

    def on_closing():
        video_app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()