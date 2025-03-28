import cv2
from ultralytics import YOLO
import supervision as sv
import time
import torch
import numpy as np


class KalmanTracker:
    def __init__(self, bbox=None):
        self.kf = cv2.KalmanFilter(8, 4)  # 8 state vars (x, y, w, h, vx, vy, vw, vh), 4 measurements (x, y, w, h)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)  # Maps measurement to state
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)  # State transition
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1  # Add velocity terms
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03  # Small process noise

        # Initialize state with bbox (x1, y1, x2, y2)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            self.kf.statePre[:4] = np.array([[x1], [y1], [w], [h]], np.float32)  # Set initial position
            self.kf.statePost[:4] = np.array([[x1], [y1], [w], [h]], np.float32)  # Ensure post state matches

        self.last_prediction = np.array([[x1], [y1], [x2], [y2]], np.float32) if bbox else np.zeros((4, 1), np.float32)

    def update(self, bbox):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            measurement = np.array([[np.float32(x1)], [np.float32(y1)], [np.float32(w)], [np.float32(h)]])
            self.kf.correct(measurement)  # Update Kalman filter with new measurement

        prediction = self.kf.predict()  # Predict next state
        self.last_prediction = prediction[:4]  # Save last prediction
        x1_pred, y1_pred, w_pred, h_pred = prediction[:4]
        return [int(x1_pred), int(y1_pred), int(x1_pred + w_pred), int(y1_pred + h_pred)]  # Return corrected bbox

    
class SmootedBBo:
    def __init__(self,alpha=0.2):
        self.alpha = alpha 
        self.smooted_bbox = None
    
    def update(self, bbox):
        if self.smooted_bbox is None:
            self.smooted_bbox = bbox
        else:
            self.smooted_bbox = [
                self.alpha  * bbox[i] + (1-self.alpha) * self.smooted_bbox[i] for i in range(4)
            ]
        return self.smooted_bbox

class InterpolatedBBox:
    def __init__(self):
        self.prev_bbox = None
        self.missing_frames = 0

    def update(self, bbox):
        if bbox is None:
            if self.prev_bbox is not None:
                self.missing_frames +=1
                interp_bbox = [
                    self.prev_bbox[i] + (i%2)*5*self.missing_frames
                    for i in range(4)
                ]
                return interp_bbox
            return None
        else:
            self.missing_frames = 0
            self.prev_bbox = bbox
            return bbox


class Tracker:
    def __init__(self, model_path, frame_size):
        self.model = YOLO(model_path, task = 'detect')
        # print("model Loaded")
        # self.model.to(torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=300, 
            minimum_matching_threshold=0.7
            
        )
        self.frame_size = frame_size
    
    def detect_frames(self, frames):
        # batch_size = 20
        detections = []
        with torch.no_grad():
            resized_frames = [cv2.resize(frame, self.frame_size)for frame in frames]
            # for i in range(0, len(resized_frames)):
            #     detection_batch = self.model.predict(resized_frames[i:i+batch_size], conf=0.1)
            #     detections += detection_batch
            detection_batch = self.model.predict(source = resized_frames, conf=0.5)
            detections += detection_batch
        return detections

    def get_objects_tracks(self, frame):
        start_time = time.time()
        detection = self.detect_frames([frame])[0]
        detection_supervision = sv.Detections.from_ultralytics(detection)
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
        end_time = time.time()
        processing_time = 1/(end_time-start_time)
        # print(processing_time)
        return detection_with_tracks


class VideoApp:
    def __init__(self ,model_path, video_path, frame_size):
        # Initialize the tracker
        self.tracker = Tracker(model_path, frame_size)
        self.frame_size = frame_size
        self.width, self.height = frame_size
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.selected_track_id = None
        self.current_frame = None
        self.current_detections = None
        self.frame_times = []
        self.trackers= {}
        self.smoothers ={}
        self.interpolators= {}
        

        # object selection
        cv2.namedWindow("Interactive Object Tracker")
        cv2.setMouseCallback("Interactive Object Tracker", self.select_object)

    def select_object(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            if self.current_detections is not None:
                
                for det in self.current_detections:
                    bbox = det[0]
                    track_id = det[4]
                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                        self.selected_track_id = track_id
                        print(f"Selected track ID: {track_id}")
                        break

    def run(self):
        prev_time = time.time()
        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Video ended or failed to capture.")
                break

            frame = cv2.resize(frame, self.frame_size)
            # print(frame.shape)
            self.current_frame = frame.copy()

            
            self.current_detections = self.tracker.get_objects_tracks(self.current_frame)

            
            for det in self.current_detections:
                bbox = det[0]
                track_id = det[4]
                print(track_id)

                # smooted_bbox = bbox
                
                
                color = (0, 0, 255)  
                if self.selected_track_id == track_id:
                    color = (0, 255, 0)  
                    r=10
                    thickness =1
                    d=0

                    (x1,y1,x2,y2) = map(int, bbox)

                    # if self.selected_track_id not in self.trackers:
                    #     self.trackers[track_id] = KalmanTracker()
                    
                    # kalman_bbox= self.trackers[track_id].update(bbox)
                    
                    # if self.selected_track_id not in self.smoothers:
                    #     self.smoothers[track_id] = SmootedBBo()

                    
                    # smooted_bbox = self.smoothers[track_id].update(bbox)

                    
                    

                    # if self.selected_track_id not in self.interpolators:
                    #     self.interpolators[track_id] = InterpolatedBBox()
                    
                    # smooted_bbox = self.interpolators[track_id].update(smooted_bbox)
                    cv2.line(frame, (int(bbox[0]) + r, int(bbox[1])), (int(bbox[0]) + r + d, int(bbox[1])), color, thickness)
                    cv2.line(frame, (int(bbox[0]), int(bbox[1]) + r), (int(bbox[0]), int(bbox[1]) + r + d), color, thickness)
                    cv2.line(frame, (int(bbox[0])+r, int(bbox[1])), (int(bbox[0]), int(bbox[1])), color, thickness )
                    cv2.line(frame, (int(bbox[0]), int(bbox[1])+r), (int(bbox[0]), int(bbox[1])), color, thickness )
                    # cv2.ellipse(frame, (int(bbox[0]) + r, int(bbox[1]) + r), (r, r), 180, 0, 90, color, thickness)

                    # Top right
                    # cv2.line(frame, (int(bbox[2]) - r, int(bbox[1])), (int(bbox[2]) - r - d, int(bbox[1])), color, thickness)
                    # cv2.line(frame, (int(bbox[2]), int(bbox[1] + r)), (int(bbox[2]), int(bbox[1]) + r + d), color, thickness)
                    # cv2.ellipse(frame, (int(bbox[2]) - r, int(bbox[1]) + r), (r, r), 270, 0, 90, color, thickness)
                    cv2.line(frame, (int(bbox[2])-r, int(bbox[1])), (int(bbox[2]), int(bbox[1])), color, thickness)
                    cv2.line(frame, (int(bbox[2]), int(bbox[1])+r), (int(bbox[2]), int(bbox[1])), color, thickness)

                    # Bottom left
                    # cv2.line(frame, (int(bbox[0]) + r, int(bbox[3])), (int(bbox[0]) + r + d, int(bbox[3])), color, thickness)
                    # cv2.line(frame, (int(bbox[0]), int(bbox[3]) - r), (int(bbox[0]), int(bbox[3]) - r - d), color, thickness)
                    # cv2.ellipse(frame, (int(bbox[0]) + r, int(bbox[3]) - r), (r, r), 90, 0, 90, color, thickness)
                    cv2.line(frame, (int(bbox[0])+r, int(bbox[3])), (int(bbox[0]), int(bbox[3])), color, thickness)
                    cv2.line(frame, (int(bbox[0]), int(bbox[3])-r), (int(bbox[0]), int(bbox[3])), color, thickness)


                    # Bottom right
                    # cv2.line(frame, (int(bbox[2]) - r, int(bbox[3])), (int(bbox[2]) - r - d, int(bbox[3])), color, thickness)
                    # cv2.line(frame, (int(bbox[2]), int(bbox[3]) - r), (int(bbox[2]), int(bbox[3]) - r - d), color, thickness)
                    # cv2.ellipse(frame, (int(bbox[2]) - r, int(bbox[3]) - r), (r, r), 0, 0, 90, color, thickness)
                    cv2.line(frame, (int(bbox[2])-r, int(bbox[3])), (int(bbox[2]), int(bbox[3])), color, thickness)
                    cv2.line(frame, (int(bbox[2]), int(bbox[3])-r), (int(bbox[2]), int(bbox[3])), color, thickness)
                # else:

                    #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    
                    # cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            

            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            if len(self.frame_times)>10:
                self.frame_times.pop(0)

            avg_fps = 1/(processing_time)
            # print(avg_fps)
            #cv2.putText(frame, f"FPS: {avg_fps: .1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("Interactive Object Tracker", frame)

            
            if cv2.waitKey(40) & 0xFF == ord('q'):

                break

        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    
    video_path = "/home/zas/tracking/128195-740906972_tiny.mp4"
    model_path = "/home/zas/tracking/visdronebest.pt"
    
    app = VideoApp(model_path, video_path, frame_size=(1280,720))
    app.run()
