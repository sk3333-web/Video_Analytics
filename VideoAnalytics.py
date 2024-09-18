import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import requests
import time
import threading
import queue
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectTracking:
    def __init__(self, stream_url, camera_id, store_id, csv_file_path_1, csv_file_path_2, csv_lock, display_queue,
                 show_video, enable_frame_skip, is_main_stream):
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.store_id = store_id
        self.csv_file_path_1 = csv_file_path_1
        self.csv_file_path_2 = csv_file_path_2
        self.csv_lock = csv_lock
        self.display_queue = display_queue
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip
        self.is_main_stream = is_main_stream

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, 'yolov8n.pt')
        self.bytetrack_yaml_path = 'bytetrack.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path).to(self.device)
        self.model.fuse()

        if self.is_main_stream:
            self.face_model = YOLO('yolov8n-face.pt').to(self.device)

        self.target_size = (640, 480)

        self.grid_rows = 6
        self.grid_cols = 6
        self.grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        self.start_time = time.time()
        self.interval = 60  # 1 minute in seconds for testing

        self.frame_skip = 3
        self.frame_count = 0

        self.frame_queue = queue.Queue(maxsize=3)

        if self.is_main_stream:
            # Demographics tracking (only for main stream)
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions = {}
            self.detected_persons = {}
            self.total_gender_age_count = {
                'Male': defaultdict(int),
                'Female': defaultdict(int)
            }
            self.roi_line = [(0, self.target_size[1] // 2), (self.target_size[0], self.target_size[1] // 2)]

    def save_counts(self):
        current_time = int(time.time())

        # CSV 1: Grid counts (for all streams)
        csv_row_1 = [self.store_id, current_time, self.camera_id] + list(self.grid_counts.flatten())

        with self.csv_lock:
            try:
                with open(self.csv_file_path_1, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row_1)

                if self.is_main_stream:
                    # CSV 2: Demographics (only for main stream)
                    male_counts = {k: self.total_gender_age_count['Male'][k] for k in self.total_gender_age_count['Male']}
                    female_counts = {k: self.total_gender_age_count['Female'][k] for k in self.total_gender_age_count['Female']}
                    csv_row_2 = [
                        self.store_id, current_time, self.camera_id,
                        self.total_count_in, self.total_count_out,
                        male_counts.get("0_18", 0), male_counts.get("19_24", 0), male_counts.get("25_35", 0),
                        male_counts.get("36_55", 0), male_counts.get("55_plus", 0),
                        female_counts.get("0_18", 0), female_counts.get("19_24", 0),
                        female_counts.get("25_35", 0), female_counts.get("36_55", 0),
                        female_counts.get("55_plus", 0)
                    ]
                    with open(self.csv_file_path_2, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(csv_row_2)

                logging.info(f"Data saved to CSVs for store {self.store_id}, camera {self.camera_id}")
            except Exception as e:
                logging.error(f"Error writing to CSVs for store {self.store_id}, camera {self.camera_id}: {e}")

        # Reset counts and tracked IDs
        self.grid_counts.fill(0)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        if self.is_main_stream:
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions.clear()
            self.detected_persons.clear()
            self.total_gender_age_count = {'Male': defaultdict(int), 'Female': defaultdict(int)}
        self.start_time = time.time()

    def classify_age(self, age):
        if age <= 18:
            return "0_18"
        elif 19 <= age <= 24:
            return "19_24"
        elif 25 <= age <= 35:
            return "25_35"
        elif 36 <= age <= 55:
            return "36_55"
        else:
            return "55_plus"

    def detect_demographics(self, person_crop, id):
        face_results = self.face_model(person_crop, device=self.device)
        if face_results and len(face_results[0].boxes) > 0:
            face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
            face = person_crop[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            try:
                analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False, silent=True)
                age = analysis[0]['age']
                age_class = self.classify_age(age)
                gender = 'Male' if analysis[0]['dominant_gender'] == 'Man' else 'Female'
                self.detected_persons[id] = {'age_class': age_class, 'gender': gender}
                self.total_gender_age_count[gender][age_class] += 1
                logging.info(f"Detected person {id}: Gender - {gender}, Age - {age}, Class - {age_class}")
            except Exception as e:
                logging.error(f"Error detecting demographics for person {id}: {e}")

    def line_crossing(self, prev_point, current_point, line_start, line_end):
        return (prev_point[1] <= line_start[1] and current_point[1] > line_start[1]) or (
                prev_point[1] > line_start[1] and current_point[1] <= line_start[1])

    def process_frame(self, frame):
        frame = cv2.resize(frame, self.target_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            try:
                results = self.model.track(
                    source=frame_tensor,
                    persist=True,
                    tracker=self.bytetrack_yaml_path,
                    classes=[0],
                    device=self.device
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id
                    if ids is not None:
                        ids = ids.cpu().numpy().astype(int)
                    else:
                        ids = []

                    for box, id in zip(boxes, ids):
                        if self.show_video:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)

                        center_point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                        grid_x = center_point[0] // (self.target_size[0] // self.grid_cols)
                        grid_y = center_point[1] // (self.target_size[1] // self.grid_rows)

                        if id not in self.grid_ids[grid_y][grid_x]:
                            self.grid_counts[grid_y, grid_x] += 1
                            self.grid_ids[grid_y][grid_x].add(id)

                        if self.is_main_stream:
                            # Check for line crossing and demographics (only for main stream)
                            if id in self.prev_positions:
                                prev_point = self.prev_positions[id]
                                if self.line_crossing(prev_point, center_point, self.roi_line[0], self.roi_line[1]):
                                    if center_point[1] > prev_point[1]:
                                        self.total_count_in += 1
                                        if id not in self.detected_persons:
                                            self.detect_demographics(frame[box[1]:box[3], box[0]:box[2]], id)
                                    else:
                                        self.total_count_out += 1

                            self.prev_positions[id] = center_point

                        if self.show_video:
                            if self.is_main_stream and id in self.detected_persons:
                                demographics = self.detected_persons[id]
                                cv2.putText(frame, f"Id{id}: {demographics['gender']}, {demographics['age_class']}",
                                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            else:
                                cv2.putText(frame, f"Id{id}", (box[0], box[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except Exception as e:
                logging.error(f"Error processing frame for store {self.store_id}, camera {self.camera_id}: {e}")

        if self.show_video:
            # Draw grid lines and display counts
            for i in range(1, self.grid_rows):
                y = i * (self.target_size[1] // self.grid_rows)
                cv2.line(frame, (0, y), (self.target_size[0], y), (0, 255, 0), 1)
            for i in range(1, self.grid_cols):
                x = i * (self.target_size[0] // self.grid_cols)
                cv2.line(frame, (x, 0), (x, self.target_size[1]), (0, 255, 0), 1)

            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    x = j * (self.target_size[0] // self.grid_cols) + 5
                    y = i * (self.target_size[1] // self.grid_rows) + 20
                    cv2.putText(frame, str(self.grid_counts[i, j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            if self.is_main_stream:
                # Draw ROI line and display counters (only for main stream)
                cv2.line(frame, self.roi_line[0], self.roi_line[1], (0, 255, 0), 2)
                cv2.putText(frame, f"In: {self.total_count_in}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Out: {self.total_count_out}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if time.time() - self.start_time >= self.interval:
            self.save_counts()

        return frame

    def process_stream(self):
        cap = cv2.VideoCapture(self.stream_url)

        if not cap.isOpened():
            logging.error(f"Failed to open camera stream for store {self.store_id}, camera {self.camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(
                    f"Failed to receive frame from store {self.store_id}, camera {self.camera_id}. Retrying...")
                time.sleep(1)
                continue

            self.frame_count += 1
            if self.enable_frame_skip and self.frame_count % self.frame_skip != 0:
                continue

            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                continue

            if self.frame_queue.qsize() > 0:
                frame_to_process = self.frame_queue.get()
                processed_frame = self.process_frame(frame_to_process)
                if self.show_video:
                    self.display_queue.put((self.camera_id, processed_frame))

        cap.release()

def display_frames(display_queue):
    windows = {}

    while True:
        try:
            camera_id, frame = display_queue.get(timeout=1)
            window_name = f"Store 1 - Camera {camera_id}"

            if window_name not in windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                windows[window_name] = True

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for window in windows:
        cv2.destroyWindow(window)

class MultiStreamObjectTracking:
    def __init__(self, stream_urls, store_id, show_video, enable_frame_skip):
        self.stream_urls = stream_urls
        self.store_id = store_id
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip
        self.csv_file_1 = 'grid_counts.csv'
        self.csv_file_2 = 'demographics.csv'
        self.csv_header_1 = ['Store_ID', 'Timestamp', 'Camera_ID'] + [f'Grid_{i}_{j}' for i in range(6) for j in
                                                                      range(6)]
        self.csv_header_2 = ['Store_ID', 'Timestamp', 'Camera_ID', 'Total_In', 'Total_Out',
                             'Male_0_18', 'Male_19_24', 'Male_25_35', 'Male_36_55', 'Male_55_plus',
                             'Female_0_18', 'Female_19_24', 'Female_25_35', 'Female_36_55', 'Female_55_plus']
        self.csv_lock = threading.Lock()
        self.display_queue = queue.Queue()

        # Ensure the CSV files are properly initialized
        with open(self.csv_file_1, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header_1)
        with open(self.csv_file_2, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header_2)
        logging.info(f"CSV files initialized: {self.csv_file_1}, {self.csv_file_2}")

    def run(self):
        threads = []
        for camera_id, stream_url in enumerate(self.stream_urls):
            is_main_stream = (camera_id == 0)  # First camera is the main stream
            ot = ObjectTracking(stream_url, camera_id, self.store_id, self.csv_file_1, self.csv_file_2,
                                self.csv_lock, self.display_queue, self.show_video, self.enable_frame_skip,
                                is_main_stream)
            thread = threading.Thread(target=ot.process_stream)
            threads.append(thread)
            thread.start()

        if self.show_video:
            display_thread = threading.Thread(target=display_frames, args=(self.display_queue,))
            display_thread.start()

        for thread in threads:
            thread.join()

        if self.show_video:
            display_thread.join()

        logging.info(f"All camera streams for store {self.store_id} have been processed")


def run_multi_stream_object_tracking(show_video, enable_frame_skip):
    rtsp_urls = [
        "video/rtsplinks",
        "video/rtsplinks",
    ]

    store_id = "1"  # You can change this or make it dynamic as needed
    multi_tracker = MultiStreamObjectTracking(rtsp_urls, store_id, show_video, enable_frame_skip)
    multi_tracker.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multi-stream object tracking')
    parser.add_argument('--show_video', action='store_true', help='Display video output')
    parser.add_argument('--enable_frame_skip', action='store_true', help='Enable frame skipping')
    args = parser.parse_args()

    run_multi_stream_object_tracking(args.show_video, args.enable_frame_skip)