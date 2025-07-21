import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import threading
import random
from datetime import datetime
import os
import time
import pygame  # for alert sounds

class ObjectMonitoringApp:
    def __init__(self):
        self.models: dict[str, YOLO] = {}
        self.current_model: YOLO | None = None
        self.cap: cv2.VideoCapture | None = None
        self.class_colors: dict[str, tuple[int, int, int]] = {}
        self.restricted_area: tuple[tuple[int,int], tuple[int,int]] | None = None
        self.csv_file = "data/detection_log.csv"
        self.object_entry_times: dict[str, float] = {}
        self.violation_screenshots_dir = "violation_screenshots"

        # alert system state
        self.alert_active = False
        self.alert_thread: threading.Thread | None = None

        # init CSV if missing
        if not os.path.exists(self.csv_file):
            pd.DataFrame(
                columns=["Timestamp", "Class", "Confidence", "Restricted Area Violation"]
            ).to_csv(self.csv_file, index=False)

        # Create violation screenshots directory if it doesn't exist
        if not os.path.exists(self.violation_screenshots_dir):
            os.makedirs(self.violation_screenshots_dir)

        # init pygame mixer
        pygame.mixer.init()

    def load_models(self, model_paths: dict[str, str]):
        """Load multiple YOLO models."""
        for name, path in model_paths.items():
            self.models[name] = YOLO(path)
        self.current_model = self.models.get("Intrusion")

    def generate_class_colors(self, model: YOLO) -> dict[str, tuple[int,int,int]]:
        """Assign a random color to each class."""
        return {
            model.names[cid]: tuple(random.randint(0, 255) for _ in range(3))
            for cid in model.names
        }

    def start_webcam(self) -> bool:
        """Open the default webcam at higher resolution."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            st.error("Unable to access webcam.")
            return False
        # increase resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True

    def stop_webcam(self):
        """Release the webcam and stop any alerts."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
        self.stop_alert()

    def play_alert_sound(self, sound_path: str):
        """Loop an alert sound while alert_active=True."""
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)
        while self.alert_active:
            time.sleep(0.1)
        pygame.mixer.music.stop()

    def start_alert(self, sound_path: str):
        """Spawn the alert thread if not already running."""
        if not self.alert_active:
            self.alert_active = True
            self.alert_thread = threading.Thread(
                target=self.play_alert_sound, args=(sound_path,), daemon=True
            )
            self.alert_thread.start()

    def stop_alert(self):
        """Signal the alert thread to stop."""
        self.alert_active = False

    def draw_restricted_area(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a semi-transparent, labeled rectangle on the frame."""
        if self.restricted_area:
            (x1, y1), (x2, y2) = self.restricted_area
            overlay = frame.copy()
            # filled rectangle for overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # label
            cv2.putText(
                frame,
                "Restricted Area",
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return frame

    def is_inside_restricted(self, box: list[int]) -> bool:
        """Check if the object's center lies within the ROI."""
        if self.restricted_area:
            (x1, y1), (x2, y2) = self.restricted_area
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            return x1 <= cx <= x2 and y1 <= cy <= y2
        return False

    def save_detection_data(self, cls: str, conf: float):
        """Append a violation record to the CSV."""
        data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Class": cls,
            "Confidence": f"{conf:.2f}",
            "Restricted Area Violation": "Yes",
        }
        pd.DataFrame([data]).to_csv(self.csv_file, mode="a", header=False, index=False)

    def save_violation_screenshot(self, frame: np.ndarray, violation_type: str):
        """Save a screenshot of the violation with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.violation_screenshots_dir}/{violation_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def update_frame(
        self,
        model: YOLO,
        conf_thresh: float,
        detect_classes: list[str],
        alert_classes: list[str],
        helmet_required: bool = False,
        no_helmet_alert: bool = False,
    ) -> tuple[np.ndarray | None, list[str]]:
        """Run detection on one frame, annotate, log, and trigger alerts."""
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        results = model(frame, conf=conf_thresh, iou=0.3)
        annotated = frame.copy()
        detected: list[str] = []
        violation = False
        no_helmet_violation = False

        # Track people and their helmet status
        people_in_roi = []
        helmets_in_roi = []

        for box in results[0].boxes:
            cid = int(box.cls)
            name = model.names[cid]
            if name not in detect_classes:
                continue

            detected.append(name)
            color = self.class_colors.get(name, (0, 255, 0))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Check if object is in restricted area
            if self.is_inside_restricted([x1, y1, x2, y2]):
                if name == "person":
                    people_in_roi.append((x1, y1, x2, y2))
                elif name == "helmet":
                    helmets_in_roi.append((x1, y1, x2, y2))
                
                violation = True
                now = time.time()
                last = self.object_entry_times.get(name, 0)
                if now - last > 2:  # throttle logging
                    self.save_detection_data(name, conf)
                    self.object_entry_times[name] = now

        # Check for helmet violations
        if helmet_required and people_in_roi:
            for person in people_in_roi:
                px1, py1, px2, py2 = person
                person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                
                # Check if person has a helmet
                has_helmet = False
                for helmet in helmets_in_roi:
                    hx1, hy1, hx2, hy2 = helmet
                    helmet_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
                    
                    # Check if helmet is near person's head (top 1/3 of person)
                    if (abs(person_center[0] - helmet_center[0]) < (px2 - px1) and
                        abs(hy1 - py1) < (py2 - py1) / 3):
                        has_helmet = True
                        break
                
                if not has_helmet:
                    no_helmet_violation = True
                    # Draw warning on person
                    cv2.putText(
                        annotated,
                        "NO HELMET!",
                        (px1, py1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

        # alert logic
        # Case 1: Person/Head detected outside red zone - play alert.mp3
        if ("person" in detected or "head" in detected) and not violation:
            self.start_alert("alert.mp3")
            self.save_violation_screenshot(annotated, "Person_Onsite_Without_Helmet")
        # Case 2: Helmet inside red zone - play alert2.mp3
        elif violation and "helmet" in detected:
            self.start_alert("alert2.mp3")
            self.save_violation_screenshot(annotated, "Person_Entering_Dangerous_Zone_With_Helmet")
        # Case 3: Person/Head inside red zone - play alert3.mp3
        elif violation and ("person" in detected or "head" in detected):
            self.start_alert("alert3.mp3")
            self.save_violation_screenshot(annotated, "Person_Entering_Dangerous_Zone_Without_Helmet")
        else:
            self.stop_alert()

        # draw ROI overlay
        annotated = self.draw_restricted_area(annotated)
        return annotated, detected

    def run(self):
        st.set_page_config(
            page_title="Real-Time Intrusion & ROI Monitoring", layout="wide"
        )
        st.markdown(
            "<h2 style='text-align:center;'>🔍 Intrusion Detection & Restricted Area Monitor</h2>",
            unsafe_allow_html=True,
        )
        st.sidebar.title("🔧 Settings")

        # model selector
        model_paths = {
            "Intrusion": "model/yolov8n.pt",
            "Helmet": "model/yolov8n-helmet.pt"  # You'll need to download this model
        }
        sel = st.sidebar.selectbox("Select Model", list(model_paths.keys()))
        if self.current_model != self.models.get(sel):
            self.current_model = self.models[sel]
            self.class_colors = self.generate_class_colors(self.current_model)

        # Add helmet-specific settings
        if sel == "Helmet":
            st.sidebar.markdown("### Helmet Detection Settings")
            helmet_required = st.sidebar.checkbox("Require Helmet in Restricted Area", value=True)
            no_helmet_alert = st.sidebar.checkbox("Alert on No Helmet", value=True)
        else:
            helmet_required = False
            no_helmet_alert = False

        # ROI sliders
        st.sidebar.markdown("### Define Restricted Area")
        width, height = 960, 720
        x1 = st.sidebar.slider("Top-Left X", 0, width, 100, key="roi_x1")
        y1 = st.sidebar.slider("Top-Left Y", 0, height, 100, key="roi_y1")
        x2 = st.sidebar.slider("Bottom-Right X", 0, width, width // 2, key="roi_x2")
        y2 = st.sidebar.slider("Bottom-Right Y", 0, height, height // 2, key="roi_y2")
        self.restricted_area = ((x1, y1), (x2, y2))

        # detection settings
        conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        all_classes = list(self.current_model.names.values())
        detect_classes = st.sidebar.multiselect(
            "Detect These Objects", all_classes, default=[]
        )
        alert_classes = st.sidebar.multiselect(
            "Alert On These Objects", all_classes, default=[]
        )

        # start/stop controls
        if st.sidebar.button("▶️ Start Webcam"):
            if self.start_webcam():
                st.success("Webcam started.")
        if st.sidebar.button("⏹️ Stop Webcam"):
            self.stop_webcam()
            st.success("Webcam stopped.")

        # live feed
        if self.cap:
            placeholder = st.empty()
            while self.cap.isOpened():
                frame, _ = self.update_frame(
                    self.current_model,
                    conf_thresh,
                    detect_classes,
                    alert_classes,
                    helmet_required,
                    no_helmet_alert
                )
                if frame is not None:
                    placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True
                    )

if __name__ == "__main__":
    app = ObjectMonitoringApp()
    app.load_models({
        "Intrusion": "model/yolov8n.pt",
        "Helmet": "model/yolov8n-helmet.pt"  # You'll need to download this model
    })
    app.run()
