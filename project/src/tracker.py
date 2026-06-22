import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv


class KalmanFilter1D:
    """
    1D Kalman filter: state = [position, velocity], measurement = position.
    Smooths noisy YOLO detections for more stable trajectory.
    """
    def __init__(self, dt=1.0, process_noise=0.01, measurement_noise=5.0):
        self.dt = dt
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 100.0
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * process_noise
        self.R = np.array([[measurement_noise]])
        self.I = np.eye(2)

    def update(self, z):
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (self.I - K @ self.H) @ P_pred
        return self.x.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()


class PendulumTracker:
    def __init__(self, model_path, conf=0.1, iou=0.3, track_history=80, box_thickness=5, target_class=0):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=box_thickness, color=sv.Color(0, 255, 0))
        self.label_annotator = sv.LabelAnnotator(text_scale=0.7, text_thickness=2)
        self.trace_annotator = sv.TraceAnnotator(thickness=3, trace_length=track_history)
        self.conf = conf
        self.iou = iou
        self.target_class = target_class
        self.kf_x = KalmanFilter1D()
        self.kf_y = KalmanFilter1D()
        self.trajectory_raw = []
        self.trajectory_filtered = []
        self.fps = 30.0

    def calibrate_pixel_per_cm(self, frame):
        win = "calibration"
        clone = frame.copy()
        display = [frame.copy()]
        points = []
        drawing = [False]

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing[0] = True
                points.clear()
                points.append((x, y))
                display[0] = clone.copy()
            elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
                tmp = clone.copy()
                cv2.line(tmp, points[0], (x, y), (0, 255, 0), 2)
                cv2.imshow(win, tmp)
            elif event == cv2.EVENT_LBUTTONUP and drawing[0]:
                drawing[0] = False
                if len(points) == 1:
                    points.append((x, y))
                else:
                    points[1] = (x, y)
                display[0] = clone.copy()
                cv2.line(display[0], points[0], points[1], (0, 215, 255), 3)
                cv2.putText(display[0], "Press ENTER to confirm, ESC to skip",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
                cv2.imshow(win, display[0])

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse_cb)
        cv2.imshow(win, clone)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(win)
                return None
            if key == 13 and len(points) == 2:
                break
        cv2.destroyWindow(win)

        p1, p2 = points[0], points[1]
        pixel_len = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        print(f"   drawn line pixel length = {pixel_len:.1f} px")
        while True:
            try:
                real_cm = float(input("   Enter real length (cm): "))
                if real_cm <= 0:
                    raise ValueError
                break
            except ValueError:
                print("   enter positive number")
        ppcm = pixel_len / real_cm
        print(f"   PIXEL_PER_CM = {ppcm:.4f}")
        return ppcm

    def process_video(self, source, fps_fallback=30.0, save_video=False, headless=False):
        cap = cv2.VideoCapture(source)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        source_id = int(source) if str(source) == "0" else source

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("output/output_tracked.mp4", fourcc, self.fps, (w, h))

        self.trajectory_raw = []
        self.trajectory_filtered = []
        self.kf_x = KalmanFilter1D()
        self.kf_y = KalmanFilter1D()
        last_cx, last_cy = None, None

        print("Tracking started (press Q to stop)")

        for result in self.model.track(source=source_id, stream=True, persist=True,
                                        conf=self.conf, iou=self.iou, classes=[self.target_class]):
            frame = result.orig_img.copy()
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                best_idx = int(detections.confidence.argmax())
                detections = detections[[best_idx]]
                x1, y1, x2, y2 = detections.xyxy[0]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                last_cx, last_cy = cx, cy
            else:
                if last_cx is None:
                    frame_idx = len(self.trajectory_raw)
                    if not headless:
                        info = f"Frame:{frame_idx}  No detection"
                        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow("Pendulum Tracking", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                cx, cy = last_cx, last_cy

            self.trajectory_raw.append((cx, cy))
            self.kf_x.update(np.array([cx]))
            self.kf_y.update(np.array([cy]))
            fx = float(self.kf_x.x[0])
            fy = float(self.kf_y.x[0])
            self.trajectory_filtered.append((fx, fy))

            annotated_dets = self.tracker.update_with_detections(detections)
            frame = self.trace_annotator.annotate(frame, annotated_dets)
            draw_dets = detections if len(detections) > 0 else annotated_dets
            frame = self.box_annotator.annotate(frame, draw_dets)
            if len(draw_dets) > 0:
                labels = [f"ball {d:.2f}" for d in draw_dets.confidence]
                frame = self.label_annotator.annotate(frame, draw_dets, labels)

            n = len(self.trajectory_raw)
            if not headless:
                cv2.putText(frame, f"Frame:{n}  cx:{cx:.0f}  FPS:{self.fps:.1f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Pendulum Tracking", frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if not headless:
            cv2.destroyAllWindows()
        if writer:
            writer.release()

        n_raw = len(self.trajectory_raw)
        n_filt = len(self.trajectory_filtered)
        print(f"Tracking done: {n_raw} raw points, {n_filt} filtered points")

        traj_raw = np.array(self.trajectory_raw)
        traj_filt = np.array(self.trajectory_filtered)
        np.save("output/trajectory_raw.npy", traj_raw)
        np.save("output/trajectory.npy", traj_filt)
        np.save("output/track_meta.npy", {"fps": self.fps, "total_frames": n_raw}, allow_pickle=True)

        return traj_raw, traj_filt, self.fps
