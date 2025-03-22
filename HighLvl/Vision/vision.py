import cv2
from ultralytics import YOLO
import numpy as np
import threading
import queue
import torch

# Verify GPU availability for PyTorch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Install CUDA Toolkit and GPU-enabled PyTorch.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Queues for inter-thread communication
frame_queue = queue.Queue(maxsize=10)  # Webcam frames to mapping
yolo_queue = queue.Queue(maxsize=10)  # Frames to YOLO
result_queue = queue.Queue(maxsize=10) # Processed results

# Load YOLO model on GPU
yolo_model = YOLO("yolov8n.pt").to("cuda")  # Nano model, explicitly on GPU

# Webcam Thread: Captures frames (CPU via OpenCV)
def webcam_thread():
    cap = cv2.VideoCapture(0)  # Default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam Thread: Starting capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam Thread: Failed to capture frame.")
            break
        try:
            frame_queue.put_nowait(frame.copy())  # For mapping
            yolo_queue.put_nowait(frame.copy())  # For YOLO
        except queue.Full:
            pass  # Skip if queues are full
    cap.release()

# YOLO Thread: GPU-accelerated object detection via PyTorch
def yolo_thread():
    print("YOLO Thread: Starting GPU-accelerated object detection...")
    while True:
        try:
            frame = yolo_queue.get(timeout=1.0)
            # Run YOLO inference on GPU
            results = yolo_model(frame, stream=True, device="cuda")
            for r in results:
                annotated_frame = r.plot()  # Bounding boxes drawn on CPU (post-GPU inference)
                result_queue.put(("yolo", annotated_frame))
            yolo_queue.task_done()
        except queue.Empty:
            continue

# Mapping Thread: Feature-based visual mapping (CPU via OpenCV)
def mapping_thread():
    orb = cv2.ORB_create(nfeatures=1000)  # ORB detector on CPU
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # CPU matcher
    prev_frame = None
    prev_kp = None
    prev_des = None
    map_image = None  # Accumulated map visualization

    print("Mapping Thread: Starting CPU-based visual mapping...")
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB features on CPU
            kp, des = orb.detectAndCompute(gray, None)
            
            if prev_frame is not None and prev_des is not None:
                # Match features between frames
                matches = bf.match(prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Extract matched points
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Estimate homography
                H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                
                # Warp previous frame to align with current
                if map_image is None:
                    map_image = prev_frame.copy()
                else:
                    h, w = frame.shape[:2]
                    warped = cv2.warpPerspective(map_image, H, (w * 2, h * 2))  # CPU operation
                    map_image = cv2.addWeighted(warped, 0.5, frame, 0.5, 0.0)
                
                # Draw keypoints for visualization
                map_display = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                result_queue.put(("map", map_display))

            # Update previous frame data
            prev_frame = frame.copy()
            prev_kp = kp
            prev_des = des
            frame_queue.task_done()
        except queue.Empty:
            continue

# Main display loop
def main():
    # Start threads
    threads = [
        threading.Thread(target=webcam_thread, daemon=True),
        threading.Thread(target=yolo_thread, daemon=True),
        threading.Thread(target=mapping_thread, daemon=True)
    ]
    for t in threads:
        t.start()

    print("Main: System running. Press 'q' to quit.")
    while True:
        try:
            result_type, frame = result_queue.get(timeout=1.0)
            if result_type == "yolo":
                cv2.imshow("YOLO Detection", frame)
            elif result_type == "map":
                cv2.imshow("Visual Map", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            result_queue.task_done()
        except queue.Empty:
            continue

    # Cleanup
    cv2.destroyAllWindows()
    print("Main: Shutting down...")

if __name__ == "__main__":
    main()