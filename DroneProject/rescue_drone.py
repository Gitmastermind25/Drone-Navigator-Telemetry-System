import cv2
import numpy as np
import time
from ultralytics import YOLO


# Title: Real-time Drone Telemetry & Target Locking

def process_drone_feed(source="vid.mp4"):
    # 1. Load the lightest possible model (Nano)
    model = YOLO('yolov8n.pt') 
    
    # 2. Open the video source
    cap = cv2.VideoCapture(source)
    
    # 3. Setup FPS and frame skipping variables
    prev_time = 0
    frame_count = 0
    CONF_THRESHOLD = 0.25  # Sensitivity for detecting people/objects

    print("System Booting... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # RESIZE EVERYTHING TO 640x640 FOR CONSISTENT HUD MATH
        frame = cv2.resize(frame, (640, 640))

        # --- OPTIMIZATION: ONLY RUN AI ON EVERY 2nd FRAME ---

        if frame_count % 2 == 0:
            # imgsz=160 is the biggest secret for CPU speed boost!
            results = model.predict(frame, imgsz=160, conf=CONF_THRESHOLD, verbose=False)

            for r in results:
                for box in r.boxes:
                    # Get Bounding Box Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Target Lock Logic: Middle 20% of the screen (256px to 384px)
                    center_x = (x1 + x2) / 2
                    is_locked = 256 <= center_x <= 384
                    
                    color = (0, 0, 255) if is_locked else (0, 255, 0) # Red if locked, Green if not
                    label = "TARGET LOCKED" if is_locked else "SCANNING"

                    # Draw the HUD Elements
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. Drone Telemetry HUD (FPS Counter)
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Display Data on Screen
        cv2.putText(frame, f"TELEMETRY FPS: {int(fps_val)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "ISA DRONE SYSTEM V1.4", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 5. Show Window
        cv2.imshow("ISA Drone Telemetry", frame)
        
        # waitKey(1) is critical for max speed
        if cv2.waitKey(33) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Mission Ended.")

if __name__ == "__main__":
    process_drone_feed()