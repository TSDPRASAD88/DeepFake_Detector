import cv2
from mtcnn import MTCNN
import sys

# 1. Initialize the detector
try:
    detector = MTCNN()
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    sys.exit(1)

# 2. Open your MacBook's camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera. Check System Settings > Privacy & Security > Camera.")
    sys.exit(1)

print("--- Face Detection Started ---")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    # Check if the frame is valid and not empty
    if not ret or frame is None or frame.shape[0] < 20 or frame.shape[1] < 20:
        continue

    try:
        # 3. Detect faces
        # We only pass frames that have actual pixel data to avoid Conv2D errors
        faces = detector.detect_faces(frame)

        # 4. Draw rectangles ONLY if faces were actually found
        if faces:
            for face in faces:
                x, y, w, h = face['box']
                # Safety check: MTCNN sometimes returns negative coordinates
                x, y = max(0, x), max(0, y)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show confidence level
                conf = face['confidence']
                cv2.putText(frame, f'Face: {conf:.2f}', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Show the window
        cv2.imshow('Face Detection Test', frame)

    except Exception as e:
        # This silently handles the "Empty Output" error if the camera flickers
        if "Conv2D" not in str(e):
            print(f"Detection error: {e}")
        continue

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("--- Session Ended ---")