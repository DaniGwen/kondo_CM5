import cv2
import vision
import models

def load_labels(file_path):
    labels = {}
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if not line: continue
                parts = line.split(" ", 1)
                if parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].lower()
                else:
                    labels[i] = line.lower()
    except Exception as e:
        print(f"Error loading labels: {e}")
    return labels

def main():
    print("Loading AI Model...")
    detector = vision.VisionBridge(models.OBJECT_DETECTION_MODEL)
    labels = load_labels(models.OBJECT_DETECTION_LABELS)
    
    print("Activating Camera...")
    frames = detector.get_frames(size=(640, 480))
    
    print("Opening VNC Window. Press 'q' in the window to quit.")
    
    try:
        for frame in frames:
            if frame is None:
                continue
                
            # Detect objects with a 15% confidence threshold
            objects = detector.get_objects(frame, threshold=0.15)
            
            # Draw the bounding boxes and labels
            vision.draw_objects(frame, objects, labels=labels)
            
            # Display the live video feed
            cv2.imshow("Robot Vision Debug", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if hasattr(frames, "close"):
            frames.close()
        print("Camera released.")

if __name__ == "__main__":
    main()