# vision.py
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate


class ObjectDetection:
    """Mimics the AIY Maker Kit 'Object' structure."""
    def __init__(self, bbox, label_id, score):
        # Expose both .bbox and .bounding_box for compatibility
        self.bbox = bbox  # (x, y, w, h)
        self.bounding_box = bbox
        self.id = label_id
        self.label = label_id
        self.score = score


def draw_objects(frame, objects, labels=None):
    """Draws bounding boxes and class labels directly onto the OpenCV frame."""
    if frame is None:
        return

    # Only try to draw boxes if objects actually exist
    if objects:
        for obj in objects:
            bbox = getattr(obj, "bbox", getattr(obj, "bounding_box", None))
            if not bbox:
                continue

            x, y, w, h = bbox
            label_id = getattr(obj, "id", None)
            score = getattr(obj, "score", 0.0)

            label_name = labels.get(label_id, str(label_id)) if labels else str(label_id)
            display_text = f"{label_name}: {score:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
            
            cv2.rectangle(
                frame, 
                (x, max(0, y - th - baseline - 4)), 
                (x + tw, max(0, y)), 
                (0, 255, 0), 
                -1
            )
            cv2.putText(
                frame, 
                display_text, 
                (x, max(0, y - 4)), 
                font, 
                font_scale, 
                (0, 0, 0), 
                thickness, 
                cv2.LINE_AA
            )

    cv2.imshow("Robot Vision Feed", frame)
    cv2.waitKey(1)


class VisionBridge:
    def __init__(self, model_path):
        # Load LiteRT engine with Edge TPU delegate
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        # Determine expected model input dimensions
        self.input_height = self.input_details["shape"][1]
        self.input_width = self.input_details["shape"][2]

    def get_frames(self, size=(640, 480), camera_index=0):
        """Camera generator supporting the size parameter."""
        cap = cv2.VideoCapture(camera_index)
        
        if isinstance(size, (tuple, list)) and len(size) == 2:
            width, height = size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def get_objects(self, frame, threshold=0.2):
        """Runs TPU inference and translates output tensors to object list."""
        if frame is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized_frame, axis=0)

        # Accommodate models expecting INT8/UINT8 vs FLOAT32
        if self.input_details["dtype"] == np.uint8:
            input_tensor = input_data.astype(np.uint8)
        else:
            input_tensor = input_data.astype(np.float32)

        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        objects = []
        frame_h, frame_w = frame.shape[:2]

        for i in range(count):
            score = float(scores[i])
            if score >= threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Convert normalized coordinates [0, 1] to pixel dimensions
                x = int(max(0, xmin * frame_w))
                y = int(max(0, ymin * frame_h))
                w = int(min(frame_w - x, (xmax - xmin) * frame_w))
                h = int(min(frame_h - y, (ymax - ymin) * frame_h))

                objects.append(ObjectDetection(
                    bbox=(x, y, w, h),
                    label_id=int(classes[i]),
                    score=score
                ))

        return objects