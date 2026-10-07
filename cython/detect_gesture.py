import sys
import asyncio
import time
import subprocess
import shutil
import argparse
import numpy as np
from PIL import Image
import cv2

# Hardware Libraries
import RPi.GPIO as GPIO
from pykondo import Kondo

# Coral / TFLite Libraries
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Import our modular head tracking
import cython.modules.head_tracking as head_tracking

# --- OVERRIDE TRACKING LIMITS FOR THIS SCRIPT ---
# We limit the head movement to 30 degrees for gesture mode
head_tracking.PAN_LIMIT_RIGHT = -30
head_tracking.PAN_LIMIT_LEFT = 30
# ------------------------------------------------

# --- Configuration & Constants ---

# Model Path
MODEL_PATH = "/Kondo-KHR-3HV/Coral USB Accelerator/examples/models/movenet_single_pose_lightning_ptq_edgetpu.tflite"

# Gesture Thresholds
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to trust a keypoint
GESTURE_HOLD_TIME = 1.0     # Seconds a gesture must be held to trigger

# GPIO
TRIG_PIN = 24  
ECHO_PIN = 23  
LED_PIN = 17   
TILT_SERVO_PIN = 18 

# Motion IDs
MOTION_HOME_STAND_ID = 2  
MOTION_WAVE_ID = 3
MOTION_CHEER_ID = 8       # Example: Both hands up
MOTION_BOW_ID = 4         # Example: Bow
MOTION_SIT_DOWN_ID = 21   
MOTION_STAND_UP_ID = 22   

# Keypoint Indices (MoveNet)
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10

# --- Global State ---
rotate_head_event = asyncio.Event() 
motion_lock = asyncio.Lock()      

state = {
    "last_face_seen_time": time.time(), 
    "last_motion_time": 0,       
    "is_sitting": False,         
    "gesture_start_time": None,
    "current_gesture": None,
    "distance_cm": 999.0,
    "current_pan": head_tracking.PAN_CENTER,
    "current_tilt": head_tracking.TILT_CENTER
}

# --- Helper Classes ---

class PseudoFace:
    """
    Wraps a keypoint (like the Nose) into an object 
    compatible with head_tracking.track_face()
    """
    def __init__(self, x, y, size=0.1):
        self.x = x
        self.y = y
        # Create a fake bounding box centered on the point
        self.bbox = type('obj', (object,), {
            'xmin': x - size,
            'xmax': x + size,
            'ymin': y - size,
            'ymax': y + size
        })

# --- Utils ---

def ensure_pigpiod(start_timeout=2.0):
    try:
        subprocess.check_call(["pgrep", "pigpiod"], stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        pass
    pigpiod_path = shutil.which("pigpiod")
    if pigpiod_path:
        subprocess.Popen(["sudo", pigpiod_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(start_timeout)
        return True
    return False

def setup_gpio():
    GPIO.setwarnings(False) 
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.HIGH) 

def cleanup_gpio(ms=None):
    if ms: ms.stop()
    GPIO.cleanup()

# --- Analysis Functions ---

def analyze_gesture(keypoints):
    """
    Analyzes keypoints to determine simple gestures.
    Returns: String (Gesture Name) or None
    """
    # Helper to get point: [y, x, score]
    def get_pt(idx): return keypoints[idx]
    
    nose = get_pt(KEYPOINT_NOSE)
    l_wrist = get_pt(KEYPOINT_LEFT_WRIST)
    r_wrist = get_pt(KEYPOINT_RIGHT_WRIST)
    l_shoulder = get_pt(KEYPOINT_LEFT_SHOULDER)
    r_shoulder = get_pt(KEYPOINT_RIGHT_SHOULDER)

    # Check Confidence
    if nose[2] < CONFIDENCE_THRESHOLD: return None

    # Gesture 1: BOTH HANDS UP (Wrists above Nose)
    # Note: In image coordinates, Lower Y value = Higher physical position
    if (l_wrist[2] > CONFIDENCE_THRESHOLD and r_wrist[2] > CONFIDENCE_THRESHOLD):
        if l_wrist[0] < nose[0] and r_wrist[0] < nose[0]:
            return "BOTH_HANDS_UP"

    # Gesture 2: ONE HAND WAVE (Wrist above Shoulder)
    if l_wrist[2] > CONFIDENCE_THRESHOLD and l_wrist[0] < l_shoulder[0]:
        return "LEFT_HAND_UP"
    
    if r_wrist[2] > CONFIDENCE_THRESHOLD and r_wrist[0] < r_shoulder[0]:
        return "RIGHT_HAND_UP"

    return None

async def run_motion(kondo, motion_id, sleep_time=10.0):
    async with motion_lock:
        print(f"[[ GESTURE DETECTED -> MOTION: {motion_id} ]]")
        kondo.call_motion(int(motion_id))
        await asyncio.sleep(sleep_time)
        state["current_pan"] = 0

# --- Async Tasks ---

async def distance_task():
    loop = asyncio.get_running_loop()
    print("Starting distance sensor...")
    while True:
        # Simple blocking measurement (wrapped)
        def measure():
            GPIO.output(TRIG_PIN, GPIO.HIGH)
            time.sleep(0.000010)
            GPIO.output(TRIG_PIN, GPIO.LOW)
            if GPIO.wait_for_edge(ECHO_PIN, GPIO.RISING, timeout=50) is None: return 999.0
            s = time.monotonic()
            if GPIO.wait_for_edge(ECHO_PIN, GPIO.FALLING, timeout=50) is None: return 999.0
            e = time.monotonic()
            return ((e - s) * 34300) / 2
            
        state["distance_cm"] = await loop.run_in_executor(None, measure)
        await asyncio.sleep(0.1)

async def idle_manager_task(kondo, ms):
    print("Starting Idle Manager...")
    # This will pick up the modified ht.PAN_LIMIT_LEFT/RIGHT because HeadBehavior reads the module globals
    head = head_tracking.HeadBehavior(kondo, ms, rotate_head_event, state)
    
    while True:
        await rotate_head_event.wait()
        idle_time = time.time() - state["last_face_seen_time"]

        # Sit Down if idle too long
        if idle_time > 20.0 and not state["is_sitting"]:
            print(f"Idle for {idle_time:.1f}s. Sitting Down...")
            ms.center()
            head.move_pan(0)
            await asyncio.sleep(0.5)
            await run_motion(kondo, MOTION_SIT_DOWN_ID, sleep_time=8.0)
            state["is_sitting"] = True

        if not await head.check_environment(): continue
        await asyncio.sleep(5.0 if state["is_sitting"] else 1.0)

async def gesture_loop(kondo, ms):
    print("Loading MoveNet Model...")
    try:
        interpreter = make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Camera warming up...")
    cap = cv2.VideoCapture(0)
    # Optimize for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Gesture System Ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue
            
        # 1. Preprocess for MoveNet (192x192)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize(common.input_size(interpreter), Image.ANTIALIAS)
        common.set_input(interpreter, resized_img)

        # 2. Inference
        interpreter.invoke()
        
        # 3. Extract Keypoints
        pose = common.output_tensor(interpreter, 0).copy().reshape(1, 17, 3)[0]
        
        # 4. Logic
        nose = pose[KEYPOINT_NOSE]
        
        # Check if we see a person (Nose confidence)
        if nose[2] > CONFIDENCE_THRESHOLD:
            # Update timers
            state["last_face_seen_time"] = time.time()
            
            # WAKE UP if sitting
            if state["is_sitting"] and not motion_lock.locked():
                 print("!!! PERSON SEEN - STANDING UP !!!")
                 rotate_head_event.clear()
                 kondo.set_angle(head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL)
                 ms.center()
                 await run_motion(kondo, MOTION_STAND_UP_ID, sleep_time=8.0)
                 state["is_sitting"] = False
                 continue

            # Stop scanning
            if rotate_head_event.is_set():
                print("!!! PERSON TRACKING !!!")
                rotate_head_event.clear()

            # --- A. GESTURE RECOGNITION ---
            detected_gesture = analyze_gesture(pose)
            
            if detected_gesture:
                if state["current_gesture"] == detected_gesture:
                    # Check hold time
                    if time.time() - state["gesture_start_time"] > GESTURE_HOLD_TIME:
                        # TRIGGER MOTION
                        if not motion_lock.locked():
                            if detected_gesture == "BOTH_HANDS_UP":
                                await run_motion(kondo, MOTION_CHEER_ID)
                            elif "HAND_UP" in detected_gesture:
                                await run_motion(kondo, MOTION_WAVE_ID)
                            
                            # Reset gesture state
                            state["current_gesture"] = None
                else:
                    # New gesture started
                    state["current_gesture"] = detected_gesture
                    state["gesture_start_time"] = time.time()
                    print(f"Gesture Detected: {detected_gesture} (Holding...)")
            else:
                state["current_gesture"] = None

            # --- B. HEAD TRACKING ---
            nose_y, nose_x, _ = nose
            
            # Convert to PseudoFace (in Pixels)
            h, w = frame.shape[:2]
            pixel_x = nose_x * w
            pixel_y = nose_y * h
            
            pseudo_face = PseudoFace(pixel_x, pixel_y)
            
            if not motion_lock.locked():
                head_tracking.track_face(kondo, ms, pseudo_face, w, h, state)

        else:
            # No person seen
            if not rotate_head_event.is_set() and not motion_lock.locked():
                print("Lost target. Resuming scan.")
                rotate_head_event.set()

        # Yield to allow other tasks to run
        await asyncio.sleep(0.001)

async def main():
    print("--- Gesture Control Init (Reduced Limits) ---")
    if not ensure_pigpiod(): sys.exit(1)
    setup_gpio()
    ms = head_tracking.MicroServo(TILT_SERVO_PIN, state)
    
    kondo = Kondo()
    if kondo.init() < 0: print("Warning: Kondo Init failed.")

    # Start Standing
    await run_motion(kondo, MOTION_HOME_STAND_ID, sleep_time=10.0)

    state["current_pan"] = 0
    state["current_tilt"] = head_tracking.TILT_CENTER
    rotate_head_event.set() 

    await asyncio.gather(
        distance_task(),
        idle_manager_task(kondo, ms),
        gesture_loop(kondo, ms)
    )

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cleanup_gpio()
        try: head_tracking.MicroServo(TILT_SERVO_PIN, state).stop()
        except: pass
        loop.close()