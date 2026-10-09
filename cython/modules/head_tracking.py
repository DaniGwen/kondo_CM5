import time
import asyncio
from gpiozero import Servo

# --- Constants ---

# Kondo Servo (Pan)
SERVO_ID_HEAD_PAN = 0

# STANDARD LIMITS
PAN_LIMIT_RIGHT = -41
PAN_LIMIT_LEFT = 45

PAN_CENTER = 0
PAN_SPEED_NORMAL = 0.3
PAN_SPEED_FAST = 0.1

# Micro Servo (Tilt) Settings
# NOTE: Lower value = Physically Higher/Up
TILT_CENTER = 740  # Standing Center
TILT_UP = 580
TILT_DOWN = 880

# --- SITTING TILT SETTINGS ---
TILT_CENTER_SITTING = 600  
TILT_UP_SITTING = 500  

# Servo Sweep Settings
SWEEP_STEP = 40
SWEEP_DELAY = 0.01

# Scan Settings
LOOK_PAUSE = 1.0

# Tracking Tuning
DEBUG_TRACKING = True
TRACKING_DIRECTION = -1.0

# === TUNING UPDATES FOR STABILITY ===
TRACKING_GAIN_PAN = 15.0      # Reduced from 55.0 to stop wind-up
TRACKING_GAIN_TILT = 70.0
TRACKING_DEADZONE = 0.08      # Increased from 0.04 to create a solid center buffer
MAX_PAN_STEP = 3.0            # Reduced from 9.0 to limit max rotation speed per frame
# ====================================


# --- Classes ---

class MicroServo:
    """Controls the Tilt Servo using modern gpiozero for CM5 compatibility."""

    def __init__(self, gpio, state_ref):
        self.gpio = gpio
        self.state = state_ref
        self.servo = Servo(gpio, min_pulse_width=0.0005, max_pulse_width=0.0025)
        self.set_us(TILT_CENTER)
        
        # Give it a moment to reach center on boot, then instantly detach to prevent jitter
        time.sleep(0.2)
        self.servo.detach()

    def set_us(self, us):
        self.state["current_tilt"] = us
        us = max(500, min(2500, int(us)))
        mapped_value = (us - 1500) / 1000.0
        self.servo.value = mapped_value

    def sweep_to(self, target_us, step=SWEEP_STEP, delay=SWEEP_DELAY):
        current = self.state.get("current_tilt", TILT_CENTER)

        if current < target_us:
            rng = range(int(current), int(target_us) + 1, abs(int(step)))
        else:
            rng = range(int(current), int(target_us) - 1, -abs(int(step)))

        for us in rng:
            self.set_us(us)
            time.sleep(delay)
            
        self.set_us(target_us)
        time.sleep(0.1) # Allow physical motor to catch up to final signal
        self.servo.detach() # Cut the PWM signal to freeze position and stop twitching

    def center(self):
        if self.state.get("is_sitting", False):
            self.sweep_to(TILT_CENTER_SITTING)
        else:
            self.sweep_to(TILT_CENTER)

    def stop(self):
        self.servo.detach()


class HeadBehavior:
    """Manages head scanning motions (looking around)."""

    def __init__(self, kondo, ms, rotate_event, state_ref):
        self.kondo = kondo
        self.ms = ms
        self.event = rotate_event
        self.state = state_ref
        self.speed = PAN_SPEED_NORMAL
        self.pause = LOOK_PAUSE

    def move_pan(self, angle, speed=None):
        if speed is None:
            speed = self.speed
        self.state["current_pan"] = angle
        self.kondo.set_angle(SERVO_ID_HEAD_PAN, angle, speed)

    async def _wait(self, duration):
        end = time.time() + duration
        while time.time() < end:
            if not self.event.is_set():
                return False
            await asyncio.sleep(0.1)
        return True

    def get_tilt_up(self):
        return TILT_UP_SITTING if self.state["is_sitting"] else TILT_UP

    def get_tilt_center(self):
        return TILT_CENTER_SITTING if self.state["is_sitting"] else TILT_CENTER

    async def look_left_up(self):
        if not self.event.is_set():
            return False
        self.move_pan(PAN_LIMIT_LEFT)
        self.ms.sweep_to(self.get_tilt_up())
        return await self._wait(self.pause)

    async def look_left_down(self):
        if not self.event.is_set():
            return False
        self.move_pan(PAN_LIMIT_LEFT)
        self.ms.sweep_to(TILT_DOWN)
        return await self._wait(self.pause)

    async def look_right_up(self):
        if not self.event.is_set():
            return False
        self.move_pan(PAN_LIMIT_RIGHT)
        self.ms.sweep_to(self.get_tilt_up())
        return await self._wait(self.pause)

    async def look_right_down(self):
        if not self.event.is_set():
            return False
        self.move_pan(PAN_LIMIT_RIGHT)
        self.ms.sweep_to(TILT_DOWN)
        return await self._wait(self.pause)

    async def center(self):
        if not self.event.is_set():
            return False
        self.ms.sweep_to(self.get_tilt_center())
        self.move_pan(0)
        return await self._wait(0.5)

    async def check_environment(self):
        if self.state["is_sitting"]:
            if not await self.look_left_up(): return False
            if not await self.center(): return False
            if not await self.look_right_up(): return False
            if not await self.center(): return False
            return True
        else:
            if not await self.look_left_up(): return False
            if not await self.look_left_down(): return False
            if not await self.center(): return False
            if not await self.look_right_up(): return False
            if not await self.look_right_down(): return False
            if not await self.center(): return False
            return True


# --- Tracking Logic ---

def track_face(kondo, ms, face, frame_width, frame_height, state):
    if state.get("skip_frames", 0) > 0:
        state["skip_frames"] -= 1
        return False

    state["last_face_seen"] = time.time()
    
    now = time.time()
    if now - state.get("last_track_time", 0) < 0.19:
        return False  
    state["last_track_time"] = now

    if hasattr(face, "bbox") and face.bbox:
        # Check if bbox is the new OpenCV tuple (x, y, w, h) or legacy object
        if isinstance(face.bbox, tuple):
            x, y, w, h = face.bbox
            abs_cx = x + (w / 2.0)
            abs_cy = y + (h / 2.0)
        else:
            box = face.bbox
            abs_cx = box.xmin + (box.xmax - box.xmin) / 2
            abs_cy = box.ymin + (box.ymax - box.ymin) / 2
    else:
        return False

    cx = abs_cx / frame_width
    cy = abs_cy / frame_height

    error_x = 0.5 - cx
    error_y = 0.5 - cy

    pan_change = (error_x * TRACKING_GAIN_PAN) * TRACKING_DIRECTION
    tilt_change = -1 * (error_y * (TRACKING_GAIN_TILT / 2))

    if pan_change > MAX_PAN_STEP:
        pan_change = MAX_PAN_STEP
    elif pan_change < -MAX_PAN_STEP:
        pan_change = -MAX_PAN_STEP

    moved = False

    if abs(error_x) > TRACKING_DEADZONE:
        current_pan = state["current_pan"]
        new_pan = current_pan + pan_change
        new_pan = max(PAN_LIMIT_RIGHT, min(PAN_LIMIT_LEFT, new_pan))

        if DEBUG_TRACKING:
            print(f"[TRACK] NormX: {cx:.2f} | Err: {error_x:.2f} | Pan: {current_pan:.1f} -> {new_pan:.1f}")

        state["current_pan"] = new_pan
        kondo.set_angle(SERVO_ID_HEAD_PAN, new_pan, PAN_SPEED_FAST)
        moved = True

    if abs(error_y) > TRACKING_DEADZONE:
        new_tilt = state["current_tilt"] + tilt_change
        new_tilt = max(TILT_UP, min(TILT_DOWN, new_tilt))
        new_tilt = max(500, new_tilt)

        ms.set_us(new_tilt)
        moved = True

    return moved