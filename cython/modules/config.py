# --- Hardware & GPIO ---
TRIG_PIN = 24
ECHO_PIN = 23
LED_PIN = 17
TILT_SERVO_PIN = 18

# --- Motion IDs ---
MOTION_HOME_STAND_ID = 2
MOTION_WAVE_ID = 3
MOTION_CROUCH_ID = 20
MOTION_SIT_DOWN_ID = 21
MOTION_STAND_UP_ID = 22
MOTION_TORQUE_OFF_ID = 23
MOTION_TORQUE_ON_ID = 24
MOTION_WALK_FORWARD_ID = 41
MOTION_WALK_BACKWARD_ID = 42
MOTION_TURN_LEFT_ID = 51
MOTION_TURN_RIGHT_ID = 52

# --- Interaction IDs ---
MOTION_BANG_BANG_ID = 64
MOTION_THE_FINGER_ID = 68
MOTION_HANDSHAKE_ID = 65

INTERACTION_MOTIONS = [
    MOTION_THE_FINGER_ID,
    MOTION_BANG_BANG_ID,
]

# --- Timing & Thresholds ---
MOTION_DEFAULT_TIMEOUT = 15.0
MOTION_GLOBAL_COOLDOWN = 40.0
SIT_DOWN_THRESHOLD = 50.0
OBSTACLE_AVOIDANCE_THRESHOLD = 15.0  
DISTANCE_THRESHOLD = 15.0

# --- CAMERA & VISION SETTINGS ---
# Lower resolution = Faster reaction time (less lag). 
# Try (320, 240) for max speed, or (640, 480) for better distance detection.
CAMERA_RESOLUTION = (320, 240)

# --- TERMINAL COLORS ---
COLOR_RED = "\033[91m"     # Important / Errors / Obstacles
COLOR_YELLOW = "\033[93m"  # Warnings
COLOR_MAGENTA = "\033[95m" # Information / Targets
COLOR_CYAN = '\033[96m'
COLOR_RESET = "\033[0m"    # Reset back to default terminal color

# --- LEG SERVO IDs (For Dynamic Leaning) ---
SERVO_ID_R_THIGH_PITCH = 15
SERVO_ID_L_THIGH_PITCH = 14 
LEAN_DIR_R = -1.0             # One side moves positive to go forward
LEAN_DIR_L = 1.0            # The other side moves negative to go forward

# --- KNEE SERVOS ---
SERVO_ID_L_KNEE = 16
SERVO_ID_R_KNEE = 17
KNEE_DIR_L = 1.0  
KNEE_DIR_R = -1.0
KNEE_HOME_ANGLE = -10.0

# --- ARM SERVO IDs (For Counterbalance) ---
SERVO_ID_R_SHOULDER_PITCH = 3
SERVO_ID_L_SHOULDER_PITCH = 2
ARM_LEAN_DIR_R = 1.0   
ARM_LEAN_DIR_L = -1.0
ARM_HOME_R = 20  # Tweak this if his right arm naturally rests a bit forward/back
ARM_HOME_L = -20

# --- TORSO SERVO ID ---
SERVO_ID_TORSO_PAN = 0

# --- FOOT IR SENSORS (RCB-4 AD Ports) ---
PIN_IR_RIGHT_FOOT = 8
PIN_IR_LEFT_FOOT = 9

# Sharp IR analog values usually INCREASE as objects get closer.
IR_DANGER_THRESHOLD = 250

MAX_SCAN_DISTANCE = 45.0