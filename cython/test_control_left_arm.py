import sys
import time
from pykondo import Kondo

# --- Servo IDs (As specified) ---
# Note: Standard KHR-3HV usually uses IDs 1,2 for Right arm and 11,12 for Left.
# We are using YOUR specified IDs:
ID_L_SHOULDER_PITCH = 1
ID_L_SHOULDER_ROLL  = 2  # "Shoulder Row"
ID_L_ELBOW          = 4

# --- Speed Constants ---
# Based on kondo.cpp logic: (255 * (1 - fraction))
# 0.1 results in high byte value (Fast)
# 0.3 results in lower byte value (Normal/Slower)
SPEED_FAST   = 0.1 
SPEED_NORMAL = 0.2
SPEED_SLOW   = 0.3

def main():
    print("--- Left Arm Direct Control Test ---")
    
    # 1. Initialize Kondo Class
    kondo = Kondo()
    ret = kondo.init()
    if ret < 0:
        print("Error: Could not initialize Kondo RCB-4.")
        sys.exit(1)
    
    print("Kondo Initialized.")
    print(f"Targeting IDs: Pitch={ID_L_SHOULDER_PITCH}, Roll={ID_L_SHOULDER_ROLL}, Elbow={ID_L_ELBOW}")

    try:
        # 2. Move to Neutral (0 Degrees)
        # Sending a position command automatically engages torque (Hold)
        print("Moving to Neutral (0 deg)...")
        kondo.set_angle(ID_L_SHOULDER_PITCH, 0, SPEED_NORMAL)
        time.sleep(0.02) # Small delay between commands is good practice
        kondo.set_angle(ID_L_SHOULDER_ROLL, 0, SPEED_NORMAL)
        time.sleep(0.02)
        kondo.set_angle(ID_L_ELBOW, 0, SPEED_NORMAL)
        
        time.sleep(2.0)

        # 3. Shoulder Pitch Movement (Lift Arm Forward)
        print("Shoulder Pitch -> +30 deg")
        kondo.set_angle(ID_L_SHOULDER_PITCH, 30, SPEED_NORMAL)
        time.sleep(1.0)

        # 4. Shoulder Roll Movement (Lift Arm Side)
        print("Shoulder Roll -> +20 deg")
        kondo.set_angle(ID_L_SHOULDER_ROLL, 20, SPEED_NORMAL)
        time.sleep(1.0)

        # 5. Elbow Movement
        print("Elbow -> +45 deg")
        kondo.set_angle(ID_L_ELBOW, 45, SPEED_NORMAL)
        time.sleep(1.0)

        # 6. Reset to Neutral
        print("Resetting to Neutral...")
        kondo.set_angle(ID_L_ELBOW, 0, SPEED_NORMAL)
        time.sleep(0.5)
        kondo.set_angle(ID_L_SHOULDER_ROLL, 0, SPEED_NORMAL)
        time.sleep(0.5)
        kondo.set_angle(ID_L_SHOULDER_PITCH, 0, SPEED_NORMAL)
        
        print("Test Complete.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()