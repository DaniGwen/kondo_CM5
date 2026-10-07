from modules.config import *

def read_ad_port(mc, pin_number):
    """
    Safely reads a single Analog-to-Digital (AD) port from the RCB-4.
    Returns the raw analog value, or 0 if the read fails.
    """
    try:
        # Check which naming convention the cython wrapper uses
        if hasattr(mc.kondo, 'get_analog'):
            return mc.kondo.get_analog(pin_number)
        elif hasattr(mc.kondo, 'get_ad'):
            return mc.kondo.get_ad(pin_number)
    except Exception:
        # If the serial bus is busy, fail safely without crashing the robot
        pass
    
    return 0

def update_foot_sensors(mc, state):
    """
    Reads the specific AD ports for the left and right foot IR sensors
    and updates the robot's state memory.
    """
    state["ir_right"] = read_ad_port(mc, PIN_IR_RIGHT_FOOT)
    state["ir_left"] = read_ad_port(mc, PIN_IR_LEFT_FOOT)