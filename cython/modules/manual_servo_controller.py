import asyncio
from modules.config import *
from modules import head_tracking


async def handle_dynamic_lean(state, mc):
    """Priority 5.8: Leans the body forward and throws arms back for balance."""
    if (
        mc.locked()
        or state.get("is_sitting", False)
        or state.get("macro_active", False)
    ):
        return False

    # --- LEAN TUNING ---
    MAX_LEAN_ANGLE = 30.0
    LEAN_STEP = 0.9
    ARM_MULTIPLIER = 1.2

    TILT_THRESHOLD = head_tracking.TILT_DOWN - 40

    current_tilt = state.get("current_tilt", head_tracking.TILT_CENTER)
    current_lean = state.get("lean_angle", 0.0)

    needs_update = False

    if current_tilt >= TILT_THRESHOLD:
        if current_lean < MAX_LEAN_ANGLE:
            current_lean = min(MAX_LEAN_ANGLE, current_lean + LEAN_STEP)
            needs_update = True

    elif current_lean > 0.0:
        current_lean = max(0.0, current_lean - LEAN_STEP)
        needs_update = True

    if needs_update:
        state["lean_angle"] = current_lean

        # 1. Pitch the legs forward
        mc.kondo.set_angle(SERVO_ID_R_THIGH_PITCH, current_lean * LEAN_DIR_R, 0.5)
        mc.kondo.set_angle(SERVO_ID_L_THIGH_PITCH, current_lean * LEAN_DIR_L, 0.5)

        # 2. Apply the arm swing as an offset to the ORIGINAL home position
        arm_swing = current_lean * ARM_MULTIPLIER
        mc.kondo.set_angle(SERVO_ID_R_SHOULDER_PITCH, ARM_HOME_R + (arm_swing * ARM_LEAN_DIR_R), 0.5)
        mc.kondo.set_angle(SERVO_ID_L_SHOULDER_PITCH, ARM_HOME_L + (arm_swing * ARM_LEAN_DIR_L), 0.5)

        if current_lean == MAX_LEAN_ANGLE:
            print(
                f"{COLOR_YELLOW}[BALANCE] Maximum forward lean ({MAX_LEAN_ANGLE} deg) engaged.{COLOR_RESET}"
            )

    return False


async def deep_scan_pose(mc, state, torso_angle):
    """Forces a deep 25-degree lean and twists the torso for scanning."""
    scan_lean = 25.0  # Max lean + 5 degrees!

    # 1. Pitch the legs forward
    mc.kondo.set_angle(SERVO_ID_R_THIGH_PITCH, scan_lean * LEAN_DIR_R, 0.3)
    mc.kondo.set_angle(SERVO_ID_L_THIGH_PITCH, scan_lean * LEAN_DIR_L, 0.3)

    # 2. Throw arms back
    arm_swing = scan_lean * 1
    mc.kondo.set_angle(
        SERVO_ID_R_SHOULDER_PITCH, ARM_HOME_R + (arm_swing * ARM_LEAN_DIR_R), 0.3
    )
    mc.kondo.set_angle(
        SERVO_ID_L_SHOULDER_PITCH, ARM_HOME_L + (arm_swing * ARM_LEAN_DIR_L), 0.3
    )

    # 3. Twist the Pelvis!
    mc.kondo.set_angle(SERVO_ID_TORSO_PAN, torso_angle, 0.3)


async def reset_posture(mc, state):
    """Safety override: Instantly stands up straight and returns arms/torso to 0."""
    if state.get("lean_angle", 0.0) > 0 or state.get("macro_active", False):
        print(f"{COLOR_YELLOW}[BALANCE] Resetting posture...{COLOR_RESET}")
        state["lean_angle"] = 0.0

        mc.kondo.set_angle(SERVO_ID_R_THIGH_PITCH, 0, 0.5)
        mc.kondo.set_angle(SERVO_ID_L_THIGH_PITCH, 0, 0.5)
        mc.kondo.set_angle(SERVO_ID_R_SHOULDER_PITCH, ARM_HOME_R, 0.5)
        mc.kondo.set_angle(SERVO_ID_L_SHOULDER_PITCH, ARM_HOME_L, 0.5)
        mc.kondo.set_angle(SERVO_ID_TORSO_PAN, 0, 0.5)

        await asyncio.sleep(0.5)


async def point_at_target(mc, state):
    """Points the correct arm directly at the target based on camera head-tracking feedback."""
    pan = state.get("current_pan", 0)
    tilt = state.get("current_tilt", head_tracking.TILT_CENTER)

    if pan > 0:
        arm_id = SERVO_ID_L_SHOULDER_PITCH
        home_angle = ARM_HOME_L
        direction = ARM_LEAN_DIR_L
        arm_name = "Left"
    else:
        arm_id = SERVO_ID_R_SHOULDER_PITCH
        home_angle = ARM_HOME_R
        direction = ARM_LEAN_DIR_R
        arm_name = "Right"

    point_dir = direction * -1.0
    tilt_offset = tilt - head_tracking.TILT_CENTER

    # Calculate the angle, but never let it drop below 15 degrees forward
    safe_angle = max(15.0, 80.0 - tilt_offset)
    point_angle = home_angle + (safe_angle * point_dir)

    print(
        f"{COLOR_MAGENTA}[GESTURE] Pointing {arm_name} arm exactly at target!{COLOR_RESET}"
    )

    mc.kondo.set_angle(arm_id, point_angle, 0.5)
    await asyncio.sleep(2.5)

    mc.kondo.set_angle(arm_id, home_angle, 0.5)
    await asyncio.sleep(0.5)
