import time
import asyncio
from modules import manual_servo_controller
from modules.config import *
from modules import head_tracking


async def startup_deep_scan(mc, ms, state):
    """Twists the torso and leans deeply to check the immediate floor area."""
    if state.get("macro_active", False): return
    state["macro_active"] = True
    
    print(f"{COLOR_MAGENTA}--- INITIATING DEEP PROXIMITY SCAN ---{COLOR_RESET}")
    try:
        # Tilt head down to look at the floor
        ms.sweep_to(head_tracking.TILT_DOWN)
        
        # Lean 25deg and Twist Left (Assuming positive is left for Torso ID 0)
        print(f"{COLOR_YELLOW}Scanning Left Side...{COLOR_RESET}")
        await manual_servo_controller.deep_scan_pose(mc, state, 35.0)
        await asyncio.sleep(2.5) # Give the camera time to recognize objects!
        
        # Lean 25deg and Twist Right
        print(f"{COLOR_YELLOW}Scanning Right Side...{COLOR_RESET}")
        await manual_servo_controller.deep_scan_pose(mc, state, -35.0)
        await asyncio.sleep(2.5)
        
        print(f"{COLOR_MAGENTA}--- SCAN COMPLETE ---{COLOR_RESET}")
    finally:
        # Always reset posture when done
        await manual_servo_controller.reset_posture(mc, state)
        ms.center()
        state["macro_active"] = False


async def return_to_base(mc, state):
    """Reads the breadcrumb history and reverses the exact steps taken."""
    history = state.get("path_history", [])
    if not history:
        return
        
    print(f"{COLOR_YELLOW}!!! INVESTIGATION COMPLETE - REVERSING PATH TO BASE !!!{COLOR_RESET}")
    
    # Read the history list backwards!
    for action, duration in reversed(history):
        if action == "fwd":
            print(f"{COLOR_YELLOW}Reversing: Walking Backward for {duration}s{COLOR_RESET}")
            while mc.locked(): await asyncio.sleep(0.1)
            await mc.run(MOTION_WALK_BACKWARD_ID, state=state, sleep_time=duration, continuous=True)
            
    # Wipe the memory clean once he is back to his starting spot
    state["path_history"] = []
    state["needs_home_reset"] = True

async def check_obstacle_safety(dist, state, mc, ms, rotate_head_event):
    """Priority 1: Checks for obstacles and manages continuous fluid walking."""

    # Never try to avoid obstacles or emergency pause if the motors are off!
    if state.get("is_sitting", False):
        return False

    is_walking_back = state.get("is_walking_back", False)
    is_walking_fwd = state.get("is_walking_fwd", False)

    # --- THE "CHILL ZONE" THRESHOLDS ---
    START_BACK_DIST = 15.0
    STOP_BACK_DIST = 25.0

    START_FWD_DIST = 70.0
    STOP_FWD_DIST = 35.0

    MIN_WALK_TIME = 1.2

    ir_right = state.get("ir_right", 0)
    ir_left = state.get("ir_left", 0)
    foot_obstacle_detected = (ir_right > IR_DANGER_THRESHOLD) or (
        ir_left > IR_DANGER_THRESHOLD
    )

    # --- DEBOUNCE TIMER FOR CLEAR PATH ---
    if dist > START_FWD_DIST and not foot_obstacle_detected:
        if state.get("clear_path_start") is None:
            state["clear_path_start"] = time.time()
    else:
        state["clear_path_start"] = None

    # 1. EMERGENCY PAUSE (Check this FIRST so we don't crash while returning)
    if is_walking_fwd and (dist <= STOP_FWD_DIST or foot_obstacle_detected):
        walked_time = time.time() - state.get("walk_fwd_start_time", time.time())
        if walked_time < MIN_WALK_TIME:
            now = time.time()
            state["return_steps_pending"] -= now - state.get(
                "last_fwd_update_time", now
            )
            state["last_fwd_update_time"] = now
            return True
        
        state["emergency_stop"] = True
        
        # Custom print message so you know WHICH sensor stopped the robot!
        if foot_obstacle_detected:
            print(
                f"{COLOR_CYAN}!!! LOW OBSTACLE DETECTED (L:{ir_left} R:{ir_right}) - PAUSING !!!{COLOR_RESET}"
            )
        else:
            print(
                f"{COLOR_RED}!!! PATH BLOCKED ({dist:.1f}cm) - PAUSING IN BUFFER ZONE !!!{COLOR_RESET}"
            )

        state["is_walking_fwd"] = False
        state["current_pan"] = 0
        state["needs_home_reset"] = True
        state["last_motion_time"] = time.time()
        return True

    # 2. START OR CONTINUE BACKWARD WALK
    if dist < START_BACK_DIST:
        if mc.locked() and not is_walking_back:
            return False

        if not is_walking_back:
            print(
                f"{COLOR_RED}!!! OBSTACLE ({dist:.1f}cm) - STARTING CONTINUOUS BACKWARD WALK !!!{COLOR_RESET}"
            )
            
            state["emergency_stop"] = True
            state["is_walking_back"] = True
            state["is_walking_fwd"] = False
            rotate_head_event.clear()

            mc.play_continuous(MOTION_WALK_BACKWARD_ID)
            state["walk_back_start_time"] = time.time()

        return True

    # 3. STOP BACKWARD WALK (Enter the Chill Zone)
    elif is_walking_back and dist >= STOP_BACK_DIST:
        walked_time = time.time() - state.get("walk_back_start_time", time.time())

        if walked_time < MIN_WALK_TIME:
            return True

        print(
            f"{COLOR_MAGENTA}!!! OBSTACLE CLEARED ({dist:.1f}cm) - ENTERING CHILL ZONE !!!{COLOR_RESET}"
        )
        state["is_walking_back"] = False
        state["return_steps_pending"] = (
            state.get("return_steps_pending", 0) + walked_time
        )
        state["current_pan"] = 0

        state["needs_home_reset"] = True
        state["last_motion_time"] = time.time()
        return True

    # 4. START OR CONTINUE FORWARD WALK (Spending Return Credits)
    is_path_stably_clear = (state.get("clear_path_start") is not None) and (
        time.time() - state["clear_path_start"] > 1.5
    )

    if is_path_stably_clear and state.get("return_steps_pending", 0) > 0:
        if mc.locked() and not is_walking_fwd:
            return False

        if not is_walking_fwd:
            print(
                f"{COLOR_MAGENTA}!!! PATH STABLY CLEAR - SPENDING RETURN CREDITS !!!{COLOR_RESET}"
            )
            state["is_walking_fwd"] = True
            rotate_head_event.clear()

            mc.play_continuous(MOTION_WALK_FORWARD_ID)
            state["walk_fwd_start_time"] = time.time()
            state["last_fwd_update_time"] = time.time()

        now = time.time()
        dt = now - state.get("last_fwd_update_time", now)
        state["last_fwd_update_time"] = now
        state["return_steps_pending"] -= dt

        if state["return_steps_pending"] <= 0:
            walked_time = time.time() - state.get("walk_fwd_start_time", time.time())

            if walked_time < MIN_WALK_TIME:
                return True

            print(
                f"{COLOR_MAGENTA}!!! RETURN COMPLETE - STOPPING FORWARD WALK !!!{COLOR_RESET}"
            )
            state["is_walking_fwd"] = False
            state["return_steps_pending"] = 0
            state["current_pan"] = 0

            state["needs_home_reset"] = True
            state["last_motion_time"] = time.time()

        return True

    return False


async def handle_wake_up(state, mc, ms, rotate_head_event):
    """Priority 2: Wakes the robot up from a sitting position when a face/target is seen."""
    if state["is_sitting"] and not mc.locked():
        print(f"{COLOR_MAGENTA}!!! TARGET DETECTED - WAKING UP !!!{COLOR_RESET}")
        state["is_sitting"] = False
        rotate_head_event.clear()

        ms.center()
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )

        print(
            f"{COLOR_MAGENTA}Enabling Torque (Motion {MOTION_TORQUE_ON_ID})...{COLOR_RESET}"
        )
        await mc.run(MOTION_TORQUE_ON_ID, state=state, sleep_time=3.0)

        print(f"{COLOR_MAGENTA}Standing Up...{COLOR_RESET}")
        await mc.run(MOTION_STAND_UP_ID, state=state)

        state["last_face_seen_time"] = time.time()
        return True

    return False


async def handle_interactions(state, mc, ms, blink_callback):
    """Priority 3: Plays random or specific interaction motions based on a cooldown timer."""
    now = time.time()
    time_since_motion = now - state["last_motion_time"]

    if time_since_motion >= MOTION_GLOBAL_COOLDOWN and not mc.locked():
        if state.get("needs_home_reset", False):
            print(
                f"{COLOR_MAGENTA}!!! ADJUSTING POSTURE FOR INTERACTION !!!{COLOR_RESET}"
            )
            mc.run_background(
                MOTION_HOME_STAND_ID, state=state, sleep_time=1.5, continuous=True
            )
            state["needs_home_reset"] = False
            return True

        print(f"{COLOR_MAGENTA}Starting Interaction Sequence...{COLOR_RESET}")
        state["last_motion_time"] = time.time()

        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )
        ms.center()
        state["current_pan"] = 0

        if blink_callback:
            asyncio.create_task(blink_callback(2))

        mid = INTERACTION_MOTIONS[state["motion_cycle_index"]]
        state["motion_cycle_index"] = (state["motion_cycle_index"] + 1) % len(
            INTERACTION_MOTIONS
        )

        print(f"{COLOR_MAGENTA}Playing Interaction Motion: {mid}{COLOR_RESET}")
        await mc.run(mid, state=state)
        return True

    return False


async def handle_handshake(dist, state, mc):
    """Priority 3.5: Offers a handshake if a person stands in the chill zone."""
    if mc.locked() or state.get("is_sitting", False):
        return False

    # Prevent handshake spamming - require 15 seconds between handshakes
    if time.time() - state.get("last_motion_time", 0) < 15.0:
        state["handshake_start"] = None
        return False

    # Jitter-Proof Zone
    if 25.0 <= dist <= 45.0:
        if state.get("handshake_start") is None:
            state["handshake_start"] = time.time()

    elif dist < 20.0 or dist > 50.0:
        state["handshake_start"] = None

    if state.get("handshake_start") is not None and (
        time.time() - state["handshake_start"] > 3.0
    ):
        if state.get("needs_home_reset", False):
            print(
                f"{COLOR_MAGENTA}!!! ADJUSTING POSTURE FOR HANDSHAKE !!!{COLOR_RESET}"
            )
            mc.run_background(
                MOTION_HOME_STAND_ID, state=state, sleep_time=1.5, continuous=True
            )
            state["needs_home_reset"] = False
            return True

        print(
            f"{COLOR_MAGENTA}!!! OFFERING HANDSHAKE (Waiting for touch sensor) !!!{COLOR_RESET}"
        )
        mc.run_background(
            MOTION_HANDSHAKE_ID, state=state, sleep_time=15.0, continuous=True
        )

        state["handshake_start"] = None
        state["last_motion_time"] = time.time()
        return True

    return False


async def handle_lost_face_turn(idle_time, state, mc):
    """Priority 5.5: Turns the body if the face disappears off the edge of the screen."""
    if mc.locked() or state.get("is_sitting", False):
        return False

    if idle_time > 1.0 and not state.get("lost_turn_done", False):
        pan = state.get("current_pan", 0)
        state["lost_turn_done"] = True

        if pan >= 30.0:
            print(
                f"{COLOR_YELLOW}!!! TARGET LOST LEFT - QUICK SEARCH TURN !!!{COLOR_RESET}"
            )
            mc.run_background(
                MOTION_TURN_LEFT_ID, state=state, sleep_time=1.5, continuous=True
            )
            state["current_pan"] = 0
            state["needs_home_reset"] = True
            return True

        elif pan <= -30.0:
            print(
                f"{COLOR_YELLOW}!!! TARGET LOST RIGHT - QUICK SEARCH TURN !!!{COLOR_RESET}"
            )
            mc.run_background(
                MOTION_TURN_RIGHT_ID, state=state, sleep_time=1.5, continuous=True
            )
            state["current_pan"] = 0
            state["needs_home_reset"] = True
            return True

    return False


async def handle_curiosity(idle_time, state, mc, ms):
    """Priority 6: Plays a smooth left/right scanning animation when bored."""
    if state.get("curiosity_done", False) or state.get("is_sitting", False):
        return False

    if idle_time < 5.0:
        return False

    if mc.locked():
        return False

    stage = state.get("curiosity_stage", 0)
    stage_time = time.time() - state.get("curiosity_stage_start", time.time())

    if stage == 0:
        print(f"{COLOR_YELLOW}??? Where did you go? (Searching Left) ???{COLOR_RESET}")
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 45, head_tracking.PAN_SPEED_NORMAL
        )
        try:
            ms.set_angle(15)
        except:
            pass

        state["curiosity_stage"] = 1
        state["curiosity_stage_start"] = time.time()
        return True

    elif stage == 1 and stage_time > 2.0:
        print(f"{COLOR_YELLOW}??? (Searching Right) ???{COLOR_RESET}")
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, -45, head_tracking.PAN_SPEED_NORMAL
        )

        state["curiosity_stage"] = 2
        state["curiosity_stage_start"] = time.time()
        return True

    elif stage == 2 and stage_time > 2.0:
        print(f"{COLOR_YELLOW}??? (Checking Center) ???{COLOR_RESET}")
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )
        try:
            ms.center()
        except:
            pass

        state["curiosity_stage"] = 3
        state["curiosity_stage_start"] = time.time()
        return True

    elif stage == 3 and stage_time > 1.5:
        print(f"{COLOR_YELLOW}... Guess they left. (Search Complete) ...{COLOR_RESET}")
        state["curiosity_done"] = True
        state["curiosity_stage"] = 0
        return False

    return True


async def investigation_sequence(mc, state):
    """A mini-macro that investigates. Only crouches if the object is low."""
    if state.get("macro_active", False):
        return

    state["macro_active"] = True
    state["emergency_stop"] = False

    try:
        await manual_servo_controller.reset_posture(mc, state)
        while mc.locked(): await asyncio.sleep(0.1)

        # --- THE FIX: Smart Leaning ---
        # If the IR sensors tripped, the object is at his feet and below the camera. Lean down!
        ir_tripped = state.get("ir_left", 0) > 250 or state.get("ir_right", 0) > 250
        
        if ir_tripped:
            print(f"{COLOR_MAGENTA}Object is low/lost. Crouching to confirm...{COLOR_RESET}")
            if not state.get("emergency_stop"):
                state["investigating_low_object"] = True
                await mc.run(MOTION_CROUCH_ID, state=state, sleep_time=4.0, continuous=True)
        else:
            print(f"{COLOR_MAGENTA}Object is in clear view. No crouch required.{COLOR_RESET}")
            await asyncio.sleep(2.0) # Just stare at it proudly for 2 seconds

        # Stand back up if we crouched
        if not state.get("emergency_stop"):
            print(f"{COLOR_MAGENTA}!!! INSPECTION COMPLETE - READY !!!{COLOR_RESET}")
            while mc.locked(): await asyncio.sleep(0.1)
            await mc.run(MOTION_HOME_STAND_ID, state=state, sleep_time=1.5, continuous=True)
            state["needs_home_reset"] = False

        if not state.get("emergency_stop"):
            await return_to_base(mc, state)
            
    finally:
        state["macro_active"] = False
        state["investigating_low_object"] = True


async def handle_investigation(dist, target_name, state, mc):
    """Priority 3.8: Approaches and investigates unknown objects."""
    if mc.locked() or state.get("is_sitting", False):
        return False

    investigated_list = state.setdefault("investigated_items", [])
    if target_name in investigated_list:
        return False

    if dist > MAX_SCAN_DISTANCE:
        return False

    # Check the foot sensors
    ir_left = state.get("ir_left", 0)
    ir_right = state.get("ir_right", 0)
    ir_detected = (ir_left > 250) or (ir_right > 250)

    # 3. APPROACH PHASE: Between 25cm and 70cm (AND IR hasn't tripped yet)
    if 25.0 <= dist <= 70.0 and not ir_detected:
        is_first_sight = False
        if state.get("current_tracking_target") != target_name:
            is_first_sight = True

        required_cooldown = 6.5 if is_first_sight else 3.0

        if time.time() - state.get("last_approach_time", 0) < required_cooldown:
            return False

        if is_first_sight:
            state["current_tracking_target"] = target_name

        print(f"{COLOR_MAGENTA}!!! APPROACHING UNKNOWN TARGET: [{target_name}] ({dist:.1f}cm) !!!{COLOR_RESET}")
        asyncio.create_task(approach_sequence(mc, state, is_first_sight))
        state["last_approach_time"] = time.time()
        return True

    # 4. INSPECT PHASE: Closer than 25cm OR IR sensor detects it at the feet!
    if dist < 25.0 or ir_detected:
        print(f"{COLOR_MAGENTA}!!! REACHED TARGET: [{target_name}] !!!{COLOR_RESET}")

        state["current_tracking_target"] = None 
        investigated_list.append(target_name)
        if len(investigated_list) > 5:
            investigated_list.pop(0)

        asyncio.create_task(investigation_sequence(mc, state))
        return True

    return False


async def approach_sequence(mc, state, is_first_sight=False):
    """Straightens posture, optionally points, and takes a step."""
    
    # 1. Prevent overlapping sequences
    if state.get("macro_active", False):
        return
        
    state["macro_active"] = True
    state["emergency_stop"] = False # This allows safety sensors to abort the walk!

    try:
        # 2. SAFETY OVERRIDE: Stand up straight before doing anything!
        await manual_servo_controller.reset_posture(mc, state)

        if is_first_sight and not state.get("emergency_stop"):
            await manual_servo_controller.point_at_target(mc, state)

        # 3. Reset wide stance if needed
        if state.get("needs_home_reset", False) and not state.get("emergency_stop"):
            
            # --- THE FIX: Patiently wait for the hardware to unlock ---
            while mc.locked():
                await asyncio.sleep(0.1)
                
            await mc.run(MOTION_HOME_STAND_ID, state=state, sleep_time=1.5, continuous=True)
            state["needs_home_reset"] = False

        # 4. Take a step
        if not state.get("emergency_stop"):
            while mc.locked():
                await asyncio.sleep(0.1)
                
            print(f"{COLOR_MAGENTA}Taking a step forward...{COLOR_RESET}")
            
            step_duration = 2.0
            await mc.run(MOTION_WALK_FORWARD_ID, state=state, sleep_time=step_duration, continuous=True)
            state.setdefault("path_history", []).append(("fwd", step_duration))
            state["needs_home_reset"] = True

    finally:
        # 5. Always release the lock when finished, even if aborted
        state["macro_active"] = False


async def handle_persistent_search(idle_time, state, mc, ms):
    """Priority 5.6: A persistent Up/Down then Left/Right search for a lost target."""

    # If we already finished searching, or the robot is busy/sitting, do nothing.
    if state.get("search_done", False) or state.get("is_sitting", False) or mc.locked():
        return False

    # Wait 1.5 seconds after losing the target before panicking and searching
    if idle_time < 3.0:
        return False

    if state.get("search_stage", 0) > 0 and state.get("search_stage_start") is None:
        return False

    stage = state.get("search_stage", 0)
    stage_time = time.time() - state.get("search_stage_start", time.time())

    if stage == 0:
        print(f"{COLOR_YELLOW}??? Target Lost! Searching UP... ???{COLOR_RESET}")
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )
        ms.sweep_to(head_tracking.TILT_UP)
        state["search_stage"] = 1
        state["search_stage_start"] = time.time()
        return True

    elif stage == 1 and stage_time > 1.5:
        print(
            f"{COLOR_YELLOW}??? Searching DOWN (Engaging posture assist)... ???{COLOR_RESET}"
        )
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )

        # Pushing the tilt down automatically triggers handle_dynamic_lean!
        ms.sweep_to(head_tracking.TILT_DOWN)

        state["search_stage"] = 2
        state["search_stage_start"] = time.time()
        return True

    elif stage == 2 and stage_time > 2.5:  # Extra time given for the lean to complete
        print(f"{COLOR_YELLOW}??? Searching LEFT... ???{COLOR_RESET}")
        ms.center()  # This makes the head look up, which automatically straightens the legs!
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 45, head_tracking.PAN_SPEED_NORMAL
        )
        state["search_stage"] = 3
        state["search_stage_start"] = time.time()
        return True

    elif stage == 3 and stage_time > 1.5:
        print(f"{COLOR_YELLOW}??? Searching RIGHT... ???{COLOR_RESET}")
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, -45, head_tracking.PAN_SPEED_NORMAL
        )
        state["search_stage"] = 4
        state["search_stage_start"] = time.time()
        return True

    elif stage == 4 and stage_time > 1.5:
        print(f"{COLOR_YELLOW}... Target is gone. Giving up search. ...{COLOR_RESET}")
        ms.center()
        mc.kondo.set_angle(
            head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL
        )

        state["search_done"] = True
        state["search_stage"] = 0
        return False

    return True  # We are actively animating
