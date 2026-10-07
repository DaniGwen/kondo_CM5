import sys
import asyncio
import time

# Hardware Libraries
from pykondo import Kondo
import vision
from vision import VisionBridge
import models

# Custom Modules
from modules import rcb4_sensor_reader
from modules import head_tracking
from modules import distance_sensor
from modules import manual_servo_controller
from modules.config import *
from modules.motion_controller import MotionController
from modules import behaviors
from modules import system_utils

# --- Global State ---
rotate_head_event = asyncio.Event()

# Shared State
state = {
    "last_face_seen_time": time.time(), # Keeping this name so behaviors.py stays happy!
    "last_motion_time": 0,
    "is_sitting": False,
    "wave_played": False,
    "motion_cycle_index": 0,
    "distance_cm": 999.0,
    "current_pan": head_tracking.PAN_CENTER,
    "current_tilt": head_tracking.TILT_CENTER,
    "return_steps_pending": 0,
    
    # --- NEW: THE ATTENTION HIERARCHY ---
    # The robot will track the first object it finds on this list.
    # If it sees a person and a bottle, it will ignore the person and track the bottle!
    "target_priorities": ["sports ball", "bottle", "person"]
}

def load_labels(file_path):
    """Reads a labels.txt file and returns a dictionary mapping IDs to strings."""
    labels = {}
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if not line: continue
                
                # Check if the line starts with a number (e.g., "0 person")
                parts = line.split(' ', 1)
                if parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].lower()
                else:
                    # Otherwise, assume line-by-line index (e.g., Line 0 = "person")
                    labels[i] = line.lower()
                    
        print(f"Loaded {len(labels)} classes from {file_path}")
    except Exception as e:
        print(f"[WARNING] Could not load {file_path}: {e}")
        
    return labels

async def idle_manager_task(mc, ms):
    print("Starting Idle Manager...")
    head = head_tracking.HeadBehavior(mc.kondo, ms, rotate_head_event, state)

    while True:
        await rotate_head_event.wait()
        idle_time = time.time() - state["last_face_seen_time"]

        if idle_time > SIT_DOWN_THRESHOLD and not state["is_sitting"]:
            print(f"Idle for {idle_time:.1f}s. Sitting Down...")
            
            if state.get("needs_home_reset", False):
                print("Resetting legs before sitting...")
                await mc.run(MOTION_HOME_STAND_ID, state=state, sleep_time=1.5, continuous=True)
                state["needs_home_reset"] = False
                
            state["is_sitting"] = True
            state["is_walking_fwd"] = False
            state["is_walking_back"] = False
            state["return_steps_pending"] = 0

            ms.center()
            head.move_pan(0)
            await asyncio.sleep(0.5)

            print("Executing Sit Down motion...")
            await mc.run(MOTION_SIT_DOWN_ID, state=state, sleep_time=4.0, continuous=True)
            print(f"Disabling Torque (Motion {MOTION_TORQUE_OFF_ID})...")
            await mc.run(MOTION_TORQUE_OFF_ID, state=state, sleep_time=3.0)

        if not await head.check_environment():
            continue
        await asyncio.sleep(5.0 if state["is_sitting"] else 1.0)

async def detect_logic(mc, ms):
    print(f"{COLOR_MAGENTA}Loading Multi-Object Vision Model...{COLOR_RESET}")
    detector = VisionBridge(models.OBJECT_DETECTION_MODEL)
    frames = detector.get_frames(size=CAMERA_RESOLUTION)
    
    label_map = load_labels(models.OBJECT_DETECTION_LABELS) 
    print(f"{COLOR_MAGENTA}Multi-Object Vision System Ready. (Tracking Debug: ON){COLOR_RESET}")

    target_lost_time = None
    try:
        print(f"{COLOR_YELLOW}[INIT] Centering head and resetting state...{COLOR_RESET}")
        state["current_pan"] = 0
        state["current_tilt"] = head_tracking.TILT_CENTER
        mc.kondo.set_angle(head_tracking.SERVO_ID_HEAD_PAN, 0, head_tracking.PAN_SPEED_NORMAL)
        try:
          ms.center()
        except Exception:
          pass
        
        # Give the motors a full second to move into position before opening the camera
        await asyncio.sleep(1.0)
    
        while True:
            try:
                frame = next(frames)
                if frame is None:
                    raise ValueError("Camera returned empty frame")
                    
                frame_height, frame_width = frame.shape[:2] if hasattr(frame, "shape") else frame.size[::-1]
                
            except StopIteration:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                # --- FIX 2: THE AIY MAKER KIT WATCHDOG ---
                print(f"{COLOR_RED}[ERROR] Camera feed dropped! Reconnecting...{COLOR_RESET}")
                
                # Give the hardware a second to clear the bus
                await asyncio.sleep(1.0) 
                
                # Reboot the AIY camera generator
                frames = detector.get_frames(size=CAMERA_RESOLUTION)
                continue

            all_objects = detector.get_objects(frame, threshold=0.2)
            dist = state["distance_cm"]
            rcb4_sensor_reader.update_foot_sensors(mc, state)
            active_target = None
            active_target_name = ""
            
            for priority_label in state["target_priorities"]:
                for obj in all_objects:
                    obj_id = getattr(obj, 'id', None)
                    detected_name = label_map.get(obj_id, getattr(obj, 'label', '')).lower()
                    
                    if detected_name == priority_label.lower():
                        active_target = obj
                        active_target_name = detected_name 
                        break 
                if active_target: break

            unknown_target = None
            unknown_target_name = ""
            if not active_target:
                for obj in all_objects:
                    obj_id = getattr(obj, 'id', None)
                    detected_name = label_map.get(obj_id, getattr(obj, 'label', '')).lower()
                    
                    if detected_name and detected_name not in state["target_priorities"]:
                        unknown_target = obj
                        unknown_target_name = detected_name
                        break

            tracking_target = active_target if active_target else unknown_target
            tracking_target_name = active_target_name if active_target else unknown_target_name

            if tracking_target:
                # If we have a target, ONLY draw the box and label for that specific target
                vision.draw_objects(frame, [tracking_target], labels=label_map)
            else:
                # If the robot is bored/searching, show everything it sees
                vision.draw_objects(frame, all_objects, labels=label_map)
                
            obstacle_active = await behaviors.check_obstacle_safety(dist, state, mc, ms, rotate_head_event)

            if tracking_target:
                target_lost_time = None
                state["lost_turn_done"] = False

                # --- THE FIX: Reset the Search Memory when target is found! ---
                if state.get("search_stage", 0) > 0 or state.get("search_done", False):
                    state["search_stage"] = 0
                    state["search_done"] = False
                # --------------------------------------------------------------

                if not obstacle_active:
                    if await behaviors.handle_wake_up(state, mc, ms, rotate_head_event): continue

                    if rotate_head_event.is_set():
                        print(f"{COLOR_MAGENTA}!!! TARGET ACQUIRED [{tracking_target_name}] - TRACKING !!!{COLOR_RESET}")
                        rotate_head_event.clear()

                    state["last_face_seen_time"] = time.time()

                    if unknown_target:
                        if await behaviors.handle_investigation(dist, unknown_target_name, state, mc): continue

                is_investigating = (getattr(mc, 'current_motion', None) == MOTION_CROUCH_ID)
                if not mc.locked() or is_investigating:
                    head_tracking.track_face(mc.kondo, ms, tracking_target, frame_width, frame_height, state)

            else:
                if target_lost_time is None:
                    target_lost_time = time.time()

                idle_time = time.time() - state.get("last_face_seen_time", time.time())

                if not obstacle_active:
                    if await behaviors.handle_lost_face_turn(idle_time, state, mc): continue

                # --- THE FIX: Trigger the Persistent Search ---
                if not obstacle_active and not state["is_sitting"]:
                    is_searching = await behaviors.handle_persistent_search(idle_time, state, mc, ms)
                else:
                    is_searching = False
                # ----------------------------------------------

                if time.time() - target_lost_time > 2.0:
                    
                    # Lock the background environment checks while actively searching
                    if is_searching:
                        rotate_head_event.clear()
                    elif not obstacle_active and not rotate_head_event.is_set() and not mc.locked() and (idle_time > 2.0):
                        print(f"{COLOR_MAGENTA}Area clear. Resuming environment check.{COLOR_RESET}")
                        rotate_head_event.set()

                    if 2.0 < dist < DISTANCE_THRESHOLD:
                        rotate_head_event.clear()
                        if not state["is_sitting"]:
                            asyncio.create_task(system_utils.blink_led(0.5))

            # Dynamic Balance / Lean Check runs continuously
            await manual_servo_controller.handle_dynamic_lean(state, mc)

            await asyncio.sleep(0.001)
    finally:
        print(f"{COLOR_RED}[SHUTDOWN] Releasing Camera Hardware...{COLOR_RESET}")
        if hasattr(frames, "close"): frames.close()

async def main():
    print(f"{COLOR_MAGENTA}--- Robot Control System Init (Multi-Target Mode) ---{COLOR_RESET}")
    if not system_utils.ensure_pigpiod(): sys.exit(1)
    system_utils.setup_gpio()

    ms = head_tracking.MicroServo(TILT_SERVO_PIN, state)
    kondo = Kondo()
    if kondo.init() < 0: 
        print(f"{COLOR_YELLOW}Warning: Kondo Init failed.{COLOR_RESET}")

    mc = MotionController(kondo, MOTION_DEFAULT_TIMEOUT)

    print(f"{COLOR_MAGENTA}Enabling Torque (Motion {MOTION_TORQUE_ON_ID})...{COLOR_RESET}")
    await mc.run(MOTION_TORQUE_ON_ID, state=state, sleep_time=2.0)

    print(f"{COLOR_MAGENTA}Executing HOME POS (ID 2)... Robot will stand/reset.{COLOR_RESET}")
    await mc.run(MOTION_HOME_STAND_ID, state=state)

    state["current_pan"] = 0
    state["current_tilt"] = head_tracking.TILT_CENTER
    state["is_sitting"] = False
    rotate_head_event.set()

    await asyncio.gather(
        distance_sensor.distance_task(state, TRIG_PIN, ECHO_PIN),
        idle_manager_task(mc, ms),
        detect_logic(mc, ms),
    )
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nStopping...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        try:
            loop.run_until_complete(asyncio.sleep(0.1))
        except asyncio.CancelledError:
            pass
    finally:
        system_utils.cleanup_gpio()
        loop.close()