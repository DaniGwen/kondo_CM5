import sys
import asyncio
import time

# Hardware Libraries
from pykondo import Kondo
from aiymakerkit import vision
import models

# Custom Modules
from modules import head_tracking
from modules import distance_sensor
from modules.config import *
from modules.motion_controller import MotionController
from modules import behaviors
from modules import system_utils

# --- Global State ---
rotate_head_event = asyncio.Event()

# Shared State
state = {
    "last_face_seen_time": time.time(),
    "last_motion_time": 0,
    "is_sitting": False,
    "wave_played": False,
    "motion_cycle_index": 0,
    "distance_cm": 999.0,
    "current_pan": head_tracking.PAN_CENTER,
    "current_tilt": head_tracking.TILT_CENTER,
    "return_steps_pending": 0,
}

# --- Async Tasks ---


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
            await mc.run(
                MOTION_SIT_DOWN_ID, state=state, sleep_time=4.0, continuous=True
            )

            print(f"Disabling Torque (Motion {MOTION_TORQUE_OFF_ID})...")
            await mc.run(MOTION_TORQUE_OFF_ID, state=state, sleep_time=3.0)

        if not await head.check_environment():
            continue
        await asyncio.sleep(5.0 if state["is_sitting"] else 1.0)


async def detect_logic(mc, ms):
    print("Loading Vision Model...")
    detector = vision.Detector(models.FACE_DETECTION_MODEL)
    frames = vision.get_frames(size=CAMERA_RESOLUTION)
    print("Vision System Ready. (Tracking Debug: ON)")

    face_lost_time = None
    try:
        while True:
            try:
                frame = next(frames)
                frame_height, frame_width = (
                    frame.shape[:2] if hasattr(frame, "shape") else frame.size[::-1]
                )
            except StopIteration:
                await asyncio.sleep(0.01)
                continue

            faces = detector.get_objects(frame, threshold=0.1)
            vision.draw_objects(frame, faces)
            dist = state["distance_cm"]

            # --- RULE 1: OBSTACLE AVOIDANCE ---
            obstacle_active = await behaviors.check_obstacle_safety(
                dist, state, mc, ms, rotate_head_event
            )

            if faces:
                face_lost_time = None
                
                # --- NEW: Reset the Quick Turn memory when a face is found! ---
                state["lost_turn_done"] = False
                # --------------------------------------------------------------

                # --- RESET CURIOSITY MEMORY ---
                if state.get("curiosity_stage", 0) > 0 or state.get("curiosity_done", False):
                    stage = state.get("curiosity_stage", 0)
                    if stage == 1: state["current_pan"] = 45  
                    elif stage == 2: state["current_pan"] = -45  

                    state["curiosity_stage"] = 0
                    state["curiosity_done"] = False

                if not obstacle_active:
                    # --- RULE 2: WAKE UP ---
                    if await behaviors.handle_wake_up(state, mc, ms, rotate_head_event):
                        continue

                    if rotate_head_event.is_set():
                        print("!!! FACE DETECTED - TRACKING !!!")
                        rotate_head_event.clear()

                    state["last_face_seen_time"] = time.time()

                    # --- RULE 3: HANDSHAKE ---
                    if await behaviors.handle_handshake(dist, state, mc):
                        continue

                    # --- RULE 3.5: INTERACTIONS ---
                    if await behaviors.handle_interactions(state, mc, ms, system_utils.blink_led):
                        continue

                # --- RULE 4: TRACKING ---
                is_handshaking = (getattr(mc, 'current_motion', None) == MOTION_HANDSHAKE_ID)
                if not mc.locked() or is_handshaking:
                    head_tracking.track_face(mc.kondo, ms, faces[0], frame_width, frame_height, state)

            else:
                # --- RULE 5: LOST FACE LOGIC ---
                if face_lost_time is None:
                    face_lost_time = time.time()

                idle_time = time.time() - state.get("last_face_seen_time", time.time())

                # --- RULE 5.5: QUICK TURN ON LOST FACE (NEW) ---
                if not obstacle_active:
                    if await behaviors.handle_lost_face_turn(idle_time, state, mc):
                        continue
                # -----------------------------------------------

                # --- RULE 6: CURIOSITY ---
                if not obstacle_active and not state["is_sitting"]:
                    is_curious = await behaviors.handle_curiosity(idle_time, state, mc, ms)
                else:
                    is_curious = False

                if time.time() - face_lost_time > 2.0:
                    if is_curious:
                        rotate_head_event.clear()
                    elif (
                        not obstacle_active
                        and not rotate_head_event.is_set()
                        and not mc.locked()
                        and (idle_time > 2.0)
                    ):
                        print("Area clear. Resuming environment check.")
                        rotate_head_event.set()

                    if 2.0 < dist < DISTANCE_THRESHOLD:
                        rotate_head_event.clear()
                        if not state["is_sitting"]:
                            asyncio.create_task(system_utils.blink_led(0.5))

            await asyncio.sleep(0.001)
    finally:
        print("[SHUTDOWN] Releasing Camera Hardware...")
        if hasattr(frames, "close"):
            frames.close()


async def main():
    print("--- Robot Control System Init ---")
    if not system_utils.ensure_pigpiod():
        sys.exit(1)
    system_utils.setup_gpio()

    ms = head_tracking.MicroServo(TILT_SERVO_PIN, state)
    kondo = Kondo()
    if kondo.init() < 0:
        print("Warning: Kondo Init failed.")

    # Initialize the new Motion Controller
    mc = MotionController(kondo, MOTION_DEFAULT_TIMEOUT)

    print(f"Enabling Torque (Motion {MOTION_TORQUE_ON_ID})...")
    await mc.run(MOTION_TORQUE_ON_ID, state=state, sleep_time=2.0)

    print("Executing HOME POS (ID 2)... Robot will stand/reset.")
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
