import time
import asyncio


class MotionController:
    def __init__(self, kondo_instance, default_timeout=15.0):
        self.kondo = kondo_instance
        self.lock = asyncio.Lock()
        self._transitioning = False
        self.default_timeout = default_timeout
        self.current_motion = None

    def locked(self):
        """Returns True if a motion is currently running or about to run."""
        return self.lock.locked() or self._transitioning

    def play_continuous(self, motion_id):
        """Safely fires a continuous motion (like walking) without blocking."""
        if self.locked():
            return False

        print(f"[[ EXECUTE CONTINUOUS MOTION: {motion_id} ]]")
        self.kondo.call_motion(int(motion_id))
        return True

    def run_background(self, motion_id, state=None, sleep_time=None, continuous=False):
        """Instantly claims the lock and schedules the motion in the background."""
        if self.locked():
            print(f"[REJECTED BG] Motion {motion_id} ignored to prevent overlap.")
            return False

        # INSTANT CLAIM: Stops the camera loop from firing a 2nd command!
        self._transitioning = True
        self.current_motion = int(motion_id)
        asyncio.create_task(self._async_run(motion_id, state, sleep_time, continuous))
        return True

    async def run(self, motion_id, state=None, sleep_time=None, continuous=False):
        """Standard awaited blocking run."""
        if self.locked():
            print(f"[REJECTED] Motion {motion_id} ignored to prevent overlap.")
            return False

        self._transitioning = True
        return await self._async_run(motion_id, state, sleep_time, continuous)

    async def _async_run(self, motion_id, state, sleep_time, continuous):
        """The actual locking logic used by both foreground and background calls."""
        async with self.lock:
            self._transitioning = False  # Hand off to the secure asyncio lock
            self.current_motion = int(motion_id)

            print(f"[[ EXECUTE MOTION: {motion_id} ]]")

            if not hasattr(self.kondo, "play_motion") or not hasattr(
                self.kondo, "is_motion_done"
            ):
                print(
                    "[WARNING] Hardware feedback methods missing! Falling back to simple sleep."
                )
                self.kondo.call_motion(int(motion_id))
                wait = sleep_time if sleep_time else 5.0
                await asyncio.sleep(wait)
                if state is not None:
                    state["current_pan"] = 0
                return True

            self.kondo.call_motion(int(motion_id))

            if continuous:
                wait = sleep_time if sleep_time else 2.0
                await asyncio.sleep(wait)
                if state is not None:
                    state["current_pan"] = 0
                return True

            timeout = sleep_time if sleep_time else self.default_timeout
            start_time = time.time()

            await asyncio.sleep(0.1)

            while time.time() - start_time < timeout:
                status = self.kondo.is_motion_done()

                if status == 1:
                    print(f">> MOTION {motion_id} DONE CONFIRMED (Hardware Flag)!")
                    break
                elif status < 0:
                    print(">> Warning: Comms error while reading motion status.")

                await asyncio.sleep(0.1)
            else:
                print(f">> TIMEOUT waiting for motion {motion_id} to finish.")

            if state is not None:
                state["current_pan"] = 0

            self.current_motion = None
            return True
