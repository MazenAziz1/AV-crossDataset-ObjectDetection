import time
import threading


class RuntimeGuard:
    def __init__(self, max_hours=10.5, check_interval=300, grace_minutes=5):
        self.max_seconds = max_hours * 3600
        self.check_interval = check_interval
        self.grace_seconds = grace_minutes * 60
        self.start_time = time.time()
        self._stop_requested = False
        self._lock = threading.Lock()

    def elapsed_hours(self):
        return (time.time() - self.start_time) / 3600

    def remaining_seconds(self):
        return max(0, self.max_seconds - (time.time() - self.start_time))

    def should_stop(self):
        return self._stop_requested or self.remaining_seconds() <= self.grace_seconds

    def request_stop(self):
        with self._lock:
            self._stop_requested = True

    def wait_for_interval(self, current_epoch, total_epochs):
        elapsed = self.elapsed_hours()
        remaining = self.remaining_seconds() / 3600
        print(
            f"[Runtime] Epoch {current_epoch}/{total_epochs} | "
            f"Elapsed: {elapsed:.1f}h | Remaining: {remaining:.1f}h"
        )
        if self.remaining_seconds() <= 0:
            return True
        time.sleep(min(self.check_interval, self.remaining_seconds()))
        return self.should_stop()
