# clip_saver.py
import cv2
from collections import deque
import threading

class GoalClipSaver:
    def __init__(self, fps, pre_seconds=5, post_seconds=15):
        self.fps = fps
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.frame_buffer = deque(maxlen=int(fps * pre_seconds))
        self.lock = threading.Lock()
        self.clip_index = 0

    def add_frame(self, frame):
        with self.lock:
            self.frame_buffer.append(frame.copy())

    def save_clip(self):
        """Start saving in a separate thread."""
        pre_frames = list(self.frame_buffer)
        t = threading.Thread(target=self._save_clip_worker, args=(pre_frames, self.clip_index))
        t.start()
        self.clip_index += 1

    def _save_clip_worker(self, pre_frames, clip_index):
        if not pre_frames:
            print(f"⚠ Buffer empty, no pre-goal frames for clip {clip_index}. Starting from now.")
            return

        height, width = pre_frames[0].shape[:2]
        out = cv2.VideoWriter(
            f"goal_clip_{clip_index}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (width, height)
        )

        # Write pre-goal frames
        for frame in pre_frames:
            out.write(frame)

        # Capture post-goal frames from live feed
        print(f"📹 Saving {self.post_seconds}s after goal...")
        for _ in range(int(self.post_seconds * self.fps)):
            with self.lock:
                if self.frame_buffer:
                    out.write(self.frame_buffer[-1])

        out.release()
        print(f"✅ Saved highlight goal_clip_{clip_index}.mp4")
