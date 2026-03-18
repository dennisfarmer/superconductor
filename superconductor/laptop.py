import cv2
import mediapipe as mp
import numpy as np
import click
import logging
import threading
import json
import time
from pathlib import Path

from superconductor.gesture_recognition import GestureRecognition
from superconductor.recipe_interface import RecipeInterface
from superconductor.magenta_client import MagentaClient

# =========================
# CONFIG
# =========================

audio_device_config = Path("var/config/audio_device.json")


def load_audio_device_config(config_path=audio_device_config):
    if not Path(config_path).exists():
        return None
    try:
        data = json.loads(Path(config_path).read_text())
        return data.get("output_device")
    except:
        return None


# =========================
# MEDIAPIPE
# =========================

class MediaPipeLandmarker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.current_handedness = []
        self.current_hand_landmarks = []
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def __call__(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.current_handedness = results.multi_handedness or []
        self.current_hand_landmarks = [
            hand.landmark for hand in (results.multi_hand_landmarks or [])
        ]
        return self.current_handedness, self.current_hand_landmarks

    def draw(self, frame, overlay, text=""):
        h, w, _ = frame.shape

        for i, hand in enumerate(self.current_hand_landmarks):
            for s, e in self._connections:
                if s < len(hand) and e < len(hand):
                    p1 = (int(hand[s].x * w), int(hand[s].y * h))
                    p2 = (int(hand[e].x * w), int(hand[e].y * h))
                    cv2.line(frame, p1, p2, (121, 237, 116), 2)

            for lm in hand:
                p = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, p, 3, (88, 205, 54), -1)

        if text:
            cv2.putText(overlay, text, (30, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (88, 205, 54), 2)


# =========================
# (KEPT BUT UNUSED)
# =========================

class PlaybackQueue:
    def __init__(self):
        self.deque = []

    def append_raw(self, *args, **kwargs):
        pass

    def pop(self, *args, **kwargs):
        return None

    def refresh(self):
        self.deque.clear()


# =========================
# FRONTEND
# =========================

class Frontend:
    def __init__(self, prompts):

        # ---- core state ----
        self.signals = {"shutdown": False}
        self.last_update = 0

        # ---- magenta ----
        self.magenta_client = MagentaClient()
        self.magenta_client.start()

        # ---- vision ----
        self.webcam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        print("Camera opened:", self.webcam.isOpened())
        if not self.webcam.isOpened():
            raise RuntimeError("Cannot open webcam")
        time.sleep(1)

        self.landmarker = MediaPipeLandmarker()

        # ---- gesture ----
        self.model_name = "palm_hold_release"
        self.gesture_recognition = GestureRecognition(self.model_name)
        self.gesture_recognition.initialize_model()

        # ---- UI ----
        self.recipe_interface = RecipeInterface(
            prompts=prompts,
            slider_up_gesture="palm_hold",
            slider_neutral_gesture="palm_release",
            slider_down_gesture="palm_hold",
            on_recipe_change=self.send_recipe,
        )

        self.run()

    # =========================
    # MAGENTA
    # =========================

    def send_recipe(self, recipe):
        if not self.magenta_client.connected:
            return

        now = time.time()
        if now - self.last_update < 0.1:
            return

        self.last_update = now
        self.magenta_client.update_recipe(recipe)

    # =========================
    # SHUTDOWN
    # =========================

    def stop(self):
        if self.signals["shutdown"]:
            return

        print("Shutting down...")
        self.signals["shutdown"] = True
        time.sleep(0.1)

        if self.webcam.isOpened():
            self.webcam.release()

        cv2.destroyAllWindows()

        if self.magenta_client:
            self.magenta_client.stop()

        print("Shutdown complete")

    # =========================
    # MAIN LOOP
    # =========================

    def run(self):
        print("Press Q to exit")

        while not self.signals["shutdown"]:
            success, frame = self.webcam.read()

            if not success or frame is None:
                print("Camera read failed")
                time.sleep(0.05)
                continue
                continue

            key = cv2.waitKey(1) & 0xFF

            h, w, _ = frame.shape
            overlay = np.zeros_like(frame)

            handedness, landmarks = self.landmarker(frame)

            gesture_name = None
            confidence = 0

            if landmarks:
                tensor = self.gesture_recognition.mediapipe_to_tensor(
                    handedness, landmarks, "Left"
                )
                tensor = self.gesture_recognition.expand_one_hand_to_two_hands(
                    tensor, "Left"
                )

                gesture_name, confidence = self.gesture_recognition(tensor)

                finger = landmarks[0][12]
                x = int((1 - finger.x) * w)
                y = int(finger.y * h)

                self.recipe_interface.update_positions(
                    pointer_x=x,
                    pointer_y=y,
                    gesture=gesture_name
                )

            # draw
            label = f"{gesture_name} ({confidence:.1f}%)" if gesture_name else "no label"
            self.landmarker.draw(frame, overlay, label)
            self.recipe_interface.draw_bars(frame, overlay)

            frame = cv2.flip(frame, 1)
            frame = cv2.add(frame, overlay)

            cv2.imshow("SuperConductor", frame)

            if key == ord("q"):
                self.stop()
                break


# =========================
# CLI
# =========================

@click.command()
@click.option("--loglevel", default="info")
def main(loglevel):
    logging.basicConfig(level=loglevel.upper())
    Frontend(prompts=["Piano", "Flute", "Trumpet"])


if __name__ == "__main__":
    main()