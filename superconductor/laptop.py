#import socket
#import json

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import click
import logging
import threading
import json
import socket
import sys
import time
from collections import deque
from time import sleep
import base64

import io
from pydub import AudioSegment
import sounddevice as sd

from . import networking
from .gesture_recognition import GestureRecognition
from .recipe_interface import RecipeInterface

audio_device_config = Path("var/config/audio_device.json")


def load_audio_device_config(config_path=audio_device_config):
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    output_device = data.get("output_device")
    return output_device if isinstance(output_device, int) else None


@click.command()
@click.option("--list", "list_devices", is_flag=True, help="List output audio devices.")
@click.option("--select", "select_device", type=int, default=None, help="Output device index to save.")
@click.option("--config-file", "config_file", default=str(audio_device_config), show_default=True)
def audio_device(list_devices, select_device, config_file):
    """List/select output audio device and save choice to config file."""
    devices = sd.query_devices()
    output_devices = [
        (idx, d) for idx, d in enumerate(devices)
        if d.get("max_output_channels", 0) > 0
    ]

    if list_devices:
        if not output_devices:
            click.echo("No output devices found.")
        for idx, dev in output_devices:
            click.echo(f"[{idx}] {dev['name']} (out={dev['max_output_channels']})")

    if select_device is not None:
        valid_indices = {idx for idx, _ in output_devices}
        if select_device not in valid_indices:
            raise click.ClickException(f"Invalid output device index: {select_device}")
        cfg_path = Path(config_file)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            json.dumps({"output_device": select_device}, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Saved output device {select_device} to {cfg_path}")

    if not list_devices and select_device is None:
        configured = load_audio_device_config(config_file)
        if configured is None:
            click.echo("No configured output device.")
        else:
            click.echo(f"Configured output device: {configured}")

@click.command()
@click.option(
    "--host", "-h", "host", default="localhost",
    help="laptop host, default=localhost",
)
@click.option(
    "--port", "-p", "port", default=6000,
    help="laptop port number, default=6000",
)
@click.option("--shutdown", "-s", is_flag=True, help="Shutdown the server.")
def update_superconductor(host: str, port: int, shutdown: bool):
    # send messages to running superconductor instance (shutdown, etc..)
    # can add extra arguments later
    # (ex: send server_host and server_port to superconductor instance or something similar)

    if shutdown:
        stop_superconductor(host, port)

    
    # ...

def stop_superconductor(host, port):
    message = json.dumps({
        "message_type": "shutdown"
    })
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host,port))
            sock.sendall(str.encode(message))
    except socket.error as err:
        sys.exit(f"Failed to send message to superconductor: {err}")
    print(f"shut down superconductor {host}:{port}")






class MediaPipeLandmarker:
    def __init__(self):

        task_file = Path("hand_landmarker.task")

        # download hand_landmarker.task for MediaPipe
        if not task_file.exists():
            urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                task_file
                )

        base_options = python.BaseOptions(model_asset_path=str(task_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
    
    
    #def process_landmarks(self, hand_landmarks, handedness):
        #landmarks = []
        #for hand_num in range(len(self.results.hand_landmarks)):
            #current_hand = hand_landmarks[hand_num]
            #current_handedness = handedness[hand_num][0]
            #landmark_list = []
            #for id, landmark in enumerate(current_hand):
                #center_x, center_y, center_z = float(landmark.x), float(landmark.y), float(landmark.z)
                #landmark_list.append([id, center_x, center_y, center_z])
            
            #landmarks.append((current_handedness.category_name, landmark_list))


    def preprocess_image(self, webcam_frame):
        img_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )
        return mp_image

    def draw_overlay_hands(self, webcam_frame, overlay_mask, text_lr:tuple[str, str]=("Left", "Right")):
        """
        text: defaults to drawing "Left" at Left wrist coord, "Right" at Right wrist coord
        """
        height, width, n_channels = webcam_frame.shape
        for idx in range(len(self.current_hand_landmarks)):
            # Retrieve the wrist coordinate:
            wrist_coordinates = self.current_hand_landmarks[idx][0]
            wrist_x = wrist_coordinates.x
            wrist_y = wrist_coordinates.y
            #wrist_z = wrist_coordinates.z
            wrist_handedness = self.current_handedness[idx][0].category_name


            # Draw text at the wrist coordinate (index 0)
            hand_label = wrist_handedness
            if hand_label == "Left":
                text = text_lr[0]
            else:
                text = text_lr[1]

            text_x = width - int(wrist_x * width)
            text_y = int(wrist_y * height)

            # =====================================================================
            font_scale = 1  # negative value to flip text vertically
            font_thickness = 1
            font_color = (88, 205, 54) # vibrant green

            cv2.putText(overlay_mask, text,
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

            # =====================================================================
            # =====================================================================


            # Draw the hand landmarks.
            vision.drawing_utils.draw_landmarks(
                webcam_frame,
                self.current_hand_landmarks[idx],
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.get_default_hand_landmarks_style(),
                vision.drawing_styles.get_default_hand_connections_style())

        #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

    def __call__(self, webcam_frame):
        """
        returns `(handedness, hand_landmarks)`
        """

        results = self.hand_landmarker.detect(
            self.preprocess_image(webcam_frame)
            )

        self.current_handedness = results.handedness
        self.current_hand_landmarks = results.hand_landmarks

        return self.current_handedness, self.current_hand_landmarks


class PlaybackQueue:
    def __init__(self):
        self.deque = deque()

    # CHANGES MADE
    # Old Stop-and-wait base64 JSON ingestion:
    # def append(self, audio_data):
    #     self.deque.append(
    #         self.preprocess_server_audio_from_json(audio_data)
    #     )
    # 
    # def preprocess_server_audio_from_json(self, audio_data):
    #     audio_bytes = base64.b64decode(audio_data)
    #     audio_io = io.BytesIO(audio_bytes)
    #     audio_io.seek(0)
    #     segment = AudioSegment.from_mp3(audio_io)
    # 
    #     sample_array = np.array(segment.get_array_of_samples())
    #     if segment.channels > 1:
    #         sample_array = sample_array.reshape((-1, segment.channels))
    #     max_int = float(1 << (8 * segment.sample_width - 1))
    #     audio = sample_array.astype(np.float32) / max_int
    #     sr = segment.frame_rate
    # 
    #     # audio to the output stream format. For now we assume float32 and 
    #     # a standard blocksize/samplerate on the output stream.
    #     return {
    #         "metadata": "",
    #         "audio": audio,
    #         "sr": sr
    #     }

    def append_raw(self, raw_bytes, channels=2, sr=44100):
        # Decode raw float32 bytes streams representing continuous audio
        audio_array = np.frombuffer(raw_bytes, dtype=np.float32)
        if channels > 1:
            audio_array = audio_array.reshape((-1, channels))
        else:
            audio_array = np.column_stack((audio_array, audio_array))
            
        self.deque.append({
            "metadata": "",
            "audio": audio_array,
            "sr": sr
        })

    # CHANGES MADE
    # Old Stop-and-Wait Queue logic:
    # def pop(self):
    #     return self.deque.popleft()
    # 
    # def __len__(self):
    #     return len(self.deque)

    def pop(self, num_frames=None):
        if len(self.deque) == 0:
            return None
            
        current_chunk = self.deque[0]
        audio = current_chunk["audio"]
        
        if num_frames is None or len(audio) <= num_frames:
            # Pop the whole chunk if it's smaller or equal to what we need
            return self.deque.popleft()
        else:
            # Split the chunk if we only need a portion
            out_chunk = {
                "metadata": current_chunk["metadata"],
                "audio": audio[:num_frames],
                "sr": current_chunk["sr"]
            }
            # Update the remaining chunk in the queue
            current_chunk["audio"] = audio[num_frames:]
            return out_chunk

    def __len__(self):
        return sum(len(chunk["audio"]) for chunk in self.deque)

    def refresh(self):
        # TODO: selectively remove audios from back of deque
        #       based on how much of the original audio we
        #       want to keep
        # currently just empties the deque
        self.deque.clear()


class Frontend:
    def __init__(self, host, port, prompts):
        self.host = host
        self.port = port

        self.server_host = "localhost"
        self.server_port = 6001

        self.signals = {}
        self.signals["shutdown"] = False
        self.lock = threading.Lock()

        self.playback_queue = PlaybackQueue()
        self.output_device = load_audio_device_config()

        self.model_name = "palm_hold_release"
        self.landmarker = MediaPipeLandmarker()
        self.webcam = cv2.VideoCapture(0)
        self.gesture_recognition = GestureRecognition(self.model_name)
        self.gesture_recognition.initialize_model()

        self.recipe_interface = RecipeInterface(
            prompts=prompts,
            slider_up_gesture = "palm_release_up",
            slider_neutral_gesture = "palm_release_down",
            slider_down_gesture = "palm_hold",
            on_recipe_change=self.send_recipe_to_server,
        )

        # create threads and start webcam loop
        self.start()

    def start(self):
        # listen for audio messages from GPU server
        tcp_server = threading.Thread(
            target = networking.tcp_server,
            args=(self.host,self.port, self.signals, self.tcp_handle_func)
        )
        tcp_server.start()

        # udp thread for recieving beat estimates / tempo updates
        # (likely handled differently later idk)
        #udp_server = threading.Thread(
            #target = network.udp_server,
            #args=(self.host, self.port, self.signals, self.udp_handle_func)
        #)
        #udp_server.start()

        ################################################################
        # wait here until server and laptop establish connection (????)
        # may be neccesary when actual GPU server on Lighthouse is being
        # used in place of the mock_local_backend.py
        ################################################################

        playback = threading.Thread(
            target=self.run_playback
        )
        playback.start()

        #webcam = threading.Thread(
            #target=self.run_webcam
        #)
        #webcam.start()
        try:
            self.run_webcam()
        except RuntimeError:
            pass


        # doesn't return until shutdown recieved
        tcp_server.join()
        #webcam.join()

    def stop(self, shutdown_server=False):
        if shutdown_server:
            with self.lock:
                try:
                    networking.tcp_client(
                        self.server_host,
                        self.server_port,
                        {
                            "message_type": "shutdown"
                        }
                    )
                except ConnectionRefusedError:
                    pass

        self.signals["shutdown"] = True

    def tcp_handle_func(self, json_data):
        if json_data["message_type"] == "shutdown":
            self.stop(shutdown_server=True)
        elif json_data["message_type"] == "register":
            self.server_host = json_data["server_host"]
            self.server_port = json_data["server_port"]
            networking.tcp_client(
                self.server_host,
                self.server_port,
                {
                    "message_type": "register_ack"
                }
            )
            # CHANGES MADE
            # Now we extract our new dedicated audio streaming port and spin up a constant listener.
            audio_port = json_data.get("audio_port")
            if audio_port is not None:
                threading.Thread(
                    target=self.run_audio_receiver,
                    args=(self.server_host, audio_port),
                    daemon=True
                ).start()

        elif json_data["message_type"] == "generated_audio":
            # CHANGES MADE
            # Old code processed base64 encoded audio files through the control channel JSON:
            # audio_data = json_data["audio_data"]
            # self.playback_queue.append(audio_data)
            pass

    def run_audio_receiver(self, host, port):
        """Continuously receive raw float32 audio chunks from the server."""
        while not self.signals["shutdown"]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((host, port))
                    LOGGER.info("Connected to audio streaming port %s:%s", host, port)
                    while not self.signals["shutdown"]:
                        # read chunk exactly matching the server (1024 frames of 2-channel float32 = 8192 bytes)
                        data = sock.recv(8192)
                        if not data:
                            LOGGER.warning("Audio stream disconnected")
                            break
                        # Queue the raw audio dynamically into our new PlaybackQueue
                        self.playback_queue.append_raw(data)
            except ConnectionRefusedError:
                sleep(1)
            except Exception as e:
                LOGGER.error("Audio receiver error: %s", e)
                sleep(1)

    def send_recipe_to_server(self, recipe):
        # NOTE: this does not incorperate a delay or smoothing
        #       for sequential request attempts
        #       (just sends lots of recipe updates
        #       to server while knob is being turned)
        try:
            networking.tcp_client(
                self.server_host,
                self.server_port,
                {
                    "message_type": "generate_audio",
                    "request_id": f"recipe_{int(time.time() * 1000)}",
                    "recipe": recipe,
                }
            )
        except ConnectionRefusedError:
            # Server may not be connected yet idk.
            pass

        # NOTE: currently this just clears the entire playback queue
        # TODO: more elegant approach that keeps portion of original
        #       queue so that the server's parallel music generations
        #       based on different recipe possibilities can be switched
        #       to seemlessly
        # NOTE: there are some things with race conditions / unpredictable
        #       ordering of segments when things are being added/deleted at
        #       the same time; supposedly collections.deque is thread-safe
        self.playback_queue.refresh()


    def run_playback(self):
        # CHANGES MADE
        # Old Stop-and-Wait Architecture:
        # while not self.signals["shutdown"]:
        #     if len(self.playback_queue) > 0:
        #         audio = self.playback_queue.pop()
        #         sd.play(
        #             data=audio["audio"],
        #             samplerate=audio["sr"],
        #             device=self.output_device,
        #             blocking=True,
        #         )
        #     else:
        #         # avoid busy waiting
        #         sleep(0.5)

        # Replace simple sd.play() with an asynchronous sd.OutputStream
        # Note: the mock server currently returns MP3s with 48000hz depending on the file
        samplerate = 44100
        channels = 2

        def callback(outdata, frames, time_info, status):
            if status:
                print("OutputStream status:", status)
            
            # Initialize with silence
            outdata.fill(0)
            
            # Pull exactly enough frames from the queue to fill the output buffer
            with self.lock:
                frames_to_fill = frames
                filled_frames = 0
                
                while frames_to_fill > 0 and len(self.playback_queue.deque) > 0:
                    chunk = self.playback_queue.pop(num_frames=frames_to_fill)
                    if not chunk:
                        break
                        
                    audio_data = chunk["audio"]
                    n_frames = len(audio_data)
                        
                    # Write into the output buffer
                    outdata[filled_frames:filled_frames+n_frames] = audio_data
                    filled_frames += n_frames
                    frames_to_fill -= n_frames

        try:
            with sd.OutputStream(
                samplerate=samplerate,
                channels=channels,
                device=self.output_device,
                callback=callback,
                blocksize=1024
            ):
                while not self.signals["shutdown"]:
                    # Keep thread alive while the stream runs in the background
                    sleep(0.5)
        except Exception as e:
            print(f"Failed to open audio stream: {e}")


    def run_webcam(self):
        gesture_name = None
        middle_finger_x = None
        middle_finger_y = None

        print("Press Q to exit")
        while not self.signals["shutdown"]:
            success, webcam_frame = self.webcam.read()
            key = cv2.waitKey(1) & 0xFF

            height, width, n_channels = webcam_frame.shape

            handedness, hand_landmarks = self.landmarker(webcam_frame)
            overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")

            isolated_hand = "Left"

            if len(hand_landmarks) > 0:
                hand_tensor = self.gesture_recognition.mediapipe_to_tensor(handedness, hand_landmarks, isolated_hand)
                hand_tensor = self.gesture_recognition.expand_one_hand_to_two_hands(hand_tensor, isolated_hand)
                gesture_name, confidence = self.gesture_recognition(hand_tensor, isolated_hand)

                # extract middle finger tip
                left_hand_landmarks = hand_landmarks[0]
                middle_finger = left_hand_landmarks[12]
                # Convert normalized coordinates (0-1) to pixel coordinates
                middle_finger_x = int((1 - middle_finger.x) * width)  # flip x for mirrored display
                middle_finger_y = int(middle_finger.y * height)

                self.recipe_interface.update_positions(
                    pointer_x=middle_finger_x,
                    pointer_y=middle_finger_y,
                    gesture=gesture_name
                )

            ####################################
            # Drawing onto webcam

            self.landmarker.draw_overlay_hands(
                webcam_frame,
                overlay_mask,
                text_lr = (f"{gesture_name} ({confidence:.1f}%)", "") if gesture_name is not None else ("no label", "")
            )

            self.recipe_interface.draw_bars(
                webcam_frame,
                overlay_mask
            )


            # ...
            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_frame = cv2.add(webcam_frame, overlay_mask)
            # draw to screen
            cv2.imshow("SuperConductor - Webcam View (Mediapipe)", webcam_frame)

            ####################################



            if key == ord("q"):
                break

# Configure logging
logging.basicConfig(filename="superconductor.log", level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=6000)
@click.option("--logfile", "logfile", default=None)
@click.option("--loglevel", "loglevel", default="info")
def main(host, port, logfile, loglevel):
    """Run Manager."""
    LOGGER.debug("____________________")
    LOGGER.debug("%s %s", host, port)

    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"Laptop:{port} [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(loglevel.upper())

    Frontend(host, port, prompts = ["Rock", "Guitar", "Jazz"])



if __name__ == "__main__":
    main()
