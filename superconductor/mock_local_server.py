"""Mock audio synthesis backend.

This backend registers itself with the laptop process, waits for a
``register_ack`` message, then sends a fixed sequence of mock MP3 files.
"""
import base64
import json
import logging
from pathlib import Path
import socket
import threading
import time

import click

# Configure logging
LOGGER = logging.getLogger(__name__)

# based on EECS 485 project 4 - map reduce


class MockBackend:
    """Mock backend that pushes a fixed set of MP3 files to the laptop."""

    def __init__(self, host, port, laptop_host, laptop_port):
        """Initialize backend server, register, and stream mock audio files."""
        LOGGER.info(
            "Starting MockBackend on %s:%s (laptop=%s:%s)",
            host,
            port,
            laptop_host,
            laptop_port,
        )
        self.host = host
        self.port = port
        self.laptop_host = laptop_host
        self.laptop_port = laptop_port
        self.shutdown_flag = False
        self.registered_event = threading.Event()
        self.audio_dir = Path(__file__).resolve().parent / "mock_server_audio"
        self.audio_filenames = [f"audio_{idx:02d}.mp3" for idx in range(1, 6)]

        # Start backend listener (for register_ack/shutdown) in a thread.
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = False
        self.thread.start()

        # Register with laptop and wait for acknowledgment.
        self._register_with_laptop()
        if not self.registered_event.wait(timeout=10):
            LOGGER.error("Did not receive register_ack from laptop in time")
            self.shutdown()
            self.thread.join()
            return

        self._send_audio_sequence()
        # Keep running to accept replay/shutdown requests.
        self.thread.join()

    def _run_server(self):
        """Run TCP server to receive control messages from the laptop."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(5)
            sock.settimeout(1)
            LOGGER.info("Backend server listening on %s:%s", self.host, self.port)

            while not self.shutdown_flag:
                try:
                    client_socket, client_addr = sock.accept()
                    LOGGER.info("Control connection from %s", client_addr)
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    LOGGER.error("Server error: %s", e)

    def _handle_client(self, client_socket):
        """Handle a control connection from the laptop."""
        try:
            with client_socket:
                client_socket.settimeout(1)
                message_chunks = []
                while True:
                    try:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        message_chunks.append(data)
                    except socket.timeout:
                        continue

                if message_chunks:
                    message_bytes = b''.join(message_chunks)
                    message_str = message_bytes.decode("utf-8")

                    try:
                        request = json.loads(message_str)
                        LOGGER.info("Received request: %s", request)
                        message_type = request.get("message_type")

                        if message_type == "register_ack":
                            self.registered_event.set()
                            LOGGER.info("Registered with laptop successfully")
                        elif message_type == "generate_audio":
                            LOGGER.info("Received recipe update; sending mock audio sequence")
                            self._send_audio_sequence()
                        elif message_type == "shutdown":
                            self.shutdown_flag = True
                    except json.JSONDecodeError as e:
                        LOGGER.error("JSON decode error: %s", e)
        except Exception as e:
            LOGGER.error("Client handling error: %s", e)

    def _register_with_laptop(self):
        """Register this backend with the running laptop process."""
        payload = {
            "message_type": "register",
            "server_host": self.host,
            "server_port": self.port,
        }

        # Retry briefly to handle startup races between laptop and backend.
        for attempt in range(1, 11):
            if self.shutdown_flag:
                return
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(2)
                    sock.connect((self.laptop_host, self.laptop_port))
                    sock.sendall(json.dumps(payload).encode("utf-8"))
                LOGGER.info("Sent register message (attempt %d)", attempt)
                return
            except OSError as e:
                LOGGER.warning(
                    "Failed to register with laptop (attempt %d): %s",
                    attempt,
                    e,
                )
                time.sleep(0.5)

        LOGGER.error("Unable to register with laptop at %s:%s", self.laptop_host, self.laptop_port)

    def _send_audio_sequence(self):
        """Send the five mock MP3 files to the laptop in fixed order."""
        for idx, audio_name in enumerate(self.audio_filenames, start=1):
            if self.shutdown_flag:
                return

            audio_path = self.audio_dir / audio_name
            if not audio_path.exists():
                LOGGER.error("Missing mock audio file: %s", audio_path)
                continue

            mp3_data = audio_path.read_bytes()
            request_id = f"mock_audio_{idx:02d}"
            self._send_generated_audio(mp3_data, request_id)
            time.sleep(0.25)

    def _send_generated_audio(self, mp3_data, request_id):
        """Send a generated_audio message to the laptop."""
        response = {
            "message_type": "generated_audio",
            "request_id": request_id,
            "status": "success",
            "audio_size": len(mp3_data),
            "audio_data": base64.b64encode(mp3_data).decode("utf-8"),
        }

        response_json = json.dumps(response)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                sock.connect((self.laptop_host, self.laptop_port))
                sock.sendall(response_json.encode("utf-8"))
            LOGGER.info("Sent %s (size: %d bytes)", request_id, len(mp3_data))
        except Exception as e:
            LOGGER.error("Failed to send audio response: %s", e)

    def _send_error_response(self, client_socket, request_id, error_message):
        """Send an error response to the client."""
        response = {
            "message_type": "audio_response",
            "request_id": request_id,
            "status": "error",
            "error": error_message,
        }
        response_json = json.dumps(response)
        try:
            client_socket.sendall(response_json.encode("utf-8"))
        except Exception as e:
            LOGGER.error("Failed to send error response: %s", e)

    def _create_mock_mp3(self, recipe):
        """Create a minimal mock MP3 file."""
        # ID3v2 tag header
        id3_header = b'ID3'
        id3_version = b'\x03\x00'  # v2.3.0
        id3_flags = b'\x00'
        id3_size = b'\x00\x00\x00\x00'

        # Create simple MP3 frames from recipe
        # Use different frequencies for each instrument
        mp3_frames = b''
        instruments = list(recipe.keys())

        # Generate frame data based on recipe (mock implementation)
        for instrument in instruments:
            # Add a simple MP3 frame header (0xFFF sync word + frame header)
            mp3_frames += b'\xFF\xFB\x10\x00'  # Frame sync + minimal MPEG1 Layer3 header

        # Pad with some silence data
        mp3_frames += b'\x00' * 1024

        return id3_header + id3_version + id3_flags + id3_size + mp3_frames

    def shutdown(self):
        """Shutdown the backend."""
        self.shutdown_flag = True
        LOGGER.info("Backend shutting down")


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=6001)
@click.option("--laptop-host", "laptop_host", default="localhost")
@click.option("--laptop-port", "laptop_port", default=6000)
@click.option("--logfile", "logfile", default=None)
@click.option("--loglevel", "loglevel", default="info")
def main(host, port, laptop_host, laptop_port, logfile, loglevel):
    """Run the mock audio synthesis backend."""
    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(f"Backend:{port} [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(loglevel.upper())
    MockBackend(host, port, laptop_host, laptop_port)


if __name__ == "__main__":
    main()
