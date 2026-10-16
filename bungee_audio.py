"""Native Bungee Basic stretching with fixed-rate stereo PCM output."""
import ctypes
from pathlib import Path
import numpy as np

SAMPLE_RATE = 48000
INPUT_FRAMES = 256


class BungeeStream:
    def __init__(self, samples):
        root = Path(__file__).parent / '.build' / 'native'
        library = root / 'libconductor_bungee.so'
        if not library.exists():
            library = root / 'libconductor_bungee.dylib'
        if not library.exists():
            raise RuntimeError('Build Bungee first: bash build_bungee.sh (from the application folder)')
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self.samples.ndim != 2 or self.samples.shape[1] != 2 or not len(self.samples):
            raise ValueError('Expected nonempty stereo audio')
        if not np.isfinite(self.samples).all():
            raise ValueError('Audio contains non-finite samples')
        self.library = ctypes.CDLL(str(library))
        self.library.conductor_create.restype = ctypes.c_void_p
        self.library.conductor_destroy.argtypes = [ctypes.c_void_p]
        self.library.conductor_destroy.restype = None
        self.library.conductor_error.restype = ctypes.c_char_p
        self.library.conductor_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_double]
        self.library.conductor_process.restype = ctypes.c_int
        self.handle = self.library.conductor_create()
        if not self.handle:
            raise RuntimeError(self.library.conductor_error().decode())
        self.position = 0
        self.indices = np.arange(INPUT_FRAMES)
        self.output = np.empty((5121, 2), dtype=np.float32)

    def render(self, speed):
        if not self.handle:
            raise RuntimeError('Bungee stream is closed')
        if not np.isfinite(speed) or not 0.05 <= speed <= 20:
            raise ValueError('Tempo ratio must be between 0.05 and 20')
        # Wrap input, not the stretcher: history survives the end of the track.
        block = self.samples[(self.indices + self.position) % len(self.samples)]
        count = self.library.conductor_process(self.handle, block.ctypes.data,
                                               self.output.ctypes.data, speed)
        if count < 0:
            raise RuntimeError(self.library.conductor_error().decode())
        self.position = (self.position + INPUT_FRAMES) % len(self.samples)
        return self.output[:count]

    def close(self):
        if self.handle:
            self.library.conductor_destroy(self.handle)
            self.handle = None
