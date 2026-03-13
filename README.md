# SuperConductor Frontend

![diagram](media/diagram.jpeg)

## Start SuperConductor
```bash
# first activate conda environment:
conda activate sc_env

# Two seperate processes
# sc-laptop: laptop.py
# sc-server: mock_local_server.py

# this calls superconductor/launcher.py to run
# bash commands for start|stop|status|restart
superconductor --start

# this sends a shutdown request to sc-laptop,
# which then shuts down sc-server 
# (can be modified to keep server running)
superconductor --stop
```

Whenever recipe is updated via on-screen slider controls, a request is made to the mock backend running locally, which sends a sequence of mp3 files (5 segments of the beginning of No Idea - Don Toliver). These are loaded into a playback queue by the laptop code and played sequentially to a specified audio device.

## Setup

```bash
conda env create -f environment.yml
conda activate sc_env
python -m pip install -e . --no-deps
# chmod +x bin/superconductor
```

# Other Commands

(see `[project.scripts]` in `./pyproject.toml`)

## Set Audio Device
```bash
sc-audio-device --list
# [1] USB-C to 3.5mm Headphone Jack Adapter (out=2)
# [3] BlackHole 2ch (out=2)
# [5] MacBook Pro Speakers (out=2)
# [6] Microsoft Teams Audio (out=1)
# [7] Steam Streaming Microphone (out=2)
# [8] Steam Streaming Speakers (out=2)
# [9] ZoomAudioDevice (out=2)

sc-audio-device --select 1
# selected [1] USB-C to 3.5mm Headphone Jack Adapter (out=2)
```

## Directory Structure

```bash
tree
├── README.md
├── bin
│   └── superconductor
├── hand_landmarker.task
├── media
│   ├── diagram.jpeg
│   └── mediapipe_quickstart_guide.py
├── pyproject.toml
├── superconductor
│   ├── gesture_recognition
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── create_datasets.py
│   │   ├── custom_dataset.py
│   │   ├── dataloader.py
│   │   ├── model.py
│   │   ├── palm_hold_release
│   │   │   └── label_map.csv
│   │   ├── palm_hold_release_model.pth
│   │   └── train.py
│   ├── laptop.py
│   ├── launcher.py
│   ├── mock_local_server.py
│   ├── mock_server_audio
│   │   ├── audio_01.mp3
│   │   ├── audio_02.mp3
│   │   ├── audio_03.mp3
│   │   ├── audio_04.mp3
│   │   ├── audio_05.mp3
│   │   └── noidea_dontoliver.mp3
│   ├── networking.py
│   └── recipe_interface
│       ├── __init__.py
│       ├── __main__.py
│       └── __pycache__
│           ├── __init__.cpython-312.pyc
│           └── __main__.cpython-312.pyc
├── superconductor.log
└── var
    └── log
        ├── laptop.log
        └── server.log
```