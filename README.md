# SuperConductor Frontend

![diagram](media/diagram.jpeg)

Start SuperConductor
--------------------

The system consists of three parts:

1.  **Server (Lighthouse GPU)**
    
2.  **SSH tunnel (connect server → local)**
    
3.  **Client (local laptop)**
    

### 1\. Start the Server (Lighthouse)

Log in to your Lighthouse account. The model is located at:

`/scratch/aimusic_project_root/aimusic_project/shared_data/magenta_native`

#### Step 1: Allocate GPU

`salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=64G --cpus-per-task=4 --time=00:15:00   `

*   Adjust --time based on how long you need the server
    
*   The job will stop automatically after the time expires
    
*   You can also run exit to release resources early
    

#### Step 2: Activate environment

`source magenta-realtime/.venv/bin/activate   `

#### Step 3: Run the server

`   python -m magenta_rt.server --tag large --device gpu --port 8000   `

*   Use Ctrl + C to stop the server manually
    

### 2\. Start SSH Tunnel (on your laptop)

Open a **new local terminal** and run:

`   ssh -N -L 9000:lh2300:8000 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu   `

This forwards:

`   localhost:9000 → lighthouse:8000   `

### 3\. Start the Client (local)

In your local project:

`   conda activate sc_env `
`python3 superconductor/laptop.py   `

*   A window will open with the camera + UI
    
*   **Press q (while the window is focused) to exit**

  

Current state of music playback:
```
The server pre-loads the MP3s into RAM as raw float32 numpy arrays and continuously streams small 1024-frame chunks over a new socket. 
The frontend has am audio stream that pulls from the buffer and  receivies these raw chunks as they arrive.
When a user gestures to change the recipe, the server instantly swaps the memory pointer for the stream track, so there might not be any gap
have to see if it still works when we integrate magenta in the backend
```

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
