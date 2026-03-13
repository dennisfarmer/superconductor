from pathlib import Path
#from dataloader import ImageToTensorPreprocessor
import pandas as pd
import cv2
import numpy as np
from random import random
from pathlib import Path
from shutil import copy
from pathlib import Path
import pandas as pd
import torch
import shutil

import time
import sys
sys.path.insert(0, '..')
import laptop

from collections import defaultdict


class CustomDatasetCreator:
    def __init__(self, dataset_name="custom_dataset"):
        self.landmarker = laptop.MediaPipeLandmarker()
        self.gesture_detection = laptop.GestureRecognition()
        self.webcam = cv2.VideoCapture(0)
        self.dataset_name = dataset_name

        self.src_directory = Path(dataset_name)
        self.src_directory.mkdir(parents=True, exist_ok=True)
        
        self.label_map_path = self.src_directory / "label_map.csv"

        if not self.label_map_path.exists():
            empty_label_map = {"gesture_name": [], "label": []}
            pd.DataFrame(empty_label_map).to_csv(self.label_map_path, index=False)

        label_map = pd.read_csv(self.label_map_path)
        if not label_map.empty:
            label_map = label_map[
                ~label_map["gesture_name"].astype(str).str.match(r"^gesture_\d+$")
            ]
            label_map = label_map.reset_index(drop=True)
            label_map["label"] = range(len(label_map))
            label_map.to_csv(self.label_map_path, index=False)
        self.label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        self.name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))


    def write_label_map(self):
        output_dict = defaultdict(list)
        for name,label in self.name_to_label.items():
            output_dict["gesture_name"].append(name)
            output_dict["label"].append(label)
        pd.DataFrame(output_dict).to_csv(self.label_map_path)

    def __call__(self, gesture_name="palm_up"):
        
        gesture_dir = self.src_directory / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        # capture every two seconds
        t = 0.5

        idx = 0
        records = []

        
        if gesture_name not in self.name_to_label:
            if self.name_to_label:
                label = max(self.name_to_label.values()) + 1
            else:
                label = 0
            self.name_to_label[gesture_name] = label
            self.write_label_map()
        else:
            label = self.name_to_label[gesture_name]

        csv_path = self.src_directory / f"{label}_{gesture_name}_records.csv"
        if csv_path.exists():
            try:
                existing_records = pd.read_csv(csv_path)
                idx = existing_records["index"].max() + 1
            except:
                idx = 0

        
        last_capture_time = time.time()
        
        has_pressed_start = False
        print("press W to start")

        print(f"{self.dataset_name} / {gesture_name}")
        while True:
            ret, webcam_frame = self.webcam.read()
            if not ret:
                break

            height, width, n_channels = webcam_frame.shape
            handedness, hand_landmarks = self.landmarker(webcam_frame)
            overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")
            self.landmarker.draw_overlay_hands(webcam_frame, overlay_mask)
            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_frame = cv2.add(webcam_frame, overlay_mask)


            if cv2.waitKey(1) & 0xFF == ord('w'):
                has_pressed_start = True

            if not has_pressed_start:
                cv2.imshow(f"Press 'W' to start - {self.dataset_name} / {gesture_name}", webcam_frame)
                continue
            else:
                cv2.imshow(f"Press 'Q' to exit - {self.dataset_name} / {gesture_name}", webcam_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            current_time = time.time()
            if current_time - last_capture_time >= t:
                
                if hand_landmarks is not None and len(hand_landmarks) > 0:
                    tensor = self.gesture_detection.mediapipe_to_tensor(handedness, hand_landmarks)
                    
                    tensor_path = gesture_dir / f"{idx}.pt"
                    torch.save(tensor, tensor_path)
                    
                    records.append({"index": idx, "label": label, "gesture_name": gesture_name})
                    idx += 1
                    print(f"captured sample with idx={idx}")
                
                last_capture_time = current_time
            
        
        cv2.destroyAllWindows()
        
        if records:
            df = pd.DataFrame(records)
            if csv_path.exists():
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
        print(f"data added for {gesture_name}: {len(records)} records")
        print("all done")



class CustomDatasetProcessor:
    def __init__(self, dataset_name="custom_dataset", train_val_split=0.8) -> None:
        self.train_val_split = train_val_split

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.src_directory = Path(dataset_name)
        self.label_map_path = self.src_directory / "label_map.csv"
        
        
        self.label_map = pd.read_csv(self.label_map_path)
        self.label_to_name = dict(zip(self.label_map["label"].astype(int), self.label_map["gesture_name"]))
        self.name_to_label = dict(zip(self.label_map["gesture_name"], self.label_map["label"].astype(int)))

        self.num_classes = len(self.label_to_name)

        self.src_train_val_dir = self.src_directory / "train"

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"
        if self.tgt_directory.exists():
            shutil.rmtree(self.tgt_directory)

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)

        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.dataset = {"index": [], "partition": [], "label": [], "position": []}

        self._process_dataset()
        print(f'processed dataset {dataset_name}!')

        df = pd.DataFrame(self.dataset)
        df.to_csv(self.tgt_directory / "gestures.csv", index=False)
    
    def _add_record(self, index: int, partition: str, label: int, position: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["position"].append(position)

    def _process_dataset(self):
        for _, row in self.label_map.iterrows():
            gesture_name = row["gesture_name"]
            label = int(row["label"])
            
            gesture_dir = self.src_directory / gesture_name
            if not gesture_dir.exists():
                continue
            
            for example in gesture_dir.iterdir():
                if example.suffix == ".pt":
                    random_number = random()
                    if random_number > self.train_val_split:
                        copy(Path.cwd() / example, self.tgt_val / f"{self.val_index}.pt")
                        self._add_record(self.val_index, "val", label, gesture_name)
                        self.val_index += 1
                    else:
                        copy(Path.cwd() / example, self.tgt_train / f"{self.train_index}.pt")
                        self._add_record(self.train_index, "train", label, gesture_name)
                        self.train_index += 1
            
