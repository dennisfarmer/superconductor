"""Gesture Recognition for parameter controls."""
from pathlib import Path
import pandas as pd
import torch

from .model import PalmModel

class GestureRecognition:
    def __init__(self, model_name = "palm_hold_release"):
        """
        `model_name` can be either `"palm_up_down"` or `"palm_hold_release"`
        """
        self.model_name = model_name
        self.model_initialized = False
        self.device = None

    def initialize_model(self):
        self.src_directory = Path(__file__).parent / self.model_name
        label_map_path = self.src_directory / "label_map.csv"
        label_map = pd.read_csv(label_map_path)
        self.label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))
        self.num_classes = label_map["label"].nunique()

        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")

        # MPS: Apple Silicon
        elif torch.backends.mps.is_available():
            print("Using MPS")
            self.device = torch.device("mps")

        # CPU: 
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        self.model = PalmModel(input_features = 63*2, num_classes=self.num_classes)
        model_path = self.src_directory.parent / f"{self.model_name}_model.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.model_initialized = True

    def __call__(self, tensor, isolated_hand="Left"):
        if not self.model_initialized:
            return None, None

        tensor = tensor.to(self.device)
        classification = self.model(tensor)
        predicted_label = torch.argmax(classification, dim=-1).item()
        predicted_gesture = self.label_to_name.get(predicted_label, f"Unknown ({predicted_label})")
        confidence = torch.softmax(classification, dim=-1).max().item() * 100

        return predicted_gesture, confidence

    def mediapipe_to_tensor(self, handedness, hand_landmarks, isolated_hand=None):
        """
        `isolated_hand=None|"Left"|"Right"`: flips to / keeps only specified hand if specified
        """
        hand_coords_dict = self.create_hands_dict(handedness, hand_landmarks)
        left_hand_coords = None
        right_hand_coords = None
        if "Right" in hand_coords_dict.keys():
            right_hand_coords = hand_coords_dict["Right"]
        if "Left" in hand_coords_dict.keys():
            left_hand_coords = hand_coords_dict["Left"]

        tensor = self.landmarks_to_tensor(left_hand_coords, right_hand_coords, isolated_hand)
        return self.normalize_to_wrist(tensor)

    def expand_one_hand_to_two_hands(self, tensor, isolated_hand="Left"):
        """
        given a tensor of a single hand, represent it as full tensor with zeros for missing hand
        """
        if isolated_hand == "Left":
            return torch.cat((tensor, torch.zeros(63)))
        elif isolated_hand == "Right":
            return torch.cat((torch.zeros(63), tensor))


    def landmarks_to_tensor(self, left_hand_coords: list[(int, float, float, float)] = None, right_hand_coords: list[(int, float, float, float)] = None, isolated_hand=None) -> torch.Tensor:
        """
        index 0-62: left hand coordinates (x0, y0, z0, x1, y1, z1, ..., )
        index 63-126: right hand coordinates (x21, y21, z21, x22, y22, z22, ...,)

        `isolated_hand=None|"Left"|"Right"`: flips to / keeps only specified hand if specified
        """
        output_tensor = torch.zeros(63*2, dtype=torch.float32)
        lh_coords = []
        if left_hand_coords is not None:
            for i,x,y,z in left_hand_coords:
                lh_coords.extend([x,y,z])
            output_tensor[:63] += torch.tensor(lh_coords, dtype=torch.float32)

        rh_coords = []
        if right_hand_coords is not None:
            for i,x,y,z in right_hand_coords:
                rh_coords.extend([x,y,z])
            output_tensor[63:] += torch.tensor(rh_coords, dtype=torch.float32)

        if isolated_hand is None:
            return output_tensor

        elif isolated_hand == "Left":
            if output_tensor[:63].sum() > 0:
                return output_tensor[:63]
            else:
                # flip right hand to be left, assumes that coordinates are normalized to 0-1
                return torch.tensor([0,1,0]).repeat(21) - output_tensor[63:]

        elif isolated_hand == "Right":
            if output_tensor[63:].sum() > 0:
                return output_tensor[63:]
            else:
                # flip left hand to right, assumes that coordinates are normalized to 0-1
                return torch.tensor([0,1,0]).repeat(21) - output_tensor[:63]
        else:
            return output_tensor


    def normalize_to_wrist(self, landmark_tensor):
        """
        works for both single hand and both hand tensors
        """
        # subtract wrist (x,y,z) from all coordinates except wrist
        # Left hand block: indices 0..62 (21 points * 3)
        if landmark_tensor.numel() >= 63:
            landmark_tensor[3:63] -= landmark_tensor[0:3].repeat(21 - 1)

        # Right hand block: indices 63..125 (21 points * 3)
        if landmark_tensor.numel() >= 126:
            landmark_tensor[66:126] -= landmark_tensor[63:66].repeat(21 - 1)

        return landmark_tensor

    def create_hands_dict(self, handedness, hand_landmarks):
        """
        Returns the following dict:
        ```
        {
            "Left": [(0, x,y,z), (1,x,y,z) ...],
            "Right": [(0,x,y,z), (1,x,y,z), ...]
        }
        ```
        """
        hands_dict = {}
        for hand_num in range(len(hand_landmarks)):
            current_hand_landmarks = hand_landmarks[hand_num]
            hand_name = handedness[hand_num][0].category_name
            coords = []
            for id, landmark in enumerate(current_hand_landmarks):
                center_x, center_y, center_z = float(landmark.x), float(landmark.y), float(landmark.z)
                coords.append([id, center_x, center_y, center_z])

            hands_dict[hand_name] = coords
        return hands_dict
