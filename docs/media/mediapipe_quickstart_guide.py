import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

# pip install mediapipe opencv-python

def main():
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

    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    frame_counter = 0
    cap = cv2.VideoCapture(0)
    print()
    print("Press Q to exit\nnote that (0,0) is top left corner of image")
    print()
    while True:
        success, img = cap.read()
        key = cv2.waitKey(1) & 0xFF

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, n_channels = img.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        # retrieve landmark coordinates and handedness from camera input
        results = hand_landmarker.detect(mp_image)

        hand_landmarks_list = results.hand_landmarks
        handedness_list = results.handedness

        # cv2.putText doesn't seem to allow horizonally flipping text,
        # so this draws text onto an unflipped overlay which is then
        # added to the flipped image prior to calling cv2.imshow()
        text_mask = np.zeros((height, width, n_channels), dtype="uint8")

        for idx in range(len(results.hand_landmarks)):
            print("===================")
            print("===================")
            print("===================")
            #for coords, hand in zip(results.handedness, results.hand_landmarks):
                #print(coords)
                #print("===================")
                #print(hand)
                #print("===================")
                #print("===================")
            print("===================")
            print("===================")
            #exit()
            # Retrieve the wrist coordinate:
            wrist_coordinates = hand_landmarks_list[idx][0]
            wrist_x = wrist_coordinates.x
            wrist_y = wrist_coordinates.y
            #wrist_z = wrist_coordinates.z
            wrist_handedness = handedness_list[idx][0].category_name


            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # =====================================================================
            # Draw handedness (left or right hand) on the image.
            # =====================================================================
            # Draw text at the top left corner of the detected hand's bounding box.
            #x_coordinates = [landmark.x for landmark in hand_landmarks]
            #y_coordinates = [landmark.y for landmark in hand_landmarks]
            #text_x = width - int(max(x_coordinates) * width)
            #text_y = int(min(y_coordinates) * height)
            #hand_label = f"{handedness[0].category_name}"

            # Draw text at the wrist coordinate (index 0)
            hand_label = wrist_handedness
            text_x = width - int(wrist_x * width)
            text_y = int(wrist_y * height)

            # =====================================================================
            font_scale = 1  # negative value to flip text vertically
            font_thickness = 1
            font_color = (88, 205, 54) # vibrant green

            cv2.putText(text_mask, hand_label,
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

            # =====================================================================
            # =====================================================================


            # Draw the hand landmarks.
            vision.drawing_utils.draw_landmarks(
                img,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.get_default_hand_landmarks_style(),
                vision.drawing_styles.get_default_hand_connections_style())

        #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

        img = cv2.flip(img, 1)
        img = cv2.add(img, text_mask)
        # draw to screen
        cv2.imshow("Image", img)

        if frame_counter % 25 == 0 and len(hand_landmarks_list) != 0:
            output_str = ""
            # note: output is horizontally flipped, so x=1-x, y=y
            #       is to keep top left as the origin (0,0)
            # 
            #       (1-x)*width, (y)*height is pixel coordinate
            for hand, coordinate in zip(handedness_list, hand_landmarks_list):
                output_str += f"{hand[0].category_name}: x={1-coordinate[0].x}, y={coordinate[0].y}, "

            print(f"wrist coordinates: {output_str}")
            frame_counter = 0

        frame_counter += 1

        if key == ord("q"):
            break


if __name__ == "__main__":
    main()