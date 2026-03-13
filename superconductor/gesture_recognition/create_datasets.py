#!/usr/bin/env python

from custom_dataset import CustomDatasetCreator, CustomDatasetProcessor


#def create_datasets():
    #data_creator = CustomDatasetCreator(dataset_name="palm_up_down")
    #data_creator(gesture_name = "palm_up")
    #data_creator(gesture_name = "palm_down")

    #data_creator = CustomDatasetCreator(dataset_name="palm_hold_release")
    #data_creator(gesture_name = "palm_hold")
    #data_creator(gesture_name = "palm_release")

    #processor = CustomDatasetProcessor(dataset_name = "palm_up_down")
    #processor = CustomDatasetProcessor(dataset_name = "palm_hold_release")


if __name__ == "__main__":
    data_creator = CustomDatasetCreator(dataset_name="palm_hold_release")
    data_creator(gesture_name = "palm_hold")
    data_creator(gesture_name = "palm_release_up")
    data_creator(gesture_name = "palm_release_down")
    processor = CustomDatasetProcessor(dataset_name = "palm_hold_release")
