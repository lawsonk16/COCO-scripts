# COCO-scripts

## bboxes_to_centerpoints_human_error
purpose: Simulate the centerpoints a human annotator might give, using image oriented bounding box data, and use them to create an imputed square bounding box as might be used for training on such data. 
description: This script takes an existing dataset and calculates single point labels for each bounding box in the dataset. Some given n value is passed in and each point is jittered in a random direction somewhere between 0-n pixels to simulate the differences between where a human annotator would place such a point and the mathematical center of the box. Using GSD values in the dataset, the average size of each object class in the dataset is calculated using the original bounding boxes, and then the imputed centerpoints and GSD values of each image are used to create a square bounding box centered on those single point labels of the average size of that object category.
 - Arguments:
  - train_fp: str, File path to geococo train annotations
  - val_fp: str, File path to geococo train annotations
  - avg_gsd: float, Average image GSD you would like to use where an image doesn't have one (not required)
- Sample call: "python3 bboxes_to_centerpoints_human_error.py -train_fp DOTA_test.json -val_fp DOTA_val.json -avg_gsd 0.5

## bboxes_to_centerpoints_geo_error
purpose: create experimental simulation of what might happen when converting geospatial coordinate labels to pixel coordinate labels
description: Create centerpoints from bounding boxes in a coco file and shift a specified percentage of them in a random direction. The amount shifted will be given in meters at the command line, and could be between 1 pixel and the number of pixels consituting that number of meters on a given image. All points will be shifted a uniform amount and direction on a per-image basis. 
 - Arguments:
  - train_fp: str, File path to geococo train annotations
  - shift_meters: int, the number of meters you would like annotations to be shifted, on an image-by-image basis, in meters
  - shift_percent int, 0-100, The percentage of images you would like to shift by the value in shift_meters (not required, default 100)
  - avg_gsd: float, Average image GSD you would like to use where an image doesn't have one (not required)
- Sample call: "python3 bboxes_to_centerpoints_geo_error.py -train_fp DOTA_test.json -shift_meters 10 -shift_percent 100"

## full_scene_vs_single_class
purpose: determine model performance differences between a dataset labeled using full scene labels and a dataset created labeling only a single category of interest
description: this script takes in a coco dataset and produces two outputs - 1) one dataset with only one category remaining and 2) another including only images from dataset 1 but with every category included such that the two datasets have roughly the same number of total annotations. This is accomplished by randomly selecting images and their annotations to create the full scene dataset until there are an equal number or more labels in the full scene dataset. 
- Arguments:
 - cat_id: int, COCO category id of the class you would like to focus on for this experiment
 - ann_fp: str, File path to coco annotations
 - img_fp: str, File path to images for the annotations
- Sample call: python3 full_scene_vs_single_class.py -cat_id 1 -ann_fp /content/FAIR1M-1p-COCO/train-COCO.json -img_fp /content/FAIR1M-1p-COCO/images/
