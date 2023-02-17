# COCO-scripts

 - bboxes_to_centerpoints_jittered.py
   - Purpose: Create centerpoints from bounding boxes in a coco file and shift a specified percentage of them in a random direction. The amount shifted will be given in meters at the command line, and could be between 1 pixel and the number of pixels consituting that number of meters on a given image. All points will be shifted a uniform amount and direction on a per-image basis. 
   - Arguments:
    - train_fp: File path to geococo train annotations
    - shift_meters: Int, the number of meters you would like annotations to be shifted, on an image-by-image basis, in meters
    - shift_percent Int, 0-100, The percentage of images you would like to shift by the value in shift_meters (not required, default 100)
    - avg_gsd: Average image GSD you would like to use (not required)
  - Sample call: "python3 bboxes_to_centerpoints_jittered.py -train_fp DOTA_test.json -shift_meters 10 -shift_percent 100"
