# FinalProject_PDLS

# A description of the project:

- In this project, we are predicting pneumonia using chest X-RAY images by predicting bounding boxes around areas of the lungs. 
- We have done a comparative analysis of various Image object detection models on the basis of their MAP (mean average precision) scores, associated costs involved, and   the amount of training times
- Our objective is to experiment with AutoML and different pretrained models (Masked-RCNN and YOLO on V100 and T4 GPUs) to find the best performing one which is cost-effective and takes less human effort to build. 

#A description of the repository:
AutomL folder contains: 
Files:
- create_test_jsonl.py to create jsonl files for test data
- create_submission_csv.py to convert automl predictions to kaggle's submission format
- create_annotation_csv.py to create annotation for the images.

Folder: 
- results: This folder contains automl predictions related CSV and jsonl files.

dicom_to_png folder contains: files related to converting DCM X ray images to PNG format

masked-rcnn - Files related to mask-RCNN

yolo - Files related to YOLO v3

Example commands to execute the code:

Jupyter notebooks can be run as is, with relevant files provided along side


Results (including charts/tables) and your observations:
