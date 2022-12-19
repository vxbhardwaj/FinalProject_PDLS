# FinalProject_PDLS

# A description of the project:

- In this project, we are predicting pneumonia using chest X-RAY images by predicting bounding boxes around areas of the lungs. 
- We have done a comparative analysis of various Image object detection models on the basis of their MAP (mean average precision) scores, associated costs involved, and   the amount of training times
- Our objective is to experiment with AutoML and different pretrained models (Masked-RCNN and YOLO on V100 and T4 GPUs) to find the best performing one which is cost-effective and takes less human effort to build. 

# A description of the repository:
AutomL folder contains: 
Files:
- create_test_jsonl.py to create jsonl files for test data
- create_submission_csv.py to convert automl predictions to kaggle's submission format
- create_annotation_csv.py to create annotation for the images.

Folder: 
- results: This folder contains automl predictions related CSV and jsonl files.

dicom_to_png folder contains: files related to converting DCM X ray images to PNG format

masked-rcnn - Files related to mask-RCNN

yolo - Files related to YOLO v3:

T4 final weights: https://storage.googleapis.com/projsub/T4/rsna_yolov3_final.weights

V100 final weights: https://storage.googleapis.com/projsub/V100/rsna_yolov3_final.weights

# Example commands to execute the code:

Jupyter notebooks can be run as is, with relevant files provided along side


# Results (including charts/tables) and your observations:

![image](https://user-images.githubusercontent.com/76259177/208469194-97ee416d-730b-4b65-bc11-797c3b7871fc.png)

![image](https://user-images.githubusercontent.com/76259177/208469300-497ad05c-c3ff-47f5-852d-fc39ab0195be.png)

![image](https://user-images.githubusercontent.com/76259177/208469359-7782454c-a5d0-4021-ad99-5dd47f836590.png)

![image](https://user-images.githubusercontent.com/76259177/208469420-12a93ba9-0d66-417e-867b-6cc56012fcee.png)

![image](https://user-images.githubusercontent.com/76259177/208469461-5bc7a9bb-22b2-4d3e-82c2-88a907ea9787.png)

![image](https://user-images.githubusercontent.com/76259177/208469510-25154e90-0afd-4969-b8a7-df5d81698f91.png)




