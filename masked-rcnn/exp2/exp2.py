#!/usr/bin/env python
# coding: utf-8

# Right now, this is substantively identical to version 8 of Henrique Mendoça's RCNN model with Coco transfer learning (see fork link at the top).  The first version, which is an exact copy, got a LB score of 0.162.  I've edited the documentaiton and added the plot code here from his later version.   This version probably does not get such a high score, just because there is a lot of randomness between one run and another.
# 
# **Mask-RCNN Starter Model for the RSNA Pneumonia Detection Challenge with transfer learning **
# 
# Using pre-trained COCO weights trained on http://cocodataset.org as in https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon
# We get the best public kernel performance so far, and also training only within the 6hrs kaggle limit.

# In[1]:


import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 


# In[2]:


DATA_DIR = '/home/jyothsnakodandera/FinalProject_PDLS/data/rsna-pneumonia-data'
ROOT_DIR = './'


# ### Install Matterport's Mask-RCNN model from github.
# See the [Matterport's implementation of Mask-RCNN](https://github.com/matterport/Mask_RCNN).

# In[3]:


# !git clone https://www.github.com/matterport/Mask_RCNN.git
# os.chdir('Mask_RCNN')
#!python setup.py -q install


# In[4]:


# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[5]:


train_dicom_dir = os.path.join(DATA_DIR, 'stage_2_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')


# In[6]:


### Download COCO pre-trained weights
# !wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
# !ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = "./pneumonia20221212T0116/mask_rcnn_pneumonia_0002.h5" #"mask_rcnn_coco.h5"


# ### Some setup functions and classes for Mask-RCNN
# 
# - dicom_fps is a list of the dicom image path and filenames 
# - image_annotions is a dictionary of the annotations keyed by the filenames
# - parsing the dataset returns a list of the image filenames and the annotations dictionary

# In[7]:


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 


# In[8]:


# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 1)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 200
    
config = DetectorConfig()
config.display()


# In[9]:


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


# ### Examine the annotation data, parse the dataset, and view dicom fields

# In[10]:


# training dataset
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train_labels.csv'))
anns.head()


# In[11]:


image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)


# In[12]:


ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
image = ds.pixel_array # get image array
# show dicom fields 
ds


# In[13]:


# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024


# ### Split the data into training and validation datasets

# In[14]:


# split dataset into training vs. validation dataset 
# split ratio is set to 0.95 vs. 0.05 (train vs. validation, respectively)
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)

val_size = int(0.05 * len(image_fps_list))
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print(len(image_fps_train), len(image_fps_val))


# ### Create and prepare the training dataset using the DetectorDataset class.

# In[15]:


# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


# In[16]:


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


# ### Display a random image with bounding boxes

# In[17]:


# Load and display random sample and their bounding boxes

# class_ids = [0]
# while class_ids[0] == 0:  ## look for a mask
#     image_id = random.choice(dataset_train.image_ids)
#     image_fp = dataset_train.image_reference(image_id)
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)

# print(image.shape)

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# masked = np.zeros(image.shape[:2])
# for i in range(mask.shape[2]):
#     masked += image[:, :, 0] * mask[:, :, i]
# plt.imshow(masked, cmap='gray')
# plt.axis('off')

# print(image_fp)
# print(class_ids)


# ### Image Augmentation. Try finetuning some variables to custom values

# In[18]:


# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# test on the same image as above
# imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
# plt.figure(figsize=(30, 12))
# _ = plt.imshow(imggrid[:, :, 0], cmap='gray')


# ### Now it's time to train the model. Note that training even a basic model can take a few hours. 
# 

# In[19]:


import keras
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# In[20]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

# Exclude the last layers because they require a matching
# number of classes
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])


# In[21]:


LEARNING_RATE = 0.005

# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")


# In[23]:
'''

time_callback = TimeHistory()
## train heads with higher lr to speedup the learning
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE*2,
            epochs=2,
            layers='heads',
            augmentation=None,
           custom_callbacks=[time_callback])
times = time_callback.times
times_list = []
times_list.append(times)
history = model.keras_model.history.history.copy()


# In[24]:


print('Epoch Training time:', times)
print(history)
'''

# In[45]:
times_list = [3100.942350625992, 2919.7681403160095]
history = {'val_loss': [2.1349234700202944, 2.147835547924042], 'val_rpn_class_loss': [0.04658446967601776, 0.05706217020750046], 'val_rpn_bbox_loss': [0.7988203525543213, 0.7603895425796509], 'val_mrcnn_class_loss': [0.3001007318496704, 0.42004683524370195], 'val_mrcnn_bbox_loss': [0.5343035399913788, 0.5152025067806244], 'val_mrcnn_mask_loss': [0.4551124614477158, 0.3951312863826752], 'loss': [1.6924433368444443, 1.6365146499872207], 'rpn_class_loss': [0.05106937070377171, 0.047678104992955925], 'rpn_bbox_loss': [0.45435774259269235, 0.4868608736619353], 'mrcnn_class_loss': [0.3177102542668581, 0.296053314357996], 'mrcnn_bbox_loss': [0.45967494644224643, 0.420139332190156], 'mrcnn_mask_loss': [0.40962940737605097, 0.38578060701489447]}

time_callback = TimeHistory()
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=10,
            layers='all',
            augmentation=augmentation,
           custom_callbacks=[time_callback])

times = time_callback.times
times_list.append(times)
news = model.keras_model.history.history
for k in news: history[k] = history[k] + news[k]


# In[49]:


print('Epoch Training time:', times)
print(history)


# In[50]:


model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE/5,
            epochs=15,
            layers='all',
            augmentation=augmentation,
           custom_callbacks=[time_callback])

times = time_callback.times
times_list.append(times)
news = model.keras_model.history.history
for k in news: history[k] = history[k] + news[k]


# In[51]:


print('Epoch Training time:', times)
print(history)


# In[25]:


epochs = range(1,len(next(iter(history.values())))+1)
df = pd.DataFrame(history, index=epochs)
df['epoch_train_time'] = times_list
df.to_csv(ROOT_DIR + 'loss.csv')


# In[26]:


plt.figure(figsize=(15,5))
plt.subplot(111)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.title('Masked RCNN Loss')
plt.legend()
plt.savefig('mrccn-loss-plot.png')
# plt.show()


# In[27]:


# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))
    
fps = []
# Pick last directory
for d in dir_names: 
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))


# In[55]:


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# ### Final steps - Create the submission file

# In[58]:


# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)


# In[59]:


# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.95):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString\n")

        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str

            file.write(out_str+"\n")


# In[ ]:


submission_fp = os.path.join(ROOT_DIR, 'submission-0.95.csv')
predict(test_image_fps, filepath=submission_fp, min_conf=0.95)
print(submission_fp)


# In[64]:


submission_fp = os.path.join(ROOT_DIR, 'submission-0.5.csv')
predict(test_image_fps, filepath=submission_fp, min_conf=0.5)
print(submission_fp)


# In[65]:


submission_fp = os.path.join(ROOT_DIR, 'submission-0.3.csv')
predict(test_image_fps, filepath=submission_fp, min_conf=0.3)
print(submission_fp)

