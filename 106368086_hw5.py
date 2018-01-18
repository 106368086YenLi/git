#coding=utf-8
import os
import re
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
import cv2

savingDir = "./data/temp"

def load_xml(xml_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    folder = root[0].text
    filename = root[1].text
    size = [int(root[4][0].text), int(root[4][1].text), int(root[4][2].text)]
    object_name = root[6][0].text
    # bbox = [int(root[6][4][0].text), int(root[6][4][1].text), int(root[6][4][2].text), int(root[6][4][3].text)]
    xmin = int(root[6][4][0].text)
    ymin = int(root[6][4][1].text)
    xmax = int(root[6][4][2].text)
    ymax = int(root[6][4][3].text)
    # imagefile = os.path.join('./data/train', filename)
    # image = cv2.imread(imagefile/)
    # image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    # savingimagefile = os.path.join('./data/temp', filename)
    # cv2.imwrite(savingimagefile, image)
    return folder, filename, size, object_name, xmin, ymin, xmax, ymax

data_dir = './data'
pattern = re.compile('(.+\/)?(\w+)\/([^_]+)_.+xml')
all_files = glob(os.path.join(data_dir, 'train/*xml'))
all_files = [re.sub(r'\\', r'/', file) for file in all_files]
print(len(all_files))

frames = []
xmins = []
xmaxs = []
ymins = []
ymaxs = []
class_ids = []
i = 0
for entry in all_files:
    r = re.match(pattern, entry)
    if r:
        folder, filename, size, object_name, xmin, ymin, xmax, ymax = load_xml(entry)
        file_name, file_extension = os.path.splitext(filename)
        if not file_name in entry:
            # print(filename, entry)
            continue
        frames.append(filename)
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        class_ids.append(1)
        if i == 18000:
            frames = [re.sub(r'xml', r'jpg', frame) for frame in frames]
            train_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs, 'ymin': ymins, 'ymaxs': ymaxs, 'class_id': class_ids})
            train_labels = train_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymaxs', 'class_id']]
            train_labels.to_csv("train_labels.csv", index=False)
            frames = []
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            class_ids = []
        i += 1
frames = [re.sub(r'xml', r'jpg', frame) for frame in frames]
val_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs, 'ymin': ymins, 'ymaxs': ymaxs, 'class_id': class_ids})
val_labels = val_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymaxs', 'class_id']]
val_labels.to_csv("val_labels.csv", index=False)


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator


classes = ['background', '9256A9', '8C3676', '5199T2', '8210M5', '8381YJ', '6791H8', '1863J6', 'RAJ6828', 'AAP2926', '2975A5', 'ADB2531', '8671K9', '1850YN', '0697QN', '0573VN', '3863J7', 'AAP2927', '0712QZ', 'AGV8338', '5786YZ', '2E0325', 'ZM5831', 'AAQ6006', '5119J3', 'AFG8307', 'AHB5826', 'RAJ8668', '6237EJ', '3719YJ', '5K5155', '7D8953', '0871RG', '479922', 'AFG1929', '8673K9', '2E1723', '5902VF', '9171QM', 'APC6071', 'AFJ1121', '7679J2', '0663UX', '1553ZS', '0198YZ', '8979YD', '3861J2', '8E8338', '6522QA', '7U9123', '9101ZR', '8586VN', 'AHF2353', '4537EH', '5592MP', '8128G5', '7050EB', '6150J5', '3827J8', '1702YC', 'AGE5239', 'RAD8819', 'ABY2895', 'AGV7939', '1508L8', 'V22999', '7177ET', '333633', '6A8338', 'AGC6259', 'AGP2206', '6908YQ', '7285UT', 'RAN9181', 'APK5605', '1550ZS', 'N88450', '8008DX', 'X59329', '0592VE', '8672K9', '5005EU', 'RBD8610', '8208S3', 'AAL8668', '3838EU', 'APM5527', '7821VH', '6790ZW', 'DK6266', '0719DE', '2712A3', 'AKZ1266', 'APM9171', 'ALG8668', '5823DY', '0685YK', 'AFJ8612', '6D9569', '1113P2', '4571H6', '8516DG', '9168T2', '0209J3', '6250M3', 'AAR8955', '7267YA', '3A3268', '9590VE', '9929UX', '6613J7', '3R3337', 'AFG7232', '5150D2', '3993YG', 'ALB3772', 'AAB0251', '3N8761', '3079DH', '8195VB', 'AKN9936', '9023L7', 'AFJ1165', 'AGS9170', 'AKP1637', '0803QB', '6795RH', '3G5889', 'V63698', 'AJN1621', '6336A6', '6336VS', 'AGS9180', '863J6', '5C7267', '9805DX', '2820J2', '5690EG', '2K2115', 'RAD9119', '2990R8', '6173EP', 'AAY7711', 'AFF6790', '3812QW', '9233L7', 'AGT7230', 'ATD1351', '3165J3', '6639H8', 'AMZ3823', 'RAF2380', '2393A9', 'AFG0386', 'AFJ8512', '0631T3', '2689B3', 'T21288', '9936YC', '7770XX', '2786L3', '5692S2', '5335QT', '7376EJ', 'ABB9182', '9889B8', '793ZP', '2782J9', 'AKG3503', 'ABB9061', '988833', '0713QZ', '5255J6', '5195B7', '8663J8', 'AGT7221', '789A6', 'APC5906', 'ANZ7938', 'ABB3877', 'AKY0053', 'RAA7500', '1663ZT', '6683YK', 'AGF9119', 'RAB7266', '6153J5', 'ADB2551', 'AFK3953', '5565MQ', '8E3310', 'APD1317', '5119J5', 'APF0059', 'AKK2970', '829W6', 'AKU7838', '2790J9', '9119G3', '8533TU', '6502KU', '0303L3', 'ACC1111', 'AGJ1565', '787CZ', 'AGT7210', 'AGV7929', 'ADD1227', '8672VN', 'RBB8058', 'ANV6770', 'APC5305', 'ALH5710', '5383YB', 'AFK3965', 'AGL5853', 'AFF9159', 'AFK3952', 'IW4266', 'ABD6580', 'ABB9051', 'AKN8705', 'AGS0100', 'AAL7555', '719M2', '2828LM', 'AGL1853', 'AGV7935', 'AAD5113', '6171KL', 'AGV793', '3T4466', 'RAQ7557', 'AGR2100', 'AAM7252', '3A2795', 'AKT2275', '5330YP', 'AKW5712', 'AQA3337', 'ABB9175', 'AGF9299', '7868QD', '0707C', '3163J3', 'AGF2753', 'ANV6753', 'ABB9183', 'AKP0999', '0756SD', '2K7429', '1289A3', 'DK8088', '7873RH', 'AAR8563', '8520N6', '2783J9', 'AKL1185', 'AKW8930', '6855ZA', '8668WW', 'AKU5781']

# 1. Set the model configuration parameters
# img_height = 300 # Height of the input images
# img_width = 480 # Width of the input images
img_height = 240 # Height of the input images
img_width = 320 # Width of the input images
img_channels = 3 # Number of color channels of the input images
# n_classes = 6 # Number of classes including the background class
n_classes = 2 # N；uber of classes including the background class
min_scale = 0.08 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]

# II. Build or load the model
# II.1 Create a new model
K.clear_session() # Clear previous models from memory.
model = build_model(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=None,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# III. Set up the data generators for the training
train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

# train_images_path = './sample/data/udacity_driving_datasets/'
# train_labels_path = './sample/data/udacity_driving_datasets/train_labels.csv'
# val_images_path = './sample/data/udacity_driving_datasets/'
# val_labels_path = './sample/data/udacity_driving_datasets/val_labels.csv'
train_images_path = './data/train'
train_labels_path = './train_labels.csv'
val_images_path = './data/train'
val_labels_path = './val_labels.csv'

train_dataset.parse_csv(images_dir=train_images_path,
                        labels_filename=train_labels_path,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=val_images_path,
                      labels_filename=val_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer=None,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

batch_size = 16

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5), # Randomly change brightness between 0.5 and 2 with probability 0.5
                                         flip=0.5, # Randomly flip horizontally with probability 0.5
                                         translate=((5, 50), (3, 30), 0.5), # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                         scale=(0.75, 1.3, 0.5), # Randomly scale between 0.75 and 1.3 with probability 0.5
                                         max_crop_and_resize=False,
                                         full_crop_and_resize=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4
                                         )

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=False,
                                     full_crop_and_resize=False,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     )

n_train_samples = train_dataset.get_n_samples()
n_val_samples = val_dataset.get_n_samples()

# IV. Run the training
epochs = 5

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(n_train_samples/batch_size),
                              epochs=epochs,
                              callbacks=[ModelCheckpoint('ssd7_weights_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           period=1),
                                           EarlyStopping(monitor='val_loss',
                                                         min_delta=0.001,
                                                         patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5,
                                                             patience=0,
                                                             epsilon=0.001,
                                                             cooldown=0)],
                              validation_data=val_generator,
                              validation_steps=ceil(n_val_samples/batch_size))

model_name = 'ssd7'
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))
print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()

# V. Make predictions

test_output = pd.read_csv('./data/sample-submission.csv')
for i in range(10000):
   filename = './data/test/'+str(i+1)+'.jpg'; #print(filename)
   X = cv2.imread(filename)
   X = np.expand_dims(X, 0)
   y_pred = model.predict(X)
   y_pred_decoded = decode_y2(y_pred, confidence_thresh=0.4, iou_threshold=0.4, top_k='all', input_coords='centroids', normalize_coords=False, img_height=None, img_width=None)
#   if len(y_pred_decoded[0])==0: label = 'unknown'; print(label);
   box = y_pred_decoded[0][0]; label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
   label = label.split(':')[0]
   print(label)
   test_output.set_value(i, 'Number', label)
test_output.to_csv("./data/test_output.csv", index=False)

