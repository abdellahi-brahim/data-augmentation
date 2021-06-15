import cv2
import os
import random
import pandas as pd
import numpy as np

from blending import blend
from util import xml_to_pandas, get_img, export_to_xml
from filepath import xml_path, gan_path, empty_stick, aug_path, aug_xml_path

def get_fly(filename):
    image = cv2.imread(filename)
    mask = 255 * np.ones(image.shape, image.dtype)
    
    mask[:, -1] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[0, :] = 0

    return image, mask

labels_df = xml_to_pandas(xml_path)
images_df = get_img(labels_df)

backgrounds = images_df['filename'].unique()
flies = os.listdir(gan_path)

annotations = []

#Create dataset of wf in diferent positions
for f in backgrounds:
    #get current background annotations
    data = images_df.loc[images_df['filename'] == f]
    ratio = len(data)
    #open empty sticky trap
    background = cv2.imread(empty_stick + f)

    if background is None:
        continue

    #get random flies from dataset
    if ratio <= len(flies):
        sample = random.sample(flies, ratio)
        flies = list(set(flies) - set(sample))
        print("Avaliable:", len(flies))
    #No more flies to add
    else:
        print(f"No more flies to add. Avaliable: {len(flies)}, Requested: {ratio}")
        break

    for fly, row in zip(sample, data.itertuples(index=True, name='Pandas')):
        img, mask = get_fly(gan_path + fly)
        img_w, img_h, _ = img.shape

        x_off = int(img_w/2)
        y_off = int(img_h/2)

        t_off = (x_off, y_off)

        center = (row.xmin + x_off, row.ymin + y_off)

        up_corner = tuple(np.subtract(center, t_off))
        down_corner = tuple(np.add(center, t_off))
        
        if background.shape[1] > row.xmin + img_w and background.shape[0] > row.ymin + img_h:
            #background = blend(img, background, (row.ymin, row.xmin), method='Segmented')
            background = cv2.seamlessClone(img, background, mask, center, cv2.NORMAL_CLONE)
        else:
            print(f"Background: {background.shape}\nLimits: {center}")
            flies.append(fly)
            continue
        
        annot = ['7'+f, background.shape[0], background.shape[1], 'WF']
        annot.extend(up_corner+down_corner)
        annotations.append(annot)

    cv2.imwrite(aug_path + '7' + f, background)
  
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
new_df = pd.DataFrame(annotations, columns=column_name)
export_to_xml(new_df, aug_xml_path)
