#Other utils
import re
import cv2
import math
import os
import re
import random
#Directories
import glob
#Data Structures
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from blending import blend

def xml_to_pandas(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        a = re.findall('\d+', xml_file )[0] + ".jpg"
        for member in root.findall('object'):
            value = (a,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def export_to_xml(augmented_images_df, filepath):
    current = ''    
    xml = ''
    previous = ''
    count = 0
    for _, row in augmented_images_df.iterrows():
        current = re.findall('\d+', row[0])[0] + '.xml'
        if(previous == ''):
            xml = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n\t<folder>--</folder>\n\t<filename>--</filename>\n\t<path>--</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n'
            xml += '\t<size>\n\t\t<width>{0}</width>\n'.format(row[1])
            xml += '\t\t<height>{0}</height\n\t</size>\n\t<segmented>3</segmented>\n'.format(row[2])
            previous = current
        if(current != previous):
            xml += '</annotation>'
            with open(filepath + previous, 'w') as f:
                f.write(xml)
            xml = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n\t<folder>--</folder>\n\t<filename>--</filename>\n\t<path>--</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n'
            xml += '\t<size>\n\t\t<width>{0}</width>\n'.format(row[1])
            xml += '\t\t<height>{0}</height>\n\t</size>\n\t<segmented>3</segmented>\n'.format(row[2])
            previous = current
            count += 1
        if(not math.isnan(row[6])):
          xml += '\t<object>\n\t\t<name>{1}</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<{2}>{3}</{4}>\n\t\t\t<{5}>{6}</{7}>\n\t\t\t<{8}>{9}</{10}>\n\t\t\t<{11}>{12}</{13}>\n\t\t</bndbox>\n\t</object>\n'.format(row[2], row[3],row.index[4],row[4],row.index[4],row.index[5],row[5],row.index[5],row.index[6],row[6],row.index[6],row.index[7],row[7],row.index[7])

def get_img(df):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns = ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax'])
    grouped = df.groupby('filename')    
    
    for filename in df['filename'].unique():
    #   Get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    
    # return dataframe with images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy

def get_fly(filename):
    image = cv2.imread(filename)
    mask = 255 * np.ones(image.shape, image.dtype)

    return image, mask

def get_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.getGaussianKernel(9,9)
    blur= cv2.GaussianBlur(gray_image,(5,5),0)

    kernel=np.ones((5,5),np.float32)/25
    averaged= cv2.filter2D(blur,-1,kernel)

    _, thresh = cv2.threshold(averaged,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    mask = np.ones(sure_bg.shape, np.uint8)
    mask = cv2.bitwise_not(sure_bg, mask)

    return mask


def get_limit_from_image(background):
    mask = get_mask(background)

    white = np.argwhere(mask == 255)
    sorted = np.sort(white[:,1])

    y1, _ = white[0]
    x1 = sorted[sorted != 0][0]
    y2, x2 = white.max(axis=0)

    return x1, y1, x2, y2

image_path = "images\\"
xml_path = "annots\\"
aug_path = "aug_img\\"
aug_xml_path = "aug_xml\\"
fly_dataset = "fly_dataset\\"
empty_stick = "empty\\"

labels_df = xml_to_pandas(xml_path)
images_df = get_img(labels_df)

backgrounds = images_df['filename'].unique()
flies = os.listdir(fly_dataset)

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
        img, mask = get_fly(fly_dataset + fly)
        img_w, img_h, _ = img.shape

        x_off = int(img_w/2)
        y_off = int(img_h/2)

        t_off = (x_off, y_off)

        center = (row.xmin + x_off, row.ymin + y_off)

        up_corner = tuple(np.subtract(center, t_off))
        down_corner = tuple(np.add(center, t_off))
        
        if background.shape[1] > row.xmin + img_w and background.shape[0] > row.ymin + img_h:
            background = blend(img, background, (row.ymin, row.xmin), method='Segmented')
            #background = cv2.seamlessClone(img, background, mask, center, cv2.NORMAL_CLONE)
        else:
            print(f"Background: {background.shape}\nLimits: {center}")
            flies.append(fly)
            continue
        
        annot = ['6'+f, background.shape[0], background.shape[1], 'WF']
        annot.extend(up_corner+down_corner)
        annotations.append(annot)

    cv2.imwrite(aug_path + '6' + f, background)
  
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
new_df = pd.DataFrame(annotations, columns=column_name)
export_to_xml(new_df, aug_xml_path)