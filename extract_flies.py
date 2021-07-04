import cv2
import numpy as np

from util import xml_to_pandas, get_img, img2mask
from filepath import path, xml_path, fly_dataset, image_path

labels_df = xml_to_pandas(xml_path)
images_df = get_img(labels_df)

file_path = None
image = None
img_array = None
filename = None
fileclass = None
index = 0
target_size = 100

square = cv2.imread(path + "square-100.png")
s_w, s_h, _ = square.shape
center = (int(s_w/2), int(s_h))

for _, row in images_df.iterrows():
    #new image
    if file_path != image_path + row[0]:
        filename = row[0]
        index = 0
        file_path = image_path + filename
        
        #open new image
        img = cv2.imread(file_path)
        img_array = np.asarray(img)

        w, h, _ = img.shape
    
    inpaint = False

    #get bounding box coordinates from dataframe
    fileclass = row[3]

    xmin = row[4]
    ymin = row[5]
    xmax = row[6]
    ymax = row[7]

    height = ymax - ymin
    weight = xmax - xmin

    print(f"Height: {height}, Weight: {weight}")
    
    height_offset = 0
    weight_offset = 0

    #resize bounding box to target size
    if height < target_size or weight < target_size:
        weight_offset = int((target_size-weight)/2)
        height_offset = int((target_size-height)/2)

        #check if new dimensions are valid
        if xmax+weight_offset > w or ymax+height_offset > h:
            inpaint = True
            continue
    else:
        #resize bounding box to square
        if height > weight:
            weight_offset = int((height - weight)/2)
        else:
            height_offset = int((weight - height)/2)
    
    index += 1

    sector = img_array[ymin-height_offset:ymax+height_offset, xmin-weight_offset:xmax+weight_offset]

    if sector.size == 0: continue

    if inpaint:
        mask = img2mask(sector)
        sector = cv2.seamlessClone(sector, square, mask, center, cv2.NORMAL_CLONE)
    
    cv2.imwrite(fly_dataset + filename.replace('.jpg', '_' + fileclass + '_' + str(index) + '.jpg'), sector)