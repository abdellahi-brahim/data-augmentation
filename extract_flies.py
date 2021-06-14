import cv2
import numpy as np

from util import xml_to_pandas, get_img
from filepath import xml_path, fly_dataset, image_path

labels_df = xml_to_pandas(xml_path)
images_df = get_img(labels_df)

file_path = None
image = None
img_array = None
filename = None
fileclass = None
index = 0

for _, row in images_df.iterrows():
    #new image
    if file_path != image_path + row[0]:
        filename = row[0]
        index = 0
        file_path = image_path + filename
        
        #open new image
        img = cv2.imread(file_path)
        img_array = np.asarray(img)

    #get bounding box coordinates from dataframe
    fileclass = row[3]

    xmin = row[4]
    ymin = row[5]
    xmax = row[6]
    ymax = row[7]

    height = ymax - ymin
    weight = xmax - xmin
    
    height_offset = 0
    weight_offset = 0

    if height > weight:
        weight_offset = int((height - weight)/2)
    else:
        height_offset = int((weight - height)/2)
    
    index += 1

    sector = img_array[ymin-height_offset:ymax+height_offset, xmin-weight_offset:xmax+weight_offset]

    if sector.size == 0: continue
    
    cv2.imwrite(fly_dataset + filename.replace('.jpg', '_' + fileclass + '_' + str(index) + '.jpg'), sector)