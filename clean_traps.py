import cv2
import numpy as np
from util import xml_to_pandas, get_img

image_path = "images\\"
xml_path = "annots\\"
aug_path = "aug_img\\"
aug_xml_path = "aug_xml\\"
fly_dataset = "fly_dataset\\"
empty_stick = "empty\\"

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

file_path = None
image = None
img_array = None
filename = None
fileclass = None
index = 0

for _, row in images_df.iterrows():
    #new image
    if file_path != image_path + row[0]:
        if img_array is not None:
            #remove 2 extra channels
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            empty = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(empty_stick + filename, empty)

        filename = row[0]
        index = 0
        file_path = image_path + filename
        
        #open new image
        img = cv2.imread(file_path)
        img_array = np.asarray(img)

        img_height, img_width,_ = img.shape
        n_channels = 3

        mask = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
        mask.fill(0)


    xmin = row[4]
    ymin = row[5]
    xmax = row[6]
    ymax = row[7]

    mask[ymin:ymax, xmin:xmax].fill(255)