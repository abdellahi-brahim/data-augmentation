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

#moldura para inpaiting
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
    
    sector = img_array[ymin:ymax, xmin:xmax]

    print("Square:", square.size)

    #se dimensoes sao inferiores: fazer inpainting
    if height < target_size or weight < target_size:
        mask = img2mask(sector)
        print("Mask:", mask.size)
        sector = cv2.seamlessClone(sector, square, mask, center, cv2.NORMAL_CLONE)

    elif height > target_size or weight > target_size:
        h_off, w_off = (height-target_size)/2, (weight-target_size)/2
        sector = img2mask[ymin-h_off:ymax+h_off, xmin-w_off:xmax+w_off]

    index += 1

    if sector.size == 0: continue
    
    cv2.imwrite(fly_dataset + filename.replace('.jpg', f"'_' + {fileclass} + '_' + {index} + '.jpg'"), sector)