import cv2
import numpy as np

from util import xml_to_pandas, get_img
from filepath import xml_path, empty_stick, image_path

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