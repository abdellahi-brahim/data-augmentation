#Imgaug
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug import augmenters as iaa
#Other utils
import re
#Image
import imageio
#Data Structures
import pandas as pd

from util import xml_to_pandas, export_to_xml, get_img
from filepath import xml_path, aug_path, aug_xml_path, image_path

def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    #create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=['filename','width','height','class','xmin','ymin','xmax', 'ymax'])
    
    grouped = df.groupby('filename')
    
    for filename in df['filename'].unique():
        #get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)

        #read the image
        image = imageio.imread(images_path+filename)
        #get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
        #clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
        #don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        
        #otherwise continue
        else:
            #write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug) 

            #create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            #rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
            #create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
            #concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            #append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])            
    
    #return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy

def bbs_obj_to_df(bbs_object):
    bbs_array = bbs_object.to_xyxy_array()
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

ia.seed(1)

labels_df = xml_to_pandas(xml_path)

images_df = get_img(labels_df)

aug_list = [
    iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Rot90((1), keep_size=False),
    iaa.Rot90((2), keep_size=False),
    iaa.Rot90((3), keep_size=False)
]

for index, aug in enumerate(aug_list):
    augmented_images_df = image_aug(images_df, image_path, aug_path, str(index+1), aug)
    export_to_xml(augmented_images_df, aug_xml_path)

