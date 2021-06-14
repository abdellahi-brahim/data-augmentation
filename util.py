#Other utils
import re
import math
import re
#Directories
import glob
#Data Structures
import xml.etree.ElementTree as ET
import pandas as pd

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