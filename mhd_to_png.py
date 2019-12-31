"""Converting .mhd images (mhd copies of DICOM) to png and identifying the border of the tumor"""
import pathlib
import shutil

import cv2
import os
import SimpleITK as Sitk
import numpy as np
import pydicom as dicom
import json

from PIL import Image

ultimate_train_folder_path = './tumor/train/'
ultimate_val_folder_path = './tumor/val/'
if os.path.exists('./tumor/'):
    shutil.rmtree('./tumor')
train_k = 0.5

# We fix max pixel value with a view to make the images displayable, otherwise the values of pixels are too small
# compared to max 16-bit colour value 256*256-1 (because the Hounsfield scale is used and its values are usually
# below 3500 for human tissues and bones)
max_p = 255 * 14
k = 255 // 14
# p/max_p = x/255*255 => x = p*255*255/max_p = p*255*255/14/255 = p*255/14

# Input folder path
sample_mhd_folder_path = "./MHD_samples/"
# Output folder path
sample_png_folder_path = "./PNG_samples/"

# Labels input folder path
label_mhd_folder_path = "./MHD_labels/"
# Labels output folder path
label_png_folder_path = "./PNG_labels/"

sample_png_folders = [x for x in os.listdir(sample_mhd_folder_path) if ".mhd" in x.lower()]
label_png_folders = [x for x in os.listdir(label_mhd_folder_path) if ".mhd" in x.lower()]
print(label_png_folders)
for screen_num, (sample_png_folder, label_png_folder) in enumerate(zip(sample_png_folders, label_png_folders)):
    json_data = {}
    sample_mhd_path = os.path.join(sample_mhd_folder_path, sample_png_folder)
    label_mhd_path = os.path.join(label_mhd_folder_path, label_png_folder)
    sample_mhd_image = Sitk.GetArrayFromImage(Sitk.ReadImage(sample_mhd_path))
    label_mhd_image = Sitk.GetArrayFromImage(Sitk.ReadImage(label_mhd_path))

    if screen_num / len(sample_png_folders) >= train_k:
        result_folder = ultimate_val_folder_path
    else:
        result_folder = ultimate_train_folder_path
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    print(label_mhd_image.shape[-1])

    for i in range(label_mhd_image.shape[-1]):

        # mhd labels
        layer = np.array(label_mhd_image[:, :, i])
        pixel_max_x, pixel_min_x = {}, {}
        """Find contour points (x is the label number for each tumor)"""
        for y_i, y in enumerate(layer):
            for x_i, x in enumerate(y):
                if x != 0:
                    if x not in pixel_min_x:
                        pixel_min_x[x], pixel_max_x[x] = {}, {}
                    if y_i not in pixel_min_x[x]:
                        pixel_min_x[x][y_i] = x_i
                        continue
                    pixel_max_x[x][y_i] = x_i

        for x in list(pixel_min_x.keys()):
            if len(pixel_min_x[x]) < 6:
                del pixel_min_x[x], pixel_max_x[x]
        if not pixel_min_x:
            if (i + 1) % 50 == 0:
                print('{} mhd sample and label images converted'.format(i + 1))
            continue

        """ Show contour points """
        # if i == 79:
        #     for y_i, y in enumerate(layer):
        #         for x_i, x in enumerate(y):
        #             print(x, end=' ')
        #         print()
        #     for x in sorted(pixel_min_x):
        #         print('****', x, '********')
        #         for y_i in sorted(pixel_min_x[x]):
        #             print(pixel_min_x[x][y_i], y_i)
        #         print('!!!!!')
        #         for y_i in sorted(pixel_max_x[x], reverse=True):
        #             print(pixel_max_x[x][y_i], y_i)

        # mhd samples
        layer = np.array(sample_mhd_image[:, :, i], dtype=np.uint16) * k
        image = sample_png_folder[:-4] + '-' + str(i + 1) + '.png'
        result_path = os.path.join(result_folder, image)
        cv2.imwrite(result_path, layer)
        # print(np.amax(layer))

        filename = image
        file_size = os.stat(result_path).st_size
        if (i + 1) % 50 == 0:
            print('{} mhd sample and label images converted'.format(i + 1))

        # json complement
        json_image_name = filename + str(file_size)
        json_data[json_image_name] = {}
        json_data[json_image_name]['filename'] = filename
        for j, x in enumerate(sorted(pixel_min_x)):
            x_list, y_list = [], []
            for y_i in sorted(pixel_min_x[x]):
                x_list.append(pixel_min_x[x][y_i])
                y_list.append(y_i)
            for y_i in sorted(pixel_max_x[x], reverse=True):
                x_list.append(pixel_max_x[x][y_i])
                y_list.append(y_i)
            if 'regions' not in json_data[json_image_name]:
                json_data[json_image_name]['regions'] = {}
            json_data[json_image_name]['regions'][str(j)] = {'shape_attributes': {'name': 'polygon',
                                                                                  'all_points_x': x_list,
                                                                                  'all_points_y': y_list},
                                                             'region_attributes': {}}

        with open(os.path.join(result_folder, 'via_region_data.json'), 'w+') as outfile:
            json.dump(json_data, outfile)
