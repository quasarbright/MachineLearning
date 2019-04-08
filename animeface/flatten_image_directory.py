import os
import sys
from shutil import copyfile
thumb_path = 'D:\\datasets\\animeface-character-dataset\\thumb'
flat_path = 'D:\\datasets\\animeface-character-dataset\\flat'
files = 0
for thumb_folder in os.listdir(thumb_path):
    imgs = os.listdir(os.path.join(thumb_path, thumb_folder))
    for img in imgs:
        if img[-4:] == '.png':
            img_path = os.path.join(thumb_path, thumb_folder, img)
            new_filename = '{}_{}'.format(thumb_folder, img)
            new_path = os.path.join(flat_path, new_filename)
            copyfile(img_path, new_path)
