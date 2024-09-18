import os
from os.path import join

def saveDir(cur_path):
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)

def checkDirs(output_folder):
    saveDir(join(output_folder, 'MaskArea'))
    saveDir(join(output_folder, 'Curves'))
    saveDir(join(output_folder, 'FilteredImages'))
    saveDir(join(output_folder, 'Original'))