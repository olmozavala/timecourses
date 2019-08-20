import os

def saveDir(cur_path):
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)

