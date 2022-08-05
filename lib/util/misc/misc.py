import glob
import json
import numpy as np

def json_to_keypoints(json_path):
    with open(json_path, 'rb') as my_file:
        data = json.load(my_file)
    return np.array(data['keypoints'])

def immediate_sub_dirs(path):
    return glob.glob(f"{path}/*/")
