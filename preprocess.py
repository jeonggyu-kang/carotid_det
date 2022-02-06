import os
import json
import shutil
import argparse
import glob

'''
Modularization (main)
    <data copy>
    [1] root directory, dst directory
    [2] .json parsing
    [3] json <-> bmp file, copy to one directory

    <new json parse>
    [1] shpaes [
        dict(
            label:str (NLI,NMA) || (FLI, FMA)
            points: [[x,y], [x,y], ... [x,y]]
        )
    ]
'''

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src_dir', type=str, default = './carotid_pp/hard_case')
    parser.add_argument('--image_dst_dir', type=str, default = None)
    return parser.parse_args()


def main():
    args = opt()

    # [1] file_rename (using index)

    


    os.makedirs(args.image_dst_dir, exist_ok = True)



    # a = glob.glob(args.image_dst_dir + '/*.bmp')
    for (root, dirs, files) in os.walk(args.image_src_dir):
        print("# root: " + root)
        if len(dirs) > 0:
            for file_name in files:
                os.path.join(os.path.join(root, dirs), file_name)
                print




if __name__ ='__main__'