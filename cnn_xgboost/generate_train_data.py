import os
import pandas as pd
import csv
import time

from core.dataset import DIGITS
from core.data_preprocess import get_mel_img_data

def generate_img(audio_root, img_root):
    # generate spoken_digit.csv file
    spoken_digit_file = os.path.join(audio_root, 'spoken_digit.csv')
    if not os.path.isfile(spoken_digit_file):
        print('Generate:', spoken_digit_file)
        with open(spoken_digit_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["File", "Label"])
            for d in DIGITS:
                spoken_digit_subroot = os.path.join(audio_root, d)
                if not os.path.isdir(spoken_digit_subroot):
                    continue
                for name in os.listdir(spoken_digit_subroot):
                    if os.path.isfile(os.path.join(spoken_digit_subroot, name)):
                        csvwriter.writerow([os.path.join(d, name), d])

    # make dirs to store rebuild gray mel images if not exist
    if not os.path.isdir(img_root):
        print('Generate:', img_root)
        os.makedirs(img_root)
    
    for d in DIGITS:
        digit_path = os.path.join(img_root, d)
        if not os.path.isdir(digit_path):
            print('Generate:', digit_path)
            os.makedirs(digit_path)

    # load spoken digit index data
    df = pd.read_csv(spoken_digit_file)
    for i in range(0, len(df)):
        audio_path = os.path.join(audio_root, df.loc[i].File)
        img_path = os.path.join(img_root, str(df.loc[i].File[:-4]) + '.jpg')
        get_mel_img_data(audio_path, img_path)

        if (i % 100) == 0:
            print(time.ctime()[11:19], i, '/', len(df))



if __name__ == '__main__':
    import argparse

    # process args
    parser = argparse.ArgumentParser(description='Generate-CNN-Train-Data')
    parser.add_argument('--root', default='/data/acp20ww/secure-robotic-delivery-system/', 
        metavar='N', help='root path of the project')
    parser.add_argument('--dataset_folder', default='ghostnet/data/speech_commands_v0.01', 
        metavar='N', help='relative path to training dataset config file')
    parser.add_argument('--image_folder', default='cnn_xgboost/data/melimages_55x55_png', 
        metavar='N', help='relative path to training image')
    args = parser.parse_args()

    audio_root = os.path.join(args.root, args.dataset_folder)
    img_root = os.path.join(args.root, args.image_folder)
    generate_img(audio_root, img_root)    




# python generate_train_data.py --root /Users/yuehui/Documents/wangwei/UniSheffield/secure-robotic-delivery-system/

# python --root /home/comrobot1/Documents/com6013/ --dataset_folder COM6013/speech_commands_v0.01 --image_folder secure-robotic-delivery-system/cnn_xgboost/data/melimages_55x55_png











