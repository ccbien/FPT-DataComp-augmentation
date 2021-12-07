import os
import shutil
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from skimage.io import imread, imsave

from augmentation import Augmentor
from utils import *


def prepare_directories(cf):
    if os.path.isdir(cf.output_dir):
        if cf.overwrite:
            shutil.rmtree(cf.output_dir)
        else:
            raise FileExistsError('The output directory already exists')
    
    image_dst = os.path.join(cf.output_dir, 'images/')
    label_dst = os.path.join(cf.output_dir, 'labels/')
    os.makedirs(image_dst)
    os.makedirs(label_dst)
    if cf.keep_origin:
        copy_all(cf.input_dir, image_dst, label_dst)
    return image_dst, label_dst


def main(cf):
    image_path = get_all_filepaths(cf.input_dir, is_image_file)
    label_path = get_all_filepaths(cf.input_dir, is_label_file)
    image_dst, label_dst = prepare_directories(cf)

    names = sorted(list(image_path.keys()))
    count = len(os.listdir(image_dst))
    augmentor = Augmentor(cf)

    for i in range(100): # big enough
        for name in names:
            try:
                image = imread(image_path[name])
                labels, bboxes = read_annotation(label_path[name])

                image_aug, bboxes_aug = augmentor.process(image, bboxes)
                if len(bboxes_aug) == 0:
                    continue
                
                name_aug = name + '_%02d' % i
                imsave(os.path.join(image_dst, name_aug + '.jpg'), image_aug)
                write_annotation(os.path.join(label_dst, name_aug + '.txt'), labels, bboxes_aug)
                
                count += 1
                if count % 10 == 0:
                    print(f'{name_aug} ({count})')
                if count == cf.limit: return
            except:
                print('Error', name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
        help='Path to a configuration file. Ex: ./config/base.yaml')
    args = parser.parse_args()
    cf = OmegaConf.load(args.config)
    main(cf)

    now = datetime.datetime.now()
    cur_time = (now.year, now.month, now.day, now.hour, now.minute, now.second)
    shutil.copy(
        args.config,
        './history/%d-%02d-%02d-%02d-%02d-%02d.yaml' % cur_time
    )
    