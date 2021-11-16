import os
import shutil
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from skimage.io import imread, imsave

from augmentation import Augmentor
from utils import read_annotation, write_annotation


def prepare_paths(cf):
    image_src = {
        'train': os.path.join(cf.input_dir, 'images/train'),
        'val': os.path.join(cf.input_dir, 'images/val'),
        'public_test': os.path.join(cf.input_dir, 'images/public_test')
    }
    label_src = {
        'train': os.path.join(cf.input_dir, 'labels/train'),
        'val': os.path.join(cf.input_dir, 'labels/val'),
        'public_test': os.path.join(cf.input_dir, 'labels/public_test')
    }
    image_dst = {
        'train': os.path.join(cf.output_dir, 'images/train'),
        'val': os.path.join(cf.output_dir, 'images/val'),
        'public_test': os.path.join(cf.output_dir, 'images/public_test')
    }
    label_dst = {
        'train': os.path.join(cf.output_dir, 'labels/train'),
        'val': os.path.join(cf.output_dir, 'labels/val'),
        'public_test': os.path.join(cf.output_dir, 'labels/public_test')
    }
    if os.path.isdir(cf.output_dir):
        if cf.overwrite:
            shutil.rmtree(cf.output_dir)
        else:
            raise FileExistsError('The output directory already exists')
    return image_src, label_src, image_dst, label_dst


def main(cf):    
    image_src, label_src, image_dst, label_dst = prepare_paths(cf)
    for kind in ('train', 'val', 'public_test'):
        if cf.keep_origin[kind]:
            shutil.copytree(image_src[kind], image_dst[kind])
            shutil.copytree(label_src[kind], label_dst[kind])
        else:
            os.makedirs(image_dst[kind])
            os.makedirs(label_dst[kind])

    augmentor = Augmentor(cf)
    for kind in ('train', 'val', 'public_test'):
        print(f'Processing "{kind}":')
        files = sorted(os.listdir(image_src[kind]))
        count = len(os.listdir(image_dst[kind]))
        for i in range(100): # big enough
            for file in tqdm(files):

                if not file.endswith('.jpg'): continue
                name = file[:-4]
                image = imread(os.path.join(image_src[kind], file))
                labels, bboxes = read_annotation(
                    os.path.join(label_src[kind], name + '.txt'))

                image_aug, bboxes_aug = augmentor.process(image, bboxes)
                name_aug = name + '_%02d' % i
                imsave(
                    os.path.join(image_dst[kind], name_aug + '.jpg'),
                    image_aug)
                write_annotation(
                    os.path.join(label_dst[kind], name_aug + '.txt'),
                    labels,
                    bboxes_aug)
                count += 1
                if count == cf.limit[kind]: break
            if count == cf.limit[kind]: break


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
    