import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Augmentor:
    def __init__(self, cf):
        self.seq = []
        self.probs = []

        if cf.gaussian_blur.prob > 0:
            sigma = (cf.gaussian_blur.sigma_min, cf.gaussian_blur.sigma_max)
            self.seq.append(iaa.blur.GaussianBlur(sigma=sigma))
            self.probs.append(cf.gaussian_blur.prob)

        if cf.linear_contrast.prob > 0:
            alpha = (cf.linear_contrast.alpha_min, cf.linear_contrast.alpha_max)
            self.seq.append(iaa.contrast.LinearContrast(alpha=alpha))
            self.probs.append(cf.linear_contrast.prob)

        if cf.add_to_brightness.prob > 0:
            self.seq.append(iaa.AddToBrightness(add=(cf.add_to_brightness.min, cf.add_to_brightness.max)))
            self.probs.append(cf.add_to_brightness.prob)

        if cf.crop.prob > 0:
            self.seq.append(iaa.Crop(percent=(cf.crop.min_ratio, cf.crop.max_ratio)))
            self.probs.append(cf.crop.prob) 

        if cf.flip_horizontal.prob > 0:
            self.seq.append(iaa.flip.Fliplr(1.0))
            self.probs.append(cf.flip_horizontal.prob)

        if cf.rotate.prob > 0:
            self.seq.append(iaa.Rotate(rotate=(cf.rotate.min_angle, cf.rotate.max_angle)))
            self.probs.append(cf.rotate.prob)
        
        # TO BE CONTINUE


    def process(self, image, bboxes):
        """
        image:
            - Numpy uint8 array with shape (H, W, 3)
            - Color range: 0 .. 255
        bboxes:
            - List of tuples (x, y, w, h) - follows YOLOv5's format
        """
        bb_list = []
        for x, y, w, h in bboxes:
            x *= image.shape[1]
            y *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]
            bb_list.append(BoundingBox(x1=x - w / 2, y1=y - h/2, x2=x + w/2, y2=y + h/2))
        bbs = BoundingBoxesOnImage(bb_list, shape=image.shape)

        # ia.seed(1)
        image_aug, bbs_aug = image, bbs
        for aug, p in zip(self.seq, self.probs):
            if np.random.rand() < p:
                image_aug, bbs_aug = aug(image=image_aug, bounding_boxes=bbs_aug)
        # TO BE CONTINUE
        
        bboxes_aug = []
        for bb in bbs_aug:
            w = bb.x2 - bb.x1
            h = bb.y2 - bb.y1
            x = bb.x1 + w / 2
            y = bb.y1 + h / 2
            x /= image_aug.shape[1]
            y /= image_aug.shape[0]
            w /= image_aug.shape[1]
            h /= image_aug.shape[0]
            bboxes_aug.append((x, y, w, h))
        
        return image_aug, bboxes_aug



