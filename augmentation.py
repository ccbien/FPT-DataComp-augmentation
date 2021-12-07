from imgaug.augmenters.flip import HorizontalFlip, fliplr
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from utils import yolo2normal, normal2yolo


class Augmentor:
    def __init__(self, cf):
        self.seq = []
        self.probs = []

        # Color transformation
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

        # Geometric transformation
        if cf.rotate.prob > 0:
            self.seq.append(iaa.Rotate(rotate=(cf.rotate.min_angle, cf.rotate.max_angle)))
            self.probs.append(cf.rotate.prob)

        if cf.flip_lr.prob > 0:
            self.seq.append(iaa.Fliplr(1.0))
            self.probs.append(cf.flip_lr.prob)

        if cf.crop.prob > 0:
            self.seq.append(iaa.Crop(percent=(cf.crop.min_ratio, cf.crop.max_ratio)))
            self.probs.append(cf.crop.prob) 

        if cf.scale_x.prob > 0:
            self.seq.append(iaa.ScaleX(scale=(cf.scale_x.min_scale, cf.scale_x.max_scale)))
            self.probs.append(cf.scale_x.prob)

        if cf.scale_y.prob > 0:
            self.seq.append(iaa.ScaleY(scale=(cf.scale_y.min_scale, cf.scale_y.max_scale)))
            self.probs.append(cf.scale_y.prob)

        if cf.shear_x.prob > 0:
            self.seq.append(iaa.ShearX(shear=(cf.shear_x.min_angle, cf.shear_x.max_angle)))
            self.probs.append(cf.shear_x.prob)

        if cf.shear_y.prob > 0:
            self.seq.append(iaa.ShearY(shear=(cf.shear_y.min_angle, cf.shear_y.max_angle)))
            self.probs.append(cf.shear_y.prob)


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
            x, y, w, h = yolo2normal((x, y, w, h), image.shape[1], image.shape[0])
            bb_list.append(BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h))
        bbs = BoundingBoxesOnImage(bb_list, shape=image.shape)

        image_aug, bbs_aug = image, bbs
        for aug, p in zip(self.seq, self.probs):
            if np.random.rand() < p:
                image_aug, bbs_aug = aug(image=image_aug, bounding_boxes=bbs_aug)
        
        bboxes_aug = []
        for bb in bbs_aug:
            x = bb.x1
            y = bb.y1
            w = bb.x2 - bb.x1
            h = bb.y2 - bb.y1
            x, y, w, h = normal2yolo((x, y, w, h), image_aug.shape[1], image_aug.shape[0])
            bboxes_aug.append((x, y, w, h))
        
        image_aug, bboxes_aug = self.remove_black_border(image_aug, bboxes_aug)
        bboxes_aug = self.postprocess_bboxes(bboxes_aug, image_aug.shape[1], image_aug.shape[0])
        return image_aug, bboxes_aug

    
    @staticmethod
    def remove_black_border(image, bboxes):
        for xmin in range(image.shape[1]):
            if image[:,xmin,:].sum() > 0:
                break
        for xmax in range(image.shape[1] - 1, -1, -1):
            if image[:,xmax,:].sum() > 0:
                break
        for ymin in range(image.shape[0]):
            if image[ymin,:,:].sum() > 0:
                break
        for ymax in range(image.shape[0] - 1, -1, -1):
            if image[ymax,:,:].sum() > 0:
                break
        xmax += 1
        ymax += 1
        
        new_image = image[ymin:ymax, xmin:xmax, :]
        new_bboxes = []
        for (x0, y0, w0, h0) in bboxes:
            x = (x0 * image.shape[1] - xmin) / (xmax - xmin)
            y = (y0 * image.shape[0] - ymin) / (ymax - ymin)
            w = w0 * image.shape[1] / (xmax - xmin)
            h = h0 * image.shape[0] / (ymax - ymin)
            new_bboxes.append((x, y, w, h))
        return new_image, new_bboxes


    @staticmethod
    def postprocess_bboxes(bboxes, W, H):
        new_bboxes = []
        eps = 1e-6
        for x, y, w, h in bboxes:
            xmin = max(0, min(1, x - w/2))
            xmax = max(0, min(1, x + w/2))
            ymin = max(0, min(1, y - h/2))
            ymax = max(0, min(1, y + h/2))
            if abs(xmin - xmax) <= eps:
                continue
            if abs(ymin - ymax) <= eps:
                continue
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2
            y = ymin + h/2
            new_bboxes.append((x, y, w, h))
        return new_bboxes




