import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Augmentor:
    def __init__(self, cf):
        self.seq = iaa.Sequential(random_order=True)

        if cf.gaussian_blur.use:
            sigma = (cf.gaussian_blur.sigma_min, cf.gaussian_blur.sigma_max)
            self.seq.add(iaa.blur.GaussianBlur(sigma=sigma))
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

        ia.seed(1)
        image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)
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



