import os
import shutil

def is_image_file(file):
    return file.lower()[-4:] in ('.png', '.jpg')

def is_label_file(file):
    return file.endswith('.txt')

def get_all_filepaths(src, checker):
    path = {}
    for root, dirs, files in os.walk(src):
        for file in files:
            if checker(file):
                name = file[:file.rfind('.')]
                path[name] = os.path.join(root, file)
    return path

def copy_all(src, image_dst=None, label_dst=None):
    for root, dirs, files in os.walk(src):
        for file in files:
            if image_dst is not None and is_image_file(file):
                shutil.copy(
                    os.path.join(root, file),
                    os.path.join(image_dst, file)
                )
            if label_dst is not None and is_label_file(file):
                shutil.copy(
                    os.path.join(root, file),
                    os.path.join(label_dst, file)
                )

def read_annotation(anno_path):
    labels = []
    bboxes = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            label, x, y, w, h = map(float, line.split())
            labels.append(int(label))
            bboxes.append((x, y, w, h))
    return labels, bboxes


def write_annotation(anno_path, labels, bboxes):
    lines = []
    for label, (x, y, w, h) in zip(labels, bboxes):
        lines.append('%d %.6f %.6f %.6f %.6f\n' % (label, x, y, w, h))
    lines[-1] = lines[-1][:-1]
    with open(anno_path, 'w') as f:
        f.writelines(lines)

def yolo2normal(bbox, W, H):
    x, y, w, h = bbox
    w *= W
    h *= H
    x = x * W - w / 2
    y = y * H - h / 2
    return x, y, w, h

def normal2yolo(bbox, W, H):
    x, y, w, h = bbox
    x += w / 2
    y += h / 2
    return x / W, y / H, w / W, h / H