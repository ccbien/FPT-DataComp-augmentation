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