import os
import cv2
import numpy as np

GTSDB_TRAIN_RANGE = 600
GTSDB_VAL_RANGE = 300

# PRESCAN_ID_TO_YOLO_ID = {
#     95: 0,
#     96: 1,
#     98: 2,
#     99: 3,
#     100: 4,
#     101: 5,
#     102: 6,
#     103: 7,
#     104: 8,
#     105: 9, 
#     106: 10,
#     107: 11
# }

def convert_ltrb_to_xywh(left, bottom, right, top, img_width, img_height):
    '''Convert (left, bottom, right, top) format to normalized (x_center, y_center, width, height)'''

    x = (right + left) / 2 / img_width
    y = (top + bottom) / 2 / img_height
    w = (right - left) / img_width
    h = (top - bottom) / img_height

    return x, y, w, h
    

def convert_GTSDB_gtfile_to_yolo(gtsdb_dir, output_dir):
    '''Convert the gt.txt file of GTSDB dataset into YOLO format'''

    # Get the image weight and height
    img_path = os.path.join(gtsdb_dir, "00000.ppm")
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    # Transfer gtfile to YOLO format ".txt" files
    # The file structure is (https://docs.ultralytics.com/zh/datasets/#contribute-new-datasets):
    # dataset/
    # ├── train/
    # │   ├── images/
    # │   └── labels/
    # └── val/
    #     ├── images/
    #     └── labels/

    # Label file format (https://docs.ultralytics.com/zh/datasets/detect/#ultralytics-yolo-format): class_id x_center y_center width height
    gtfile_path = os.path.join(gtsdb_dir, "gt.txt")
    with open(gtfile_path, "r") as gtfile:
        annotations = gtfile.readlines()
    lines = [line.strip() for line in annotations]

    labels_train_dir = os.path.join(output_dir, "train/labels")
    labels_val_dir = os.path.join(output_dir, "val/labels")
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    for line in lines:
        parts = line.split(";")
        filename, left, bottom, right, top, class_id = parts[0], *map(int, parts[1:5]), parts[5]
        x, y, w, h = convert_ltrb_to_xywh(left, bottom, right, top, img_width, img_height)

        # Put the labels of the first GTSDB_TRAIN_RANGE images into "train/labels" folder
        img_id = int(filename.replace(".ppm", ""))
        if img_id < GTSDB_TRAIN_RANGE:
            label_file_path = os.path.join(labels_train_dir, filename.replace(".ppm", ".txt"))
            with open(label_file_path, "a") as label_file:
                label_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        else:
            label_file_path = os.path.join(labels_val_dir, filename.replace(".ppm", ".txt"))
            with open(label_file_path, "a") as label_file:
                label_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
def convert_prescan_gt_to_yolo_gt(prescan_gt: np.ndarray, img_width, img_height, numid_to_classid) -> str:
    '''
    Convert one Prescan ground truth to YOLO format

    Format of Prescan ground truth: (object_id, left, right, bottom, top), here the coordinate values are displayed in values between -1 and 1. The coordinates system is 
    Prescan format, where the origin is at the bottom-left corner.
    Format of YOLO ground truth: (label_id, x_center, y_center, width, height), here the (x_center, y_center, width, height) are normalized to between 0 and 1. The coordinates 
    system is Opencv format, where the origin is at the top-left corner.
    '''
    object_id = int(prescan_gt[0])
    class_id = numid_to_classid[object_id]

    l, r, b, t = prescan_gt[1:]
    # Convert lrbt (-1, 1) to normal size (img_width, img_height)
    l = (l + 1) * (img_width / 2)
    r = (r + 1) * (img_width / 2)
    b = (b + 1) * (img_height / 2)
    t = (t + 1) * (img_height / 2)

    # Convert lrbt (imimg_width, img_height) from Prescan coordinate system to yolo's
    b, t = img_height - t, img_height - b

    # Convert lrbt (normal size) to normalized xywh
    x = (l + r) / 2 / img_width
    y = (b + t) / 2 / img_height
    w = (r - l) / img_width
    h = (t - b) / img_height

    yolo_gt = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"

    return yolo_gt