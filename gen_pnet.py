import cv2
import numpy.random as rnd
import tensorflow as tf
import numpy as np
from typing import Tuple
from pathlib import Path
from utils.data_helper import IoU, make_tfrecord_example, bbox_format
import os
from tqdm import tqdm




def gen_random_neg_samples(
        image: np.ndarray, height: int, width: int, crop_size: int, gt_boxes: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter, total_neg_num: int) -> int:

    neg_num = 0

    while neg_num < 50:
        
        size = rnd.randint(crop_size, min(width, height) // 2)
        
        nx = rnd.randint(0, width - size)
        ny = rnd.randint(0, height - size)

        crop_box = np.array([nx, ny, nx + size, ny + size])

        ious = IoU(crop_box, gt_boxes)

        if ious.max() < 0.3:
            cropped_img = image[ny: ny + size, nx: nx + size, :]
            resized_cropped = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

            neg_num += 1

            sample = make_tfrecord_example(resized_cropped, label=0)

            tfrecord_writer.write(sample)

            total_neg_num += 1

    return total_neg_num
            


def gen_samples_has_overlap_with_gt(
        image: np.ndarray, height: int, width: int, crop_size: int, gt_boxes: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter, total_neg_num: int, total_pos_num: int, total_part_num: int) -> Tuple[int, int, int]:


    for box in gt_boxes:
        
        x1, y1, x2, y2 = box

        bw = x2 - x1 + 1
        bh = y2 - y1 + 1

        if max(bw, bh) < 40:
            continue
        if bw < 0:
            continue
        if bh < 0:
            continue
        if x1 < 0:
            continue
        if y1 < 0:
            continue
        
        for _ in range(5):
            size = rnd.randint(crop_size, min(height, width) // 2)

            # delta_x & delta_y are the offsets of (x1, y1) respectively
            delta_x = rnd.randint(max(-size, -x1), bw)
            delta_y = rnd.randint(max(-size, -y1), bh)

            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))

            if nx1 + size > width:
                continue
            if ny1 + size > height:
                continue

            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])

            ious = IoU(crop_box, gt_boxes)

            if ious.max() < 0.3:
                cropped_image = image[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_cropped = cv2.resize(cropped_image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                sample = make_tfrecord_example(resized_cropped, 0)
                tfrecord_writer.write(sample)

                total_neg_num += 1


        for _ in range(20):
            size = rnd.randint(int(min(bw, bh) * 0.8), np.ceil(max(bw, bh) * 1.25))

            # delta here is the offset of box center
            delta_x = rnd.randint(-bw * 0.2, bw * 0.2)
            delta_y = rnd.randint(-bh * 0.2, bh * 0.2)

            nx1 = int(max(x1 + bw / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + bh / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx1 > width:
                continue
            if ny1 > height:
                continue

            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            offsets = np.array([offset_x1, offset_y1, offset_x2, offset_y2])

            cropped_image = image[ny1: ny2, nx1: nx2, :]
            resized_cropped = cv2.resize(cropped_image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

            box_ = np.reshape(box, (1, -1))
            iou = IoU(crop_box, box_)

            if iou >= 0.65:
                sample = make_tfrecord_example(resized_cropped, 1, offsets)
                tfrecord_writer.write(sample)
                total_pos_num += 1

            elif iou >= 0.4:
                sample = make_tfrecord_example(resized_cropped, 2, offsets)
                tfrecord_writer.write(sample)
                total_part_num += 1


    return total_neg_num, total_pos_num, total_part_num






def gen_samples(
    annotation_file_dir: str | Path,
    image_path: str | Path,
    tfrecord_writer: tf.io.TFRecordWriter,
) -> None:
    
    total_neg_num = 0
    total_pos_num = 0
    total_part_num = 0

    crop_size = 12

    pbar = tqdm(total=12880)
    with open(annotation_file_dir) as rf:
        read_iter = iter(rf.readline, "")
        while read_iter:
            try:
                file_name = next(read_iter).strip()
                img_path = str(Path(image_path, file_name))

                image = cv2.imread(img_path)

                height, width, _ = image.shape
                box_num = eval(next(read_iter))

                boxes = []
                for _ in range(box_num):
                    anno = list(map(float, next(read_iter).strip().split()))
                    box = anno[:4]  # xywh

                    boxes.append(box)
                gt_boxes = np.array(boxes)
                gt_boxes = bbox_format(gt_boxes)

                total_neg_num = gen_random_neg_samples(image, height, width, crop_size, gt_boxes, tfrecord_writer, total_neg_num)
                
                total_neg_num, total_pos_num, total_part_num = gen_samples_has_overlap_with_gt(
                    image, height, width, crop_size, gt_boxes, tfrecord_writer,
                    total_neg_num, total_pos_num, total_part_num
                )

                pbar.update(1)

            except StopIteration:
                print(
                    "Finished traversing the file, we are at the end of read iteration"
                )
                break

    
    print(f"Generated {total_neg_num} negative samples in total.")
    print(f"Generated {total_pos_num} positive samples in total.")
    print(f"Generated {total_part_num} partial samples in total.")


annotations_train = Path(
    "data", "WIDER_FACE", "wider_face_split", "wider_face_train_bbx_gt.txt"
)


img_path = "data/WIDER_FACE/WIDER_train/images"

record_path = "data/pnet_records/"
if not os.path.exists(record_path):
    os.makedirs(record_path)

writer = tf.io.TFRecordWriter(path=f"{record_path}/training_data")

gen_samples(
    annotation_file_dir=annotations_train, image_path=img_path, tfrecord_writer=writer
)
