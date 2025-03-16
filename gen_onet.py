import numpy as np
import numpy.random as rnd
import cv2
from pathlib import Path
import tensorflow as tf
import os
from tqdm import tqdm
from typing import Tuple
import random

from models.mtcnn import MTCNN

from utils.data_helper import IoU, make_tfrecord_example, bbox_format



def gen_samples(
        annatation_dir: str|Path, 
        image_path: str|Path, 
        tfrecord_path: str|Path,
        pnet_weight_path: str|Path,
        rnet_weight_path: str|Path,
        landmarks_anno_path: str|Path,
        landmarks_gtboxes_anno_path: str|Path,
        landmarks_image_path: str|Path
    ):

    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    writer = tf.io.TFRecordWriter(path = str(Path(tfrecord_path, "training_data")))

    total_neg_num = 0
    total_pos_num = 0
    total_part_num = 0
    total_landm_num = 0

    # init_model
    mtcnn = MTCNN(pnet_weight_path, rnet_weight_path, None)
    
    pbar = tqdm(total=12880)
    with open(annatation_dir) as rf:
        read_iter = iter(rf.readline, "")
        img_num = 0
        while read_iter:
            try:
                file_name = next(read_iter).strip()
                img_num += 1
                img_path = str(Path(image_path, file_name))

                image = cv2.imread(img_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                box_num = eval(next(read_iter))

                boxes = []
                for _ in range(box_num):
                    anno = list(map(float, next(read_iter).strip().split()))
                    box = anno[:4]  # xywh
                    boxes.append(box)
                trueboxes = np.array(boxes)
                trueboxes = bbox_format(trueboxes) # xyxy

                neg_num, pos_num, part_num = gen_samples_per_image(mtcnn, image, trueboxes, writer)

                total_neg_num += neg_num
                total_pos_num += pos_num
                total_part_num += part_num

            
                pbar.update(1)
                
            except StopIteration:
                print(
                    "Finished traversing the file, we are at the end of read iteration"
                )
                break
    
    total_landm_num = gen_landmarks_samples(mtcnn, landmarks_anno_path, landmarks_gtboxes_anno_path, landmarks_image_path, writer)

    print(f"Generated {total_neg_num} negative samples in total.")
    print(f"Generated {total_pos_num} positive samples in total.")
    print(f"Generated {total_part_num} partial samples in total.")
    print(f"Generated {total_landm_num} landmarks samples in total.")






def gen_samples_per_image(
        mtcnn: MTCNN, 
        image: np.ndarray,
        gt_boxes: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> Tuple[int, int, int]:

    onet_input_size = 48

    height, width = image.shape[:2]
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    proposals = mtcnn.pnet_stage(img_rgb)

    if proposals.shape[0] == 0:
        return 0, 0, 0
    
    refined = mtcnn.rnet_stage(img_rgb, proposals)

    if refined.shape[0] == 0:
        return 0, 0, 0
    
    refined_boxes = refined.numpy()

    neg_num = 0
    pos_num = 0
    part_num = 0

    neg_samples = []
    pos_samples = []
    part_samples = []

    for box in refined_boxes:
        
        x1, y1, x2, y2 = [box[i].astype(int) for i in range(4)]
        bh = y2 - y1 + 1
        bw = x2 - x1 + 1
        if min(bh, bw) < 40:
            continue
        if x1 < 0:
            continue
        if y1 < 0:
            continue
        if x2 > width - 1:
            continue
        if y2 > height - 1:
            continue

        cropped = image[y1:y2 + 1, x1:x2 + 1, :]
        resized_cropped = cv2.resize(cropped, (onet_input_size, onet_input_size), interpolation=cv2.INTER_LINEAR)

        ious = IoU(box, gt_boxes)
        max_iou = np.max(ious)

        epsilon = 1e-9
        landmarks = np.array([epsilon] * 10)

        if max_iou < 0.3:
            # negative
            label = 0
            neg_sample = make_tfrecord_example(resized_cropped, label, landmarks=landmarks)
            neg_samples.append(neg_sample)
            neg_num += 1

        else:

            gt_box_idx = np.argmax(ious)
            [bx1, by1, bx2, by2] = gt_boxes[gt_box_idx]

            offsets = get_box_offsets(bx1, by1, bx2, by2, x1, y1, x2, y2, bw, bh)

            if max_iou >= 0.65:
                # positive
                label = 1
                pos_sample = make_tfrecord_example(resized_cropped, label, bbox=offsets, landmarks=landmarks)
                pos_samples.append(pos_sample)
                pos_num += 1

            elif max_iou >= 0.4:
                # partial
                label = 2
                part_sample = make_tfrecord_example(resized_cropped, label, bbox=offsets, landmarks=landmarks)
                part_samples.append(part_sample)
                part_num += 1

    # prevent # Prevent excessive negative samples
    if neg_num > 4 * pos_num:
        neg_samples = random.sample(neg_samples, k=3*pos_num)
        neg_num = 3 * pos_num

    if part_num > 2 * pos_num:
        part_samples = random.sample(part_samples, k=2*pos_num)
        part_num = part_num

    # ensuring Neg:Pos:Part = 3:1:1
    samples = neg_samples + pos_samples + part_samples 
    random.shuffle(samples)

    for sample in samples:
        tfrecord_writer.write(sample)

    return neg_num, pos_num, part_num


       


def is_valid_pts(pts: np.ndarray, height: int, width: int) -> bool:
    for pt in pts:
        x, y = pt[0], pt[1]
        if x < 0 or x > width - 1:
            return False
        if y < 0 or y > height - 1:
            return False
    return True



def is_valid_gt_box(box, height, width):
    x1, y1, w, h = [int(box[i]) for i in range(4)]
    x2 = x1 + w - 1
    y2 = y1 + h - 1

    if max(w, h) < 40 or min(w, h) <= 0:
        return False
    if x1 < 0 or y1 < 0:
        return False
    if x2 > width - 1 or y2 > height - 1:
        return False
    return True





def is_valid_box(box, height, width):
    x1, y1, x2, y2 = [box[i].astype(int) for i in range(4)]
    if x1 < 0 or y1 < 0:
        return False
    if x2 > width - 1 or y2 > width - 1:
        return False
    return True





def gen_landmarks_samples(
        mtcnn: MTCNN,
        landmarks_anno_path: str|Path,
        landmarks_gtboxes_anno_path: str|Path,
        landmarks_image_path: str|Path,
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> int:

    total_landmarks_num = 0


    with open(landmarks_anno_path) as lrf, open(landmarks_gtboxes_anno_path) as brf:
        ld_iter = iter(lrf.readline, "")
        bx_iter = iter(brf.readline, "")
        skips = 2
        for i in range(skips):
            next(ld_iter)
            next(bx_iter)

        count = 0
        pbar = tqdm(total=202599)
        while ld_iter and bx_iter:
            try:
                ld_info = next(ld_iter).strip().split()
                bx_info = next(bx_iter).strip().split()
                if ld_info[0] != bx_info[0]:
                    print(f"Checkout {count+2} lines of landmark_anno file and box_anno file")
                    break

                img_id = ld_info[0]

                img_path = str(Path(landmarks_image_path, img_id))
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
                
                pts = np.array(list(map(float, ld_info[1:])))
                pts = np.reshape(pts, (5, 2))

                if not is_valid_pts(pts, height, width):
                    continue

                box = np.array(list(map(float, bx_info[1:])))

                if not is_valid_gt_box(box, height, width):
                    continue
                
                total_landmarks_num += gen_landmarks_samples_per_image(mtcnn, img, height, width, box, pts, tfrecord_writer)

                count += 1

                pbar.update(1)
            except StopIteration:
                print(
                    "Finished traversing the file, we are at the end of read iteration"
                )
                
                break

    return total_landmarks_num






def gen_landmarks_samples_per_image(
        mtcnn: MTCNN,
        image: np.ndarray,
        height: int,
        width: int,
        gt_box: np.ndarray, 
        landmarks: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> int:

    onet_input_size = 48

    landm_num = 0

    landm_num += gen_from_ground_truth(image, height, width, onet_input_size, gt_box, landmarks, tfrecord_writer)

    landm_num += gen_from_random_shift(image, height, width, onet_input_size, gt_box, landmarks, tfrecord_writer)

    # landm_num += gen_from_detection(mtcnn, image, height, width, onet_input_size, gt_box, landmarks, tfrecord_writer)

    return landm_num




def crop_and_resize(image, crop_size, crop_shape):
    x1, y1, x2, y2 = [int(crop_shape[i]) for i in range(4)]
    cropped = image[y1: y2+1, x1: x2+1, :]
    cropped_resized = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    
    return cropped_resized





def get_box_offsets(x1, y1, x2, y2, ref_x1, ref_y1, ref_x2, ref_y2, scaling_w, scaling_h):
    x1_offset = (x1 - ref_x1) / float(scaling_w)
    y1_offset = (y1 - ref_y1) / float(scaling_h)
    x2_offset = (x2 - ref_x2) / float(scaling_w)
    y2_offset = (y2 - ref_y2) / float(scaling_h)

    offsets = np.array([x1_offset, y1_offset, x2_offset, y2_offset])

    return offsets





def get_landmarks_offsets(
        landmarks: np.ndarray, # shape [n, 2] 
        ref_x: int,
        ref_y: int,
        scaling_w: int,
        scaling_h: int
    ) -> np.ndarray:
    
    pts_x = landmarks[:, 0]
    pts_y = landmarks[:, 1]

    pts_x_offsets = (pts_x - ref_x) / float(scaling_w)
    pts_y_offsets = (pts_y - ref_y) / float(scaling_h)

    landmarks_offsets = np.stack([pts_x_offsets, pts_y_offsets], axis=1)
    landmarks_offsets = np.reshape(landmarks_offsets, (1, -1))

    return landmarks_offsets






def gen_from_ground_truth(
        image: np.ndarray, 
        height: int, 
        width: int,
        crop_size: int,
        gt_box: np.ndarray, 
        landmarks: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> int:

    if not is_valid_gt_box(gt_box, height, width):
        return 0
    
    x1, y1, bw, bh = [gt_box[i] for i in range(4)] 
    x2 = x1 + bw - 1
    y2 = y1 + bh - 1

    crop_shape = [x1, y1, x2, y2]
    cropped_resized = crop_and_resize(image, crop_size, crop_shape)

    ldm_offsets = get_landmarks_offsets(landmarks, x1, y1, bw, bh)

    label = 3
    epsilon = 1e-9
    box_offset = np.array([epsilon, epsilon, epsilon, epsilon])

    sample = make_tfrecord_example(cropped_resized, label, bbox=box_offset, landmarks=ldm_offsets)
    tfrecord_writer.write(sample)
    
    return 1



def gen_from_detection(
        mtcnn: MTCNN,
        image: np.ndarray, 
        height: int,
        width: int,
        crop_size: int,
        gt_box: np.ndarray, 
        landmarks: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> int:
    
    x1, y1, w, h = [gt_box[i].astype(int) for i in range(4)]
    
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    
    if max(w, h) < 40 or min(w, h) <= 0:
        return 0
    if x1 < 0 or y1 < 0:
        return 0
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    proposals = mtcnn.pnet_stage(img_rgb)
    if proposals.shape[0] == 0:
        return 0
    
    refined = mtcnn.rnet_stage(img_rgb, proposals)
    if refined.shape[0] == 0:
        return 0
    
    gt_box = np.reshape(gt_box, (1, -1))
    gt_box = bbox_format(gt_box)
    
    refined_boxes = refined.numpy()
    
    landm_num = 0

    for proposal in refined_boxes:
        nx1, ny1, nx2, ny2 = proposal

        bw = nx2 - nx1 + 1
        bh = ny2 - ny1 + 1

        if min(bw, bh) <= 0 or max(bw, bh) < 40:
            continue
        
        if not is_valid_box(proposal, height, width):
            continue

        iou = IoU(proposal, gt_box)

        if iou >= 0.65:
            # crop and resize
            crop_shape = [nx1, ny1, nx2, ny2]
            cropped_resized = crop_and_resize(image, crop_size, crop_shape)

            # offset
            landmarks_offsets = get_landmarks_offsets(landmarks, nx1, ny1, bw, bh)

            offsets = get_box_offsets(x1, y1, x2, y2, nx1, ny1, nx2, ny2, bw, bh)

            label = 3

            landm_sample = make_tfrecord_example(cropped_resized, label, offsets, landmarks_offsets)
            
            tfrecord_writer.write(landm_sample)

            landm_num += 1

    return landm_num
     





def gen_from_random_shift(
        image: np.ndarray, 
        height: int,
        width: int,
        crop_size: int,
        gt_box: np.ndarray, 
        landmarks: np.ndarray, 
        tfrecord_writer: tf.io.TFRecordWriter
    ) -> int:
    
    x1, y1, w, h = gt_box

    x2 = x1 + w - 1
    y2 = y1 + h - 1

    if max(w, h) < 40 or min(w, h) <= 0:
        return 0
    if x1 < 0 or y1 < 0:
        return 0
    
    gt = np.reshape(gt_box, (1, -1))
    gt = bbox_format(gt)

    landm_num = 0

    for _ in range(5):
        # random box size
        box_size = rnd.randint(int(min(w, h) * 0.8), np.ceil(max(w, h) * 1.25))
        delta_x = rnd.randint(int(-w * 0.2), int(w * 0.2))
        delta_y = rnd.randint(int(-h * 0.2), int(h * 0.2))

        nx1 = max(int(x1 + w / 2 - box_size / 2 + delta_x), 0)
        ny1 = max(int(y1 + h / 2 - box_size / 2 + delta_y), 0)

        nx2 = nx1 + box_size - 1
        ny2 = ny1 + box_size - 1


        crop_box = np.array([nx1, ny1, nx2, ny2])

        if not is_valid_box(crop_box, height, width):
            continue

        iou = IoU(crop_box, gt)

        if iou >= 0.65:
            # crop and resize
            crop_shape = [nx1, ny1, nx2, ny2]
            cropped_resized = crop_and_resize(image, crop_size, crop_shape)

            # offset
            landmarks_offsets = get_landmarks_offsets(landmarks, nx1, ny1, box_size, box_size)

            offsets = get_box_offsets(x1, y1, x2, y2, nx1, ny1, nx2, ny2, box_size, box_size)

            label = 3

            landm_sample = make_tfrecord_example(cropped_resized, label, offsets, landmarks_offsets)
            
            tfrecord_writer.write(landm_sample)

            landm_num += 1

    return landm_num







annotations_train = Path("data", "WIDER_FACE", "wider_face_split", "wider_face_train_bbx_gt.txt")

img_path = "data/WIDER_FACE/WIDER_train/images"

tfrecord_path = Path("data", "onet_records")

landmarks_anno_path = "data/Celeba/Anno/list_landmarks_celeba.txt"
landmarks_gtboxes_anno_path = "data/Celeba/Anno/list_bbox_celeba.txt"
landmarks_image_path = "data/Celeba/Celeba_origins/img_celeba"

pnet_weight_path = "weights/pnet/30.keras"
rnet_weight_path = "weights/rnet/60.keras"

gen_samples(annotations_train, img_path, tfrecord_path, pnet_weight_path, rnet_weight_path, 
    landmarks_anno_path, landmarks_gtboxes_anno_path, landmarks_image_path)