import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
import os
from tqdm import tqdm
import random
from typing import Tuple


from models.mtcnn import MTCNN

from utils.data_helper import IoU, make_tfrecord_example, bbox_format



def gen_samples(
        annatation_dir: str|Path, 
        image_path: str|Path, 
        tfrecord_path: str|Path,
        pnet_weight_path: str|Path
    ):

    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    writer = tf.io.TFRecordWriter(path = f"{tfrecord_path}/training_data")


    total_neg_num = 0
    total_pos_num = 0
    total_part_num = 0
    # init_model
    mtcnn = MTCNN(pnet_weight_path, None, None)
    
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
                box_num = eval(next(read_iter))

                boxes = []
                for _ in range(box_num):
                    anno = list(map(float, next(read_iter).strip().split()))
                    box = anno[:4]  # xywh
                    boxes.append(box)
                trueboxes = np.array(boxes)
                gt_boxes = bbox_format(trueboxes) # xyxy

                neg_num, pos_num, part_num = gen_samples_per_image(mtcnn, image, gt_boxes, writer)

                total_neg_num += neg_num
                total_pos_num += pos_num
                total_part_num += part_num


                pbar.update(1)
                
            except StopIteration:
                print(
                    "Finished traversing the file, we are at the end of read iteration"
                )
                break
    
    print(f"Generated {total_neg_num} negative samples in total.")
    print(f"Generated {total_pos_num} positive samples in total.")
    print(f"Generated {total_part_num} partial samples in total.")
                            

def gen_samples_per_image(mtcnn: MTCNN, image: np.ndarray, ground_truth: np.ndarray, tfrecord_writer: tf.io.TFRecordWriter) -> Tuple[int, int, int]:
    # 
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    proposals = mtcnn.pnet_stage(img_rgb).numpy()
    # no face proposal
    if proposals.shape[0] == 0:
        return 0, 0, 0
    
    height, width = image.shape[:2]
    # for each proposal, if iou less than 0.3 with any ground_truth, we treat it as negative and labeled `0`
    # iou greater than 0.65 with any ground_truth, we treat it as positive
    
    rnet_input_size = 24
    
    neg_num = 0
    pos_num = 0
    part_num = 0

    neg_samples = []
    pos_samples = []
    part_samples = []

    for proposal in proposals:

        x1, y1, x2, y2 = [proposal[i].astype(int) for i in range(4)]
        bh = y2 - y1 + 1
        bw = x2 - x1 + 1
        if min(bh, bw) < 20:
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
        resized_cropped = cv2.resize(cropped, (rnet_input_size, rnet_input_size), interpolation=cv2.INTER_LINEAR)
        
        ious = IoU(proposal, ground_truth)
        max_iou = np.max(ious)
        
        if max_iou < 0.3: 
            # negtive
            label = 0
            neg_sample = make_tfrecord_example(resized_cropped, label)
            neg_samples.append(neg_sample)
            neg_num += 1

        else:
            
            true_box_idx = np.argmax(ious)
            [bx1, by1, bx2, by2] = ground_truth[true_box_idx]
                
            offset_x1 = (bx1 - x1) / float(bw)
            offset_y1 = (by1 - y1) / float(bh)
            offset_x2 = (bx2 - x2) / float(bw)
            offset_y2 = (by2 - y2) / float(bh)
            offset = np.array([offset_x1, offset_y1, offset_x2, offset_y2])

            if max_iou >= 0.65:
                # positive
                label = 1
                pos_sample = make_tfrecord_example(resized_cropped, label, bbox=offset)
                pos_samples.append(pos_sample)
                pos_num += 1


            
            elif max_iou >= 0.4:
                # partial
                label = 2
                part_sample = make_tfrecord_example(resized_cropped, label, bbox=offset)
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


 

annotations_train = Path("data", "WIDER_FACE", "wider_face_split", "wider_face_train_bbx_gt.txt")

img_path = "data/WIDER_FACE/WIDER_train/images"

tfrecord_path = Path("data", "rnet_records")

pnet_weight_path = "weights/pnet/30.keras"

gen_samples(annotations_train, img_path, tfrecord_path, pnet_weight_path)

