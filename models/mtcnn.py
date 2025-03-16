import tensorflow as tf
from models.networks import Pnet, Rnet, Onet
from utils.box_utils import generate_bboxes, calibrate_box, convert_to_square, get_image_boxes
from utils.data_helper import preprocess


class MTCNN(object):
    def __init__(self,
        pnet_path: str | None,
        rnet_path: str | None,
        onet_path: str | None,
        min_face_size = 20.0, 
        thresholds = [0.5, 0.5, 0.5],
        nms_thresholds = [0.6, 0.6, 0.3],
        max_nms_output_num=300,
        scaling_factor = 0.707
    ):
        self.pnet = Pnet()
        self.pnet(tf.ones((1, 12, 12, 3)))
        if pnet_path:
            self.pnet.load_weights(pnet_path)
        self.rnet = Rnet()
        self.rnet(tf.ones((1, 24, 24, 3)))
        if rnet_path:
            self.rnet.load_weights(rnet_path)
        self.onet = Onet()
        self.onet(tf.ones((1, 48, 48, 3)))
        if onet_path:
            self.onet.load_weights(onet_path)
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.max_nms_output_num = max_nms_output_num
        self.scaling_factor = scaling_factor

    
    def detect(self, img):
        bboxes = self.p_step(img)

        if len(bboxes) == 0:
            return [], [], []
        bboxes = self.r_step(img, bboxes)

        if len(bboxes) == 0:
            return [], [], []

        bboxes, landmarks, scores = self.o_step(img, bboxes)

        if len(bboxes) == 0:
            return [], [], []

        return bboxes, landmarks, scores


    def build_scaling_pyramid(self, height, width):
        scales = []

        min_side = min(height, width)
        
        min_detection_size = 12
        m = min_detection_size / self.min_face_size
        
        min_side *= m
        factor_count = 0
        while min_side > self.min_face_size:
            scales.append(m * self.scaling_factor ** factor_count)
            min_side *= self.scaling_factor
            factor_count += 1

        return scales
    

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ]
    )
    def scale_search(self, img, height, width, scale):
        hs = tf.math.ceil(height * scale)
        ws = tf.math.ceil(width * scale)
        img_in = tf.image.resize(img, (hs, ws))
        img_in = tf.expand_dims(img_in, 0)

        probs, offsets = self.pnet(img_in)
        info = generate_bboxes(probs[0], offsets[0], scale, self.thresholds[0])
        if info.shape[0] == 0:
            return info
        bboxes = info[:, :4] 
        scores = info[:, 4]
        nms_idx = tf.image.non_max_suppression(
            bboxes, scores, self.max_nms_output_num, iou_threshold=self.nms_thresholds[0]
        )
        info = tf.gather(info, nms_idx)
        return info
    

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        ]
    )
    def bbox_alignment(self, bboxes, offsets):
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)
        return bboxes
    

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
        ]
    )
    def landmark_alignment(self, bboxes, landmarks):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1
        h = y2 - y1

        landmarks = tf.stack(
            [
                landmarks[:, 0] * w + x1,
                landmarks[:, 1] * h + y1,
                landmarks[:, 2] * w + x1,
                landmarks[:, 3] * h + y1,
                landmarks[:, 4] * w + x1,
                landmarks[:, 5] * h + y1,
                landmarks[:, 6] * w + x1,
                landmarks[:, 7] * h + y1,
                landmarks[:, 8] * w + x1,
                landmarks[:, 9] * h + y1,
            ]
        )

        landmarks = tf.transpose(landmarks)

        return landmarks


    def pnet_stage(self, img):
        img = preprocess(img)
        height, width = img.shape[:2]

        img = tf.convert_to_tensor(img, tf.float32)
        
        scales = self.build_scaling_pyramid(height, width)
        proposals = []
        for scale in scales:
            scale_proposal = self.scale_search(img, height, width, scale)
            proposals.append(scale_proposal)
        proposals = tf.concat(proposals, axis=0)

        bboxes, scores, offsets = proposals[:, :4], proposals[:, 4], proposals[:, 5:]
        bboxes = self.bbox_alignment(bboxes, offsets)

        nms_ids = tf.image.non_max_suppression(
            boxes=bboxes, 
            scores=scores, 
            max_output_size=self.max_nms_output_num,
            iou_threshold=self.nms_thresholds[0]
        )

        bboxes = tf.gather(bboxes, nms_ids)

        return bboxes


    def rnet_stage(self, img, bboxes):
        img = preprocess(img)
        height, width = img.shape[:2]
        num_boxes = bboxes.shape[0]
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)

        probs, offsets = self.rnet(img_boxes)
        valid_idx = tf.argmax(probs, axis=-1) == 1

        bboxes = tf.boolean_mask(bboxes, valid_idx)

        if bboxes.shape[0] == 0:
            return tf.zeros((0, 4))
        
        
        offsets = tf.boolean_mask(offsets, valid_idx)
        scores = tf.boolean_mask(probs[:, 1], valid_idx)

        bboxes = self.bbox_alignment(bboxes, offsets)

        nms_ids = tf.image.non_max_suppression(
            boxes = bboxes,
            scores=scores,
            max_output_size=self.max_nms_output_num,
            iou_threshold=self.nms_thresholds[1]
        )

        bboxes = tf.gather(bboxes, nms_ids)

        return bboxes


    def onet_stage(self, img, bboxes):
        img = preprocess(img)
        height, width = img.shape[:2]
        num_boxes = bboxes.shape[0]
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=48)
        probs, offsets, landmarks = self.onet(img_boxes)
        
        face_scores = probs[:, 1]
        
        valid_idx = face_scores > self.thresholds[2]

        bboxes = tf.boolean_mask(bboxes, valid_idx)

        if bboxes.shape[0] == 0:
            bboxes = tf.zeros((0, 4))
            scores = tf.zeros((0,))
            landmarks = tf.zeros((0, 10))

            return bboxes, scores, landmarks

        offsets = tf.boolean_mask(offsets, valid_idx)
        scores = tf.boolean_mask(probs[:, 1], valid_idx)
        landmarks = tf.boolean_mask(landmarks, valid_idx)

        landmarks = self.landmark_alignment(bboxes, landmarks)

        bboxes = calibrate_box(bboxes, offsets)

        nms_idx = tf.image.non_max_suppression(
            bboxes,
            scores,
            self.max_nms_output_num,
            iou_threshold=self.nms_thresholds[2],
        )
        
        bboxes = tf.gather(bboxes, nms_idx)
        landmarks = tf.gather(landmarks, nms_idx)
        scores = tf.gather(scores, nms_idx)

        return bboxes, landmarks, scores

        