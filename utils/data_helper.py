import tensorflow as tf
import numpy as np
import cv2


def image_augmentation(inputs: tf.Tensor) -> tf.Tensor:
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)

    return inputs


def img_normalize(img: tf.Tensor) -> tf.Tensor:
    center= 127.5
    scaler = 1.0 / 127.5
    img = (tf.cast(img, tf.float32) - center) * scaler
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    center= 127.5
    scaler = 1.0 / 127.5
    img = (img - center) * scaler
    return img


def IoU(pr_box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: np.ndarray , shape (4, ): x1, y1, x2, y2
        input box
    boxes: np.ndarray, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: np.ndarray, shape (n, )
        IoU
    """
    x1, y1, x2, y2 = pr_box
    pr_area = (x2 - x1 + 1) * (y2 - y1 + 1)

    x1s = gt_boxes[:, 0]
    y1s = gt_boxes[:, 1]
    x2s = gt_boxes[:, 2]
    y2s = gt_boxes[:, 3]
    areas = (x2s - x1s + 1) * (y2s - y1s + 1)

    # calculate inters
    xx1 = np.maximum(x1, x1s)
    yy1 = np.maximum(y1, y1s)
    xx2 = np.minimum(x2, x2s)
    yy2 = np.minimum(y2, y2s)

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inters = w * h

    unions = areas + pr_area - inters

    ious = inters / unions

    return ious


def bbox_format(bboxes: np.ndarray, src_fmt: str = "LTWH", dst_fmt: str = "LTRB") -> np.ndarray:
    if src_fmt == dst_fmt:
        return bboxes
    if src_fmt == "LTWH" and dst_fmt == "LTRB":
        left = bboxes[:, 0]
        top = bboxes[:, 1]
        width = bboxes[:, 2]
        height = bboxes[:, 3]
        right = left + width - 1
        bottom = top + height - 1
        new_bboxes = np.stack([left, top, right, bottom], axis=1)
    elif src_fmt == "LTRB" and dst_fmt == "LTWH":
        left = bboxes[:, 0]
        top = bboxes[:, 1]
        right = bboxes[:, 2]
        bottom = bboxes[:, 3]
        width = right - left + 1
        height = bottom - top + 1
        new_bboxes = np.stack([left, top, width, height], axis=1)

    return new_bboxes


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def encode_img2bytes(img, ext=".jpg") -> bytes:
    _, img_encode = cv2.imencode(ext, img)
    img_bytes = img_encode.tobytes()
    return img_bytes


def make_tfrecord_example(
    image: np.ndarray,
    label: int,
    bbox: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0]]),
    landmarks: np.ndarray | None = None,
) -> bytes:
    sample = {
        "image": _bytes_feature(encode_img2bytes(image)),
        "label": _int64_feature(label),
        "bbox": _bytes_feature(bbox.tobytes()),
    }
    if landmarks is not None:
        sample["landmarks"] = _bytes_feature(landmarks.tobytes())
    features = tf.train.Features(feature=sample)
    tf_example = tf.train.Example(features=features)
    example = tf_example.SerializeToString()
    return example


def parse_fn_pnet(example_proto):
    shape = ()
    sample_desc = {
        "image": tf.io.FixedLenFeature(shape, tf.string),
        "label": tf.io.FixedLenFeature(shape, tf.int64),
        "bbox": tf.io.FixedLenFeature(shape, tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, sample_desc)
    image = tf.io.decode_image(parsed_features["image"])
    label = parsed_features["label"]
    bbox = tf.io.decode_raw(parsed_features["bbox"], tf.float64)

    return {"image": image, "label": label, "bbox": bbox}


def parse_fn_rnet(example_proto):
    shape = ()
    sample_desc = {
        "image": tf.io.FixedLenFeature(shape, tf.string),
        "label": tf.io.FixedLenFeature(shape, tf.int64),
        "bbox": tf.io.FixedLenFeature(shape, tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, sample_desc)
    image = tf.io.decode_image(parsed_features["image"])
    label = parsed_features["label"]
    bbox = tf.io.decode_raw(parsed_features["bbox"], tf.float64)

    return {"image": image, "label": label, "bbox": bbox}


def parse_fn_onet(example_proto):
    shape = ()
    sample_desc = {
        "image": tf.io.FixedLenFeature(shape, tf.string),
        "label": tf.io.FixedLenFeature(shape, tf.int64),
        "bbox": tf.io.FixedLenFeature(shape, tf.string),
        "landmarks": tf.io.FixedLenFeature(shape, tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, sample_desc)
    image = tf.io.decode_image(parsed_features["image"])
    label = parsed_features["label"]
    bbox = tf.io.decode_raw(parsed_features["bbox"], tf.float64)
    landmarks = tf.io.decode_raw(parsed_features["landmarks"], tf.float64)

    return {"image": image, "label": label, "bbox": bbox, "landmarks": landmarks}
