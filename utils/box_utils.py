import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    ]   
)
def convert_to_square(bboxes: tf.Tensor):
    """
    Parameters:
        bboxes: float tensor of shape [n, 4]

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1
    w = x2 - x1
    max_side = tf.math.maximum(h, w)

    dx1 = x1 + w * 0.5 - max_side * 0.5
    dy1 = y1 + h * 0.5 - max_side * 0.5
    dx2 = dx1 + max_side
    dy2 = dy1 + max_side

    # box :  [dx1,dy1,dx1+max_sie,dy1+max_size]--> box: h = w = max_size
    return tf.stack(
        [
            tf.round(dx1),
            tf.round(dy1),
            tf.round(dx2),
            tf.round(dy2)
        ],
        axis=1
    )

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    ]   
)
def calibrate_box(bboxes: tf.Tensor, offsets: tf.Tensor):
    """
    Parameters:
        bboxes: float tensor of shape [n, 4].
        offsets: float tensor of shape [n, 4].

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1
    h = y2 - y1

    # (w,h,w,h)*delta+(x1,y1,x2,y2), each size:(n,)
    translation = tf.stack([w, h, w, h], axis=1) * offsets
    
    calibrated = bboxes + translation # e.g. x1_true = delta_x1 * (x2 - x1) + x1
    
    return calibrated


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    ]
)
def get_image_boxes(
        bboxes: tf.Tensor, 
        img: tf.Tensor, 
        height: float, 
        width: float, 
        num_boxes: int, 
        size: int=24
    ):
    """
    Parameters:
        bboxes: float tensor of shape [n, 4]
        img: image tensor
        height: float, image height
        width: float, image width
        num_boxes: int, number of rows in bboxes
        size: int, size of cutouts

    Returns:
        float tensor of shape [n, size, size, 3]
    """
    x1 = tf.math.maximum(bboxes[:, 0], 0.0) / width
    y1 = tf.math.maximum(bboxes[:, 1], 0.0) / height
    x2 = tf.math.minimum(bboxes[:, 2], width) / width
    y2 = tf.math.minimum(bboxes[:, 3], height) / height
    
    boxes = tf.stack([y1, x1, y2, x2], axis=1)  
    # see https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    img_boxes = tf.image.crop_and_resize(
        tf.expand_dims(img, 0), boxes, tf.zeros(num_boxes, dtype=tf.int32), (size, size)
    )  
    
    return img_boxes



@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    ]
)
def generate_bboxes(
        probs: tf.Tensor, 
        offsets: tf.Tensor, 
        scale: float, 
        threshold: float
    ) -> tf.Tensor:
    """
    Parameters:
        probs: float tensor of shape [b, m, 2], output of PNet
        offsets: float tensor of shape [b, m, 4], output of PNet
        scale: float, scale of the input image
        threshold: float, confidence threshold

    Returns:
        float tensor of shape [N, 9]
    """

    stride = 2.0
    cell_size = 12.0

    face_probs = probs[:, :, 1]  # shape (b, m)
    face_ids = tf.where(face_probs > threshold) # shape (N, 2)

    if face_ids.shape[0] == 0: # no face detected
        return tf.zeros((0, 9))
    
    offsets = tf.gather_nd(offsets, face_ids)

    face_scores = tf.expand_dims(tf.gather_nd(face_probs, face_ids), axis=1)
    
    face_ids = tf.cast(face_ids, tf.float32)
    pos_row = face_ids[:, 0]
    pos_col = face_ids[:, 1]
    
    x1 = tf.expand_dims(tf.round((stride * pos_col) / scale), axis=1)
    y1 = tf.expand_dims(tf.round((stride * pos_row) / scale), axis=1)
    x2 = tf.expand_dims(tf.round((stride * pos_col + cell_size) / scale), axis=1)
    y2 = tf.expand_dims(tf.round((stride * pos_row + cell_size) / scale), axis=1)

    bounding_boxes = tf.concat(
        [ x1, y1, x2, y2, face_scores, offsets ],
        axis=1
    )

    return bounding_boxes






