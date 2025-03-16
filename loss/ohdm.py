import tensorflow as tf
from tf_keras import losses


def face_cls_loss(face_cls_predicts: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    # cls_ofdm using top 70% error samples as hard ones to calculate loss

    # postives and negitves
    cls_ids = tf.where(tf.equal(labels, 0)|tf.equal(labels, 1))
    cls_ids = tf.squeeze(cls_ids, axis=-1)

    valid_cls = tf.gather(face_cls_predicts, cls_ids)
    valid_labels = tf.gather(labels, cls_ids)
    cls_loss = losses.sparse_categorical_crossentropy(valid_labels, valid_cls)
    keep_num = int(valid_labels.shape[0] * 0.7)
    topk_hard_loss, _ = tf.math.top_k(cls_loss, keep_num)
    face_cls_ohdm = tf.reduce_mean(topk_hard_loss)

    return face_cls_ohdm


def face_cls_loss_blc(face_cls_predicts: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    
    valid_ids = tf.where(tf.equal(labels, 0) | tf.equal(labels, 1))
    valid_ids = tf.squeeze(valid_ids, axis=-1)
    valid_preds = tf.gather(face_cls_predicts, valid_ids)
    valid_labels = tf.gather(labels, valid_ids)

    cls_ce = losses.sparse_categorical_crossentropy(valid_labels, valid_preds)

    pos_ids = tf.where(tf.equal(labels, 1))
    pos_ids = tf.squeeze(pos_ids, axis=-1)

    neg_ids = tf.where(tf.equal(labels, 0))
    neg_ids = tf.squeeze(neg_ids, axis=-1)

    pos_ce = tf.gather(cls_ce, pos_ids)
    neg_ce = tf.gather(cls_ce, neg_ids)

    keep_num = min(pos_ce.shape[0], neg_ce.shape[0])

    top_neg, _ = tf.math.top_k(neg_ce, keep_num)

    cls_ohdm = tf.concat([pos_ce, top_neg], axis=0)

    return tf.reduce_mean(cls_ohdm)



def boxes_reg_loss(predicts: tf.Tensor, grand_truth: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:

    # partials and positives
    box_ids = tf.where(tf.equal(labels, 1)|tf.equal(labels, 2))
    box_ids = tf.squeeze(box_ids, axis=-1)

    valid_boxes_gt = tf.gather(grand_truth, box_ids)
    valid_boxes_preds = tf.gather(predicts, box_ids)
    box_mse = losses.mean_squared_error(valid_boxes_gt, valid_boxes_preds)

    box_reg_loss = tf.reduce_mean(box_mse)
    
    return box_reg_loss


def landmarks_loc_loss(predicts: tf.Tensor, grand_truth: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    # partial and positives
    landmark_ids = tf.where(tf.equal(labels, 3))
    landmark_ids = tf.squeeze(landmark_ids, axis=-1)


    valid_landmarks_preds = tf.gather(predicts, landmark_ids)
    valid_landmarks_gt = tf.gather(grand_truth, landmark_ids)
    landmakrs_mse = losses.mean_squared_error(valid_landmarks_gt, valid_landmarks_preds)
 
    
    land_loss = tf.reduce_mean(landmakrs_mse)    

    return land_loss
    

def pnet_loss(
    face_cls: tf.Tensor,
    boxes_reg: tf.Tensor,
    labels: tf.Tensor,
    bboxes: tf.Tensor
) -> tf.Tensor:

    face_cls = tf.squeeze(face_cls, (1, 2))
    boxes_reg = tf.squeeze(boxes_reg, (1, 2))

    face_cls_ohdm = face_cls_loss_blc(face_cls, labels)
    box_reg_loss = boxes_reg_loss(boxes_reg, bboxes, labels)
    
    return face_cls_ohdm + 0.5 * box_reg_loss
    


def rnet_loss(
    face_cls: tf.Tensor,
    boxes_reg: tf.Tensor,
    labels: tf.Tensor,
    bboxes: tf.Tensor
) -> tf.Tensor:

    face_cls_ohdm = face_cls_loss_blc(face_cls, labels)
    box_reg_loss = boxes_reg_loss(boxes_reg, bboxes, labels)

    return face_cls_ohdm + 0.5 * box_reg_loss



def onet_loss(
    face_cls: tf.Tensor,
    boxes_reg: tf.Tensor,
    landmarks_loc: tf.Tensor,
    labels: tf.Tensor,
    bboxes: tf.Tensor,
    landmarks: tf.Tensor
) -> tf.Tensor:

    face_cls_ohdm = face_cls_loss(face_cls, labels)
    box_reg_loss = boxes_reg_loss(boxes_reg, bboxes, labels)
    landmarks_loss = landmarks_loc_loss(landmarks_loc, landmarks, labels)

    return face_cls_ohdm + 0.5 * box_reg_loss + landmarks_loss

