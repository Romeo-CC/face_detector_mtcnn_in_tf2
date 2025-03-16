import cv2

def draw_faces(img, bboxes, landmarks, scores):
    if landmarks is None:
        for box in bboxes:
            img = cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )
    else:
        for box, landmark, score in zip(bboxes, landmarks, scores):
            img = cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )
            for i in range(0, 9, 2):
                x = int(landmark[i])
                y = int(landmark[i + 1])
                img = cv2.circle(img, (x, y), 3, (0, 0, 255))

            img = cv2.putText(
                img,
                "{:.2f}".format(score),
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
            )
    
    return img
