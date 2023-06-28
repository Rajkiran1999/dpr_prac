import os
import cv2
import matplotlib.pyplot as plt

def visualize_detections(image, boxes, confidences, ellipses, true_positives, save_path):
    image = image * 255
    image = image.astype("uint8")
    for ellipse in ellipses:
        image = cv2.ellipse(image, (ellipse[3], ellipse[4]), (ellipse[0], ellipse[1]), ellipse[2], 0, 360, (255, 0, 0),
                            thickness=2)

    for box, confidence, true_positive in zip(boxes, confidences, true_positives):
        color = (0, 0, 255)
        if true_positive == 1:
            color = (0, 255, 0)
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image, f"{confidence:.2f}", (box[0] + 7, box[1] + 20), font, 0.7, color, thickness=2)

    save_dir = os.sep.join(save_path.split(os.sep)[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(save_path, image)

def plot_pr_curve(precision, recall, average_precision):
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"AP={average_precision:0.2f}")
    plt.show()