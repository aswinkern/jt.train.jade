import os
import cv2
import numpy as np
import json
import errno


class Utils:
    def __init__(self):
        pass

    def postprocess(self, raw_image, heatmap, area_threshold=500, anomaly_sum_threshold=5000, erode_ks=1):
        """
        Filter heatmap — remove small noise regions.
        Returns binary heatmap with only significant anomaly regions.
        """
        if erode_ks > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ks, erode_ks))
            heatmap = cv2.erode(heatmap, kernel)

        binary = np.where(heatmap > 0, 255, 0).astype(np.uint8)

        try:
            _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = np.zeros_like(binary)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_threshold:
                continue
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            anomaly_sum = np.sum(heatmap * mask)
            if anomaly_sum < anomaly_sum_threshold:
                continue
            cv2.drawContours(filtered, [contour], 0, 255, -1)

        return filtered

    def overlay_heatmap(self, input_image, heatmap, alpha=0.6, invert=False):
        """
        Overlay heatmap on input image.
        """
        if invert:
            heatmap = 255 - heatmap

        if len(input_image.shape) == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(input_image, alpha, heatmap_color, 1 - alpha, 0)
        return overlayed

    def get_bounding_boxes(self, image, heatmap, get_bb_image=True):
        """
        Extract bounding boxes from heatmap contours.
        Returns list of [xmin, ymin, xmax, ymax] normalized boxes.
        """
        h, w = heatmap.shape[:2]

        try:
            _, contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        bb_image = image.copy() if get_bb_image else None

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            xmin = x / w
            ymin = y / h
            xmax = (x + bw) / w
            ymax = (y + bh) / h
            boxes.append([xmin, ymin, xmax, ymax])

            if get_bb_image:
                cv2.rectangle(bb_image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        return boxes, bb_image

    def convert_pred_to_dict(self, boxes):
        """
        Convert list of boxes to detection dict.
        """
        return {
            "num_detections": len(boxes),
            "detection_boxes": boxes,
            "detection_scores": [1.0] * len(boxes),
            "detection_classes": [1] * len(boxes)
        }

    def concat_images(self, img1, img2, max_dim=2048):
        """
        Concatenate two images side by side.
        """
        if img1.max() < 2.0:
            img1 = np.uint8(255 * img1)
        if img2.max() < 2.0:
            img2 = np.uint8(255 * img2)

        max_side = max(img1.shape[:2])
        if max_side > max_dim:
            ratio = max_dim / max_side
            img1 = cv2.resize(img1, (int(img1.shape[1] * ratio), int(img1.shape[0] * ratio)))
            img2 = cv2.resize(img2, (int(img2.shape[1] * ratio), int(img2.shape[0] * ratio)))

        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        return np.concatenate([img1, img2], axis=1)

    def make_sure_path_exists(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_json(self, save_dir, pred_dict, image_name):
        self.make_sure_path_exists(save_dir)
        save_path = os.path.join(save_dir, image_name.replace(".png", ".json").replace(".jpg", ".json"))
        with open(save_path, "w") as f:
            json.dump(pred_dict, f, indent=2)
