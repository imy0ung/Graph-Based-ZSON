import cv2
from ultralytics import YOLO
import numpy as np
from vision_models.coco_classes import COCO_CLASSES




class YoloV8Detector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLO("./weights/yolov8x.pt")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: list
                    ):
        self.classes = classes
        # self.model.set_classes(classes)

    def detect(self,
               image: np.ndarray
               ):
        image = np.flip(image, axis=-1) # to bgr
        results = self.model(image, verbose=False)[0]
        preds = {}
        preds["boxes"] = []
        preds["scores"] = []
        preds["class_names"] = []
        preds["class_ids"] = []
        boxes = results.boxes
        for bbox, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            if conf > self.confidence_threshold:
                class_id = int(cls)
                class_name = COCO_CLASSES[class_id]
                # 모든 COCO 클래스를 탐지 (self.classes가 None이거나 모든 클래스 허용)
                if self.classes is None or class_name in self.classes:
                    preds["boxes"].append([bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
                    preds["scores"].append(conf.item())
                    preds["class_names"].append(class_name)
                    preds["class_ids"].append(class_id)

        return preds

    def detect_all(self,
                   image: np.ndarray,
                   confidence_threshold: float = None
                   ):
        """
        Detect all COCO classes (not just target class).
        Returns all detections with class names.
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        image = np.flip(image, axis=-1)  # to bgr
        results = self.model(image, verbose=False)[0]
        preds = {
            "boxes": [],
            "scores": [],
            "class_names": [],
            "class_ids": []
        }
        boxes = results.boxes
        for bbox, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            if conf > confidence_threshold:
                class_id = int(cls)
                class_name = COCO_CLASSES[class_id]
                preds["boxes"].append([bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()])
                preds["scores"].append(conf.item())
                preds["class_names"].append(class_name)
                preds["class_ids"].append(class_id)

        return preds


if __name__ == "__main__":
    from PIL import Image
    # Test the YOLO v8 Detector
    detector = YoloV8Detector(confidence_threshold=0.8)
    detector.set_classes(["bed"])
    # Load an image
    image = np.array(Image.open("/home/finn/active/MON/bed.jpeg"))

    # Detect objects in the image
    detections = detector.detect(image)


