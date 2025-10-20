from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2
from datetime import datetime


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime
    class_id: int


class DogHumanDetector:
    DOG_CLASSES = ["dog"]
    HUMAN_CLASSES = ["person"]

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

        self.class_names = self.model.names

        self.dog_class_ids = [
            idx for idx, name in self.class_names.items()
            if name.lower() in self.DOG_CLASSES
        ]
        self.human_class_ids = [
            idx for idx, name in self.class_names.items()
            if name.lower() in self.HUMAN_CLASSES
        ]

        print(f"Initialized detector with model: {model_name}")
        print(f"Dog class IDs: {self.dog_class_ids}")
        print(f"Human class IDs: {self.human_class_ids}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        timestamp = datetime.now()

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])

                if class_id not in self.dog_class_ids + self.human_class_ids:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                detection = Detection(
                    class_name=self.class_names[class_id],
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    timestamp=timestamp,
                    class_id=class_id
                )
                detections.append(detection)

        return detections

    def detect_dogs(self, frame: np.ndarray) -> List[Detection]:
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.class_id in self.dog_class_ids]

    def detect_humans(self, frame: np.ndarray) -> List[Detection]:
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.class_id in self.human_class_ids]

    def is_dog_unsupervised(self, frame: np.ndarray) -> Tuple[bool, List[Detection], List[Detection]]:
        dogs = self.detect_dogs(frame)
        humans = self.detect_humans(frame)

        is_unsupervised = len(dogs) > 0 and len(humans) == 0
        return is_unsupervised, dogs, humans

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            if detection.class_id in self.dog_class_ids:
                color = (0, 255, 0)  # Green for dogs
            elif detection.class_id in self.human_class_ids:
                color = (255, 0, 0)  # Blue for humans
            else:
                color = (128, 128, 128)  # Gray for others

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return annotated_frame

    def get_detection_summary(self, detections: List[Detection]) -> Dict[str, int]:
        summary = {"dogs": 0, "humans": 0}

        for detection in detections:
            if detection.class_id in self.dog_class_ids:
                summary["dogs"] += 1
            elif detection.class_id in self.human_class_ids:
                summary["humans"] += 1

        return summary