# RnC-Final-Task
This project implements an AI-based system to automatically detect and read scoreboards in football match videos using YOLO for detection and OCR for text recognition. It can also extract event-based clips (goals, cards) and update scores in real time.
Features

Detects scoreboard region in live or recorded football videos using YOLO.

Reads scores, time, and extra information with OCR.

Saves video clips of significant events (goals, red/yellow cards).

Real-time scoreboard display with annotation overlay.

GPU-accelerated processing for faster performance.

Technologies Used

YOLO (Ultralytics): Object detection for scoreboard localization.

EasyOCR / PaddleOCR: Extracts text from detected scoreboard regions.

OpenCV: Video handling, drawing annotations, and saving clips.

Python: Integrates detection, OCR, and video processing.

CUDA / GPU: Speeds up model inference.

Label Studio: Annotating scoreboard regions for model training.

Dataset & Training

Annotation Tool: Label Studio, outputs YOLO-compatible annotations.

Classes: Single class – Scoreboard, covering multiple scoreboard styles.

Training Setup:

YOLO trained on annotated frames with GPU acceleration.

Data augmentation: rotation, crop, brightness/contrast adjustments.

Evaluated using mAP, precision, and recall metrics.
