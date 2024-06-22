# Radar-Detection
Self-Driving Cars Project for NYCU Courses, Fall 2023
# Detection
Radar Detection with YOLOv8

## File Structure - vehicle_detection
```
├── bonus_data
│   ├── images
│   │   ├── train
│   │   └──  val
│   ├── labels
│   │   ├── train
│   │   └──  val
│   └── train_data # train_data from google drive bonus folder
│       └── ...
├── runs
│   └── dectect # weights will be saved here
├── ultralytics # tools for yolo
│   └── ...
├── yolo_best # best weights and json for competition and bonus
│   ├── yolov8l_bonus_train6_best_pred.json
│   ├── yolov8l_bonus.pt
│   ├── yolov8m_.json
│   └── yolov8m_.pt
├── yolov8_dataset
│   ├── images
│   │   ├── train
│   │   └──  val
│   └── labels
│       ├── train
│       └──  val
├── all_eva #Evaluation code
├── data_process_bonus.ipynb # generate labels for traning bonus model
├── data_process.ipynb # generate labels for traning competition model
├── dataset.yaml
├── README.md
└── weights_to_json.py

```
