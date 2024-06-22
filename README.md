# Radar-Detection
Self-Driving Cars Project for NYCU Courses, Fall 2023

## Objectives
Used existing radar images to train a detector with the goal of recognizing various vehicles in the images (e.g., trucks, minibuses, bicycles, buses)

## Bonus
Verified model performance using data from Guangfu Road, Hsinchu, Taiwan

## Complete File Structure - vehicle_detection
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
├── weights_to_json.py
├── yolov8l.pt
├── yolov8m.pt
├── yolov8n.pt
├── yolov8s.pt
└── yolov8x.pt

```


