import os
import json
from ultralytics import YOLO

def process_epoch(epoch_num, image_directory):
    """
    Process predictions for a single epoch and save to JSON
    """
    # Load model for this epoch
    model = YOLO(f'C:/lcy/2023_final/vehicle_detection/runs/detect/train6/weights/epoch{epoch_num}.pt')
    sequence_name = f'yolov8l_bonus_{epoch_num}'
    output_file_path = sequence_name + "_pred.json"
    
    predictions = []
    
    # Get all images in sorted order
    img_paths = sorted(os.listdir(image_directory))
    
    # Process each image
    for filename in img_paths:
        if not filename.endswith(".png"):
            continue
            
        # Get predictions for this image
        file_path = os.path.join(image_directory, filename)
        sample_token = filename.split('.')[0]
        results = model(file_path)
        
        # Process each detection
        for i in range(len(results[0].boxes.xyxy)):
            # Get box coordinates and confidence
            x_min, y_min, x_max, y_max = results[0].boxes.xyxy[i].cpu()
            x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
            score = float(results[0].boxes.conf[i].cpu())
            
            # Create prediction object
            coor = [
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min]
            ]
            
            predicts_object = {
                'sample_token': sample_token,
                'points': coor,
                'name': "car",
                'score': score
            }
            predictions.append(predicts_object)
    
    # Save predictions to JSON file
    with open(output_file_path, "w") as outfile:
        json.dump(predictions, outfile, indent=2)

def main():
    # Configuration
    image_directory = 'C:/lcy/2023_final/Bonus_Image'
    start_epoch = 1
    end_epoch = 141
    
    # Process each epoch
    for epoch in range(start_epoch, end_epoch):
        print(f"Processing epoch {epoch}...")
        process_epoch(epoch, image_directory)
        print(f"Completed epoch {epoch}")

if __name__ == "__main__":
    main()
