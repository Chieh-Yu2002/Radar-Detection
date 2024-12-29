import os
import numpy as np
import shutil
import json
import re

def gen_boundingbox(bbox, angle):
    theta = np.deg2rad(-angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    points = np.array([[bbox[0], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                       [bbox[0], bbox[1] + bbox[3]]]).T

    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    T = np.array([[cx], [cy]])

    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)

    min_x = np.min(points[0, :])
    min_y = np.min(points[1, :])
    max_x = np.max(points[0, :])
    max_y = np.max(points[1, :])

    return min_x, min_y, max_x, max_y

def gen_boundingbox_yolov8(bbox, angle, img_width, img_height):
    min_x, min_y, max_x, max_y = gen_boundingbox(bbox, angle)
    x_center = ((min_x + max_x) / 2) / img_width
    y_center = ((min_y + max_y) / 2) / img_height
    width = (max_x - min_x) / img_width
    height = (max_y - min_y) / img_height
    return x_center, y_center, width, height

def get_radar_dicts(folders, root_dir):
    dataset_dicts = []
    idd = 0
    frame_nums = 0
    loss_files = {}
    
    for folder in folders:
        frame_nums += len(os.listdir(os.path.join(root_dir, folder, 'Navtech_Cartesian')))
        radar_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
        annotation_path = os.path.join(root_dir, folder, 'annotations', 'annotations.json')
        
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)

        radar_files = os.listdir(radar_folder)
        radar_files.sort()
        
        for frame_number in range(len(radar_files)):
            record = {}
            objs = []
            bb_created = False
            idd += 1
            filename = os.path.join(radar_folder, radar_files[frame_number])

            if not os.path.isfile(filename):
                print(filename)
                continue
                
            record["file_name"] = folder + radar_files[frame_number]
            record["image_id"] = idd
            record["height"] = 1152
            record["width"] = 1152
            record["annotations"] = objs

            for object in annotation:
                try:
                    if object['bboxes'][frame_number]:
                        class_obj = object['class_name']
                        if 'pedestrian' in class_obj:
                            print(filename, class_obj, object['bboxes'][frame_number])
                            continue
                        position = object['bboxes'][frame_number]['position']
                        rotation = object['bboxes'][frame_number]['rotation']
                    
                        x_center, y_center, width, height = gen_boundingbox_yolov8(position, rotation, 1600, 1600)
                        obj = {
                            "bbox": [x_center, y_center, width, height],
                            "category_id": 0,
                        }
                        objs.append(obj)
                        bb_created = True
                except:
                    loss_files.update({folder + radar_files[frame_number][:-3]+'txt': idd})
            
            if bb_created:
                dataset_dicts.append(record)

    print(f"Total frames: {frame_nums}")
    return dataset_dicts, loss_files

def main():
    # Set up directories
    save_dir = './yolov8_dataset'
    train_dir = './train_data'
    test_dir = './test_data'
    
    # Copy training images
    folders = os.listdir(train_dir)
    folders.sort()
    for folder in folders:
        radar_folder = os.path.join(train_dir, folder, 'Navtech_Cartesian')
        radar_files = os.listdir(radar_folder)
        radar_files.sort()
        for radar_file in radar_files:
            if radar_file.endswith('.png'):
                shutil.copy(
                    os.path.join(radar_folder, radar_file),
                    os.path.join(save_dir, 'images', 'train', folder+radar_file)
                )

    # Copy validation images
    folders = os.listdir(test_dir)
    folders.sort()
    for folder in folders:
        radar_folder = os.path.join(test_dir, folder, 'Navtech_Cartesian')
        radar_files = os.listdir(radar_folder)
        radar_files.sort()
        for radar_file in radar_files:
            if radar_file.endswith('.png'):
                shutil.copy(
                    os.path.join(radar_folder, radar_file),
                    os.path.join(save_dir, 'images', 'val', folder+radar_file)
                )

    # Process training data
    folders = os.listdir(train_dir)
    data_dict, loss_files = get_radar_dicts(folders, root_dir=train_dir)

    # Create label files
    for data in data_dict:    
        label_file = os.path.join(save_dir, 'labels', 'train', data['file_name'][:-3]+'txt')
        with open(label_file, 'w') as file:
            for obj in data['annotations']:
                class_id = obj['category_id']
                x_center, y_center, width, height = obj['bbox']
                file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Handle missing files
    for data in loss_files.keys():
        # Create empty label file
        empty_file = os.path.join(save_dir, 'labels', 'train', data)
        with open(empty_file, 'w') as file:
            pass
        
        # Remove corresponding image
        image_file = os.path.join(save_dir, 'images', 'train', data[:-3]+'png')
        if os.path.exists(image_file):
            os.remove(image_file)

    # Clean up validation files
    val_labels = os.listdir(os.path.join(save_dir, 'labels', 'val'))
    val_images = os.listdir(os.path.join(save_dir, 'images', 'val'))
    
    for data in val_images:
        if data[:-3]+'txt' not in val_labels:
            os.remove(os.path.join(save_dir, 'images', 'val', 'rain_4_'+data[7:-3]+'png'))

if __name__ == "__main__":
    main()
