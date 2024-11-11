import os
import numpy as np
import shutil
import json
import re

def get_bonus_radar_dicts(image, annotations, sources, image_folder, label_folder):
    """
    Process radar image annotations and create corresponding label files
    """
    image_data = {img['id']: (img['file_name'], img['width'], img['height']) for img in image}
    image_annotations = {img_id: [] for img_id in image_data}

    for ann in annotations:
        img_id, bbox = ann['image_id'], ann['bbox']
        width, height = image_data[img_id][1], image_data[img_id][2]
        x_center, y_center = (bbox[0] + bbox[2] / 2) / width, (bbox[1] + bbox[3] / 2) / height
        bbox_width, bbox_height = bbox[2] / width, bbox[3] / height
        image_annotations[img_id].append(f"{ann['category_id']} {x_center} {y_center} {bbox_width} {bbox_height}")

    for img_id, ann_list in image_annotations.items():
        with open(os.path.join(label_folder, f"{img_id}.txt"), 'w') as file:
            file.writelines('\n'.join(ann_list))
        shutil.copy(os.path.join(sources, image_data[img_id][0]), os.path.join(image_folder, f"{img_id}.png"))

def main():
    # Define directories
    save_dir = './bonus_data'
    train_dir = './bonus_data/train_data/'
    test_dir = './bonus_data/test_data/'
    train_file = os.path.join(save_dir, 'train.txt')
    test_file = os.path.join(save_dir, 'test.txt')

    # Process annotations
    annotations_file = './bonus_data/train_data/train/annotations/annotations.json'
    with open(annotations_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    annotations = data['annotations']

    source = 'C:/lcy/2023_final/vehicle_detection/bonus_data/train_data/train/Navtech_Cartesian'
    image_folder = './bonus_data/images/train'
    label_folder = './bonus_data/labels/train'

    # Process radar dictionaries
    get_bonus_radar_dicts(images, annotations, source, image_folder, label_folder)

    # Split data into train and validation sets
    label_folder = os.path.join(save_dir, 'labels', 'train')
    image_folder = os.path.join(save_dir, 'images', 'train')

    # Move random subset to validation set
    for file in np.random.choice(os.listdir(label_folder), 40, replace=False):
        file = file[:-4]
        shutil.move(os.path.join(label_folder, file + '.txt'), 
                   os.path.join(save_dir, 'labels', 'val', file + '.txt'))
        shutil.move(os.path.join(image_folder, file + '.png'), 
                   os.path.join(save_dir, 'images', 'val', file + '.png'))

if __name__ == "__main__":
    main()
