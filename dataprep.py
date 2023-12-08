import os
import shutil

# Base directory where the NABirds dataset is located
base_dir = "datasets/nabirds"  # Replace with the actual path to your NABirds dataset
images_base_dir = os.path.join(base_dir, "images")

# Directory for train and test datasets
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# File paths
train_test_split_file = os.path.join(base_dir, "train_test_split.txt")
images_file = os.path.join(base_dir, "images.txt")

# Read images file to build a dictionary to map image IDs to file paths
image_paths = {}
with open(images_file, "r") as file:
    for line in file:
        image_id, image_rel_path = line.strip().split()
        image_paths[image_id] = image_rel_path


# Function to copy images to the appropriate dataset directory
def copy_image(image_id, is_train):
    source_image_path = os.path.join(images_base_dir, image_paths[image_id])
    species_dir = os.path.dirname(image_paths[image_id])

    if is_train:
        destination_dir = os.path.join(train_dir, species_dir)
    else:
        destination_dir = os.path.join(test_dir, species_dir)

    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy(source_image_path, destination_dir)


# Read the train/test split file and copy images to the corresponding directories
with open(train_test_split_file, "r") as file:
    for line in file:
        image_id, is_train = line.strip().split()
        copy_image(image_id, int(is_train) == 1)

print("Dataset preparation is complete.")
