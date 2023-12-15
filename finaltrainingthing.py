import torch
from IPython.display import Image  # for displaying images
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt


# Function to get the data from XML Annotation
def extract_xml_file(xml_file):
    xml_root = ET.parse(xml_file).getroot()

    # Initialise the info dict
    img_info_dict = {}
    img_info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in xml_root:
        # Get the file name
        if elem.tag == "filename":
            img_info_dict["filename"] = elem.text

        # Get size of the image
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            img_info_dict["image_size"] = tuple(image_size)

        # Get bounding box of the image
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            img_info_dict["bboxes"].append(bbox)

    return img_info_dict


print(
    extract_xml_file(
        os.path.join(
            "datasets",
            "cub_200_2011_xml",
            "train_labels",
            "Acadian_Flycatcher_0003_29094.xml",
        )
    )
)

class_names = []
class_name_to_id_mapping = {}


# get class names
def get_class_names(info_dict):
    for b in info_dict["bboxes"]:
        class_names.append(b["class"])


def mapping_to_class_name_to_id(class_names):
    unique_class_names = np.unique(class_names)
    for i, unique_label in enumerate(unique_class_names):
        class_name_to_id_mapping[unique_label] = i


# Get the all train and validation xml annotations file path
train_annotations_labels = [
    os.path.join("./datasets/cub_200_2011_xml/train_labels", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/train_labels")
    if x[-3:] == "xml"
]
train_annotations_labels.sort()
# # test
test_annotations_labels = [
    os.path.join("./datasets/cub_200_2011_xml/valid_labels", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/valid_labels")
    if x[-3:] == "xml"
]
test_annotations_labels.sort()


# extract xml file and append label into class_names list container
for i, ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    get_class_names(info_dict)

# If all label store on list container than mapping them unique number
mapping_to_class_name_to_id(class_names)

len(train_annotations_labels), len(class_name_to_id_mapping), len(
    test_annotations_labels
)

print(class_name_to_id_mapping)


# Convert the info dict to the required yolo txl file format and write it to disk
def convert_to_yolov8(info_dict, path):
    print_buffer = []

    # For each bounding box
    for bbox in info_dict["bboxes"]:
        try:
            # get class id for each label
            class_id = class_name_to_id_mapping[bbox["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v8
        b_center_x = (bbox["xmin"] + bbox["xmax"]) / 2
        b_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
        b_width = bbox["xmax"] - bbox["xmin"]
        b_height = bbox["ymax"] - bbox["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bounding box details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    # Name of the file which we have to save
    save_file_name = os.path.join(path, info_dict["filename"].replace("jpg", ""))
    save_file_name += ".txt"
    print(save_file_name)
    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


# Convert the info dict to the required yolo txl file format and write it to disk
def convert_to_yolov8(info_dict, path):
    print_buffer = []

    # For each bounding box
    for bbox in info_dict["bboxes"]:
        try:
            # get class id for each label
            class_id = class_name_to_id_mapping[bbox["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v8
        b_center_x = (bbox["xmin"] + bbox["xmax"]) / 2
        b_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
        b_width = bbox["xmax"] - bbox["xmin"]
        b_height = bbox["ymax"] - bbox["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bounding box details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    # Name of the file which we have to save
    save_file_name = os.path.join(path, info_dict["filename"].replace("jpg", ""))
    save_file_name += ".txt"
    print(save_file_name)
    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


# Convert and save the train annotations
for i, ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict, "./datasets/cub_200_2011_xml/train_images/")

train_annotations_labels = [
    os.path.join("./datasets/cub_200_2011_xml/train_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/train_images/")
    if x[-3:] == "txt"
]

# Convert and save the test annotations
for i, ann in enumerate(tqdm(test_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict, "./datasets/cub_200_2011_xml/valid_images/")

test_annotations_labels = [
    os.path.join("./datasets/cub_200_2011_xml/valid_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/valid_images/")
    if x[-3:] == "txt"
]

len(train_annotations_labels), len(test_annotations_labels)


random.seed(0)

class_id_to_name_mapping = dict(
    zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys())
)


def plot_image_with_bounding_box(image, annotation_list):
    """
    image : It's actual numpy formatted image you input.
    annotation_list : It's give as label with bounding box.

    """

    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    t_annotations = np.copy(annotations)
    t_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    t_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    t_annotations[:, 1] = t_annotations[:, 1] - (t_annotations[:, 3] / 2)
    t_annotations[:, 2] = t_annotations[:, 2] - (t_annotations[:, 4] / 2)
    t_annotations[:, 3] = t_annotations[:, 1] + t_annotations[:, 3]
    t_annotations[:, 4] = t_annotations[:, 2] + t_annotations[:, 4]

    for ann in t_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()


# Get any random label file
label_file = random.choice(train_annotations_labels)
with open(label_file, "r") as file:
    label_with_bounding_box = file.read().split("\n")[:-1]
    label_with_bounding_box = [x.split(" ") for x in label_with_bounding_box]
    label_with_bounding_box = [[float(y) for y in x] for x in label_with_bounding_box]

# Get the equal image file
image_file = label_file.replace("annotations", "images").replace("txt", "jpg")

assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)


# Plot the Bounding Box
plot_image_with_bounding_box(image, label_with_bounding_box)

# Read images and labels
train_images = [
    os.path.join("./datasets/cub_200_2011_xml/train_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/train_images/")
    if x[-3:] == "jpg"
]
train_labels = [
    os.path.join("./datasets/cub_200_2011_xml/train_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/train_images/")
    if x[-3:] == "txt"
]

test_images = [
    os.path.join("./datasets/cub_200_2011_xml/valid_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/valid_images/")
    if x[-3:] == "jpg"
]
test_labels = [
    os.path.join("./datasets/cub_200_2011_xml/valid_images/", x)
    for x in os.listdir("./datasets/cub_200_2011_xml/valid_images/")
    if x[-3:] == "txt"
]

train_images.sort()
train_labels.sort()

test_images.sort()
test_labels.sort()

# # Split the dataset into valid-test splits
val_images, test_images, val_label, test_label = train_test_split(
    test_images, test_labels, test_size=0.5, random_state=1
)

# check how many image have each categories
len(train_images), len(train_labels), len(val_images), len(val_label), len(
    test_images
), len(test_label)

# copilot
os.makedirs("./bird_species/train/images", exist_ok=True)
os.makedirs("./bird_species/train/labels", exist_ok=True)
os.makedirs("./bird_species/val/images", exist_ok=True)
os.makedirs("./bird_species/val/labels", exist_ok=True)
os.makedirs("./bird_species/test/images", exist_ok=True)
os.makedirs("./bird_species/test/labels", exist_ok=True)


# Utility function to move images
def move_files(list_of_files, dst_folder):
    for f in list_of_files:
        try:
            shutil.move(f, dst_folder)
        except:
            print(f)
            assert False


# Move the splits into their folders
move_files(train_images, "./bird_species/train/images/")
print("train")
move_files(val_images, "./bird_species/val/images/")
print("val")
move_files(test_images, "./bird_species/test/images/")
print("test")
move_files(train_labels, "./bird_species/train/labels/")
print("train label")
move_files(val_label, "./bird_species/val/labels/")
print("val_label")
move_files(test_label, "./bird_species/test/labels/")
print("test_label")
