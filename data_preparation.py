import os
import sys
import glob
import urllib.request
import tarfile
import requests, zipfile, io

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import Dataset, Iterator

def main():
    if not os.path.exists('./data/CellData/chest_xray'):
        os.makedirs('./data', exist_ok=True)
        #urllib.request.urlretrieve("https://data.mendeley.com/datasets/rscbjbr9sj/3/files/4356bbc1-92da-4738-8e27-b0ba604f07f4/ZhangLabData.zip?dl=1", "data/chest_xray.zip.gz")
        #tar = tarfile.open("data/chest_xray.tar.gz")
        #tar.extractall(path='./data/')
        #os.remove('data/chest_xray.tar.gz')

        response = requests.get('https://data.mendeley.com/datasets/rscbjbr9sj/3/files/4356bbc1-92da-4738-8e27-b0ba604f07f4/ZhangLabData.zip?dl=1')
        z = zipfile.ZipFile(io.BytesIO(response.content))
        if response:
            z.extractall('./data/')
        else:
            print('An error ocurred while downloading the data:', response.text)
            sys.exit()
        
    img_normal = plt.imread('./data/CellData/chest_xray/train/NORMAL/NORMAL-28501-0001.jpeg')
    img_penumonia_bacteria = plt.imread('./data/CellData/chest_xray/train/PNEUMONIA/BACTERIA-213622-0001.jpeg')
    img_penumonia_virus = plt.imread('./data/CellData/chest_xray/train/PNEUMONIA/VIRUS-3637528-0002.jpeg')

    plt.figure(figsize=(12, 5))

    plt.subplot(1,3,1).set_title('NORMAL')
    plt.imshow(img_normal, cmap='gray')

    plt.subplot(1,3,2).set_title('PNEUMONIA/Bacteria')
    plt.imshow(img_penumonia_bacteria, cmap='gray')

    plt.subplot(1,3,3).set_title('PNEUMONIA/Virus')
    plt.imshow(img_penumonia_virus, cmap='gray')

    plt.tight_layout()
    plt.show()

def get_labeled_files(folder):
    x = []
    y = []
    
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
                continue # we do not investigate other dirs
            for image_filename in os.listdir(folder + folderName):
                img_path = folder + folderName + '/' + image_filename
                if img_path is not None and str.endswith(img_path, 'jpeg'):
                    x.append(img_path)
                    y.append(label)
    
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

NUM_CLASSES = 2

# This function takes image paths as arguments and reads corresponding images
def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASSES)
    # read the img from file and decode it using tf
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3, name="decoded_images")
    return img_decoded, one_hot

# This function takes image and resizes it to smaller format (150x150)
def image_resize(images, labels):
    # Be very careful with resizing images like this and make sure to read the doc!
    # Otherwise, bad things can happen - https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    resized_image = tf.image.resize_images(images, (150, 150), align_corners=True)
    resized_image_asint = tf.cast(resized_image, tf.int32)
    return resized_image_asint, labels

# Execution plan is defined here.
# Since it uses lazy evaluation, the images will not be read after calling build_pipeline_plan()
# We need to use iterator defined here in tf context
def build_pipeline_plan(img_paths, labels, batch_size):
    # We build a tensor of image paths and labels
    tr_data = Dataset.from_tensor_slices((img_paths, labels))
    # First step of input pipeline - read images in paths as jpegs
    tr_data_imgs = tr_data.map(input_parser)
    # Apply resize to each image in the pipeline
    tr_data_imgs = tr_data_imgs.map(image_resize)
    # Gives us opportuinty to batch images into small groups
    tr_dataset = tr_data_imgs.batch(batch_size)
    # create TensorFlow Iterator object directly from input pipeline
    iterator = tr_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

# Function to execute defined pipeline in Tensorflow session
def process_pipeline(next_element):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # get each element of the training dataset until the end is reached
        # in our case only one iteration since we read everything as 1 batch
        # can be multiple iterations if we decrease BATCH_SIZE to eg. 10
        images = []
        labels_hot = []
        while True:
            try:
                elem = sess.run(next_element)
                images = elem[0]
                labels_hot = elem[1]
            except tf.errors.OutOfRangeError:
                print("Finished reading the dataset")
                return images, labels_hot

def load_dataset(path, batch_size):
    tf.reset_default_graph()
    files, labels = get_labeled_files(path)
    p = tf.constant(files, name="train_imgs")
    l = tf.constant(labels, name="train_labels")
    
    next_element = build_pipeline_plan(p, l, batch_size=batch_size)
    imgs, labels = process_pipeline(next_element)
    return imgs, labels

if __name__ == '__main__':
    main()