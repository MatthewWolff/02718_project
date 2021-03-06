import os
import sys
import requests, zipfile, io

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.utils import shuffle

def main():
    if not os.path.exists('./data/CellData/chest_xray'):
        os.makedirs('./data', exist_ok=True)
        
        response = requests.get('https://data.mendeley.com/datasets/rscbjbr9sj/3/files/4356bbc1-92da-4738-8e27-b0ba604f07f4/ZhangLabData.zip?dl=1')
        z = zipfile.ZipFile(io.BytesIO(response.content))
        print('Data downloaded successfully')
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
    # shuffle data
    x, y = shuffle(x, y, random_state=0) # TODO: shuffle in a consistent manner for reproducibility
    return x, y

NUM_CLASSES = 2

# This function takes image paths as arguments and reads corresponding images
def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASSES)
    # read the img from file and decode it using tf
    img_file = tf.io.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3, name="decoded_images")
    return img_decoded, one_hot

# This function takes image and resizes it to smaller format (150x150)
def image_resize(images, labels):
    # Be very careful with resizing images like this and make sure to read the doc!
    # Otherwise, bad things can happen - https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    resized_image = tf.image.resize(images, (150, 150))
    resized_image_asint = tf.cast(resized_image, tf.int32)
    return resized_image_asint, labels

# Execution plan is defined here.
# Since it uses lazy evaluation, the images will not be read after calling build_pipeline_plan()
# We need to use iterator defined here in tf context
def build_pipeline_plan(img_paths, labels, batch_size):
    # We build a tensor of image paths and labels
    tr_data = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    # First step of input pipeline - read images in paths as jpegs
    tr_data_imgs = tr_data.map(input_parser)
    # Apply resize to each image in the pipeline
    tr_data_imgs = tr_data_imgs.map(image_resize)
    # Gives us opportuinty to batch images into small groups
    tr_dataset = tr_data_imgs.batch(batch_size)
    # create TensorFlow Iterator object directly from input pipeline
    iterator = tf.compat.v1.data.make_one_shot_iterator(tr_dataset)
    next_element = iterator.get_next()
    return next_element

# Function to execute defined pipeline in Tensorflow session
def process_pipeline(img_paths, labels, batch_size):
    '''
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
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
    '''
    # We build a tensor of image paths and labels
    tr_data = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    # First step of input pipeline - read images in paths as jpegs
    tr_data_imgs = tr_data.map(input_parser)
    # Apply resize to each image in the pipeline
    tr_data_imgs = tr_data_imgs.map(image_resize)
    # Gives us opportuinty to batch images into small groups
    tr_dataset_batched = tr_data_imgs.batch(batch_size)
    return tr_dataset_batched

#@tf.function
def load_dataset(path, batch_size):
    #tf.compat.v1.reset_default_graph()
    files, labels = get_labeled_files(path)
    p = tf.constant(files, name="train_imgs")
    l = tf.constant(labels, name="train_labels")
    
    images = np.asarray([], dtype=np.int32)
    labels_hot = np.asarray([], dtype=np.float32)

    batched_dataset = process_pipeline(p, l, batch_size=batch_size)
    for batch in batched_dataset: # only 1 batch due to large batch_size 
        images = batch[0]
        labels_hot = batch[1]

    return images.numpy(), labels_hot.numpy()

if __name__ == '__main__':
    main()