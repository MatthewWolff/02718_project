import os
from data_preparation_tf2 import load_dataset
from model_plots import plot_confusion_matrix, plot_learning_curves, plot_roc_curve

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

def main():
    MODEL_NAME = 'InceptionResNetV2'
    NUM_CLASSES = 2

    # load data
    X_train, y_train = load_dataset("./data/CellData/chest_xray/train/", 6000)
    X_test, y_test = load_dataset("./data/CellData/chest_xray/test/", 6000)

    print(X_train.shape)
    print(y_train.shape)

    print(y_train[:5,])
    print(y_train[-5:,])

    '''
    # explore data
    plt.subplot(1,3,1)
    sns.countplot(np.argmax(y_train, axis=1)).set_title('TRAIN DATASET')

    plt.subplot(1,3,2)
    sns.countplot(np.argmax(y_test, axis=1)).set_title('TEST DATASET')

    plt.tight_layout()
    plt.show()
    '''

    # convolutional base
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # classifier (fully connected layers on top)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # freeze the convolutional base
    for layer in base_model.layers: # check that conv_base in frozen!!!
        layer.trainable = False
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # This callback saves the weights of the model after each epoch
    checkpoint = ModelCheckpoint(
        'inception_resnet_v2/weights.epoch_{epoch:02d}.hdf5',
        monitor='val_loss', 
        save_best_only=False, 
        save_weights_only=False,
        mode='auto',
        verbose=1
    )

    # This callback writes logs for TensorBoard
    tensorboard = TensorBoard(
        log_dir='./Graph2', 
        histogram_freq=0,  
        write_graph=True
    )

    # directory to store the model weights
    os.makedirs('./inception_resnet_v2', exist_ok=True)

    # set class weights
    y_labels = np.argmax(y_train, axis=1)
    classweight = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
    print('class_weights:', classweight)

    history = model.fit(
        x=X_train, 
        y=y_train, 
        batch_size=64,
        epochs=20,
        verbose=1,
        callbacks=[checkpoint, tensorboard],
        validation_split=0.25,
        shuffle=True,
        class_weight=classweight)

    # pick the best model
    idx = np.argmin(history.history['val_loss']) 
    #model.load_weights("model/weights.epoch_{:02d}.hdf5".format(idx + 1))

    print("Loading the best model")
    print("epoch: {}, val_loss: {}, val_acc: {}".format(idx + 1, history.history['val_loss'][idx], history.history['val_acc'][idx]))

    # compute metrics on balanced test dataset
    rus = RandomUnderSampler(random_state=42)

    X_test_flat_shape = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
    X_test_flat = X_test.reshape(X_test.shape[0], X_test_flat_shape)

    y_test_flat = np.argmax(y_test, axis=1)

    X_res, y_res = rus.fit_resample(X_test_flat, y_test_flat)

    #print(X_res.shape)
    #print(y_res.shape)

    y_test_rus = to_categorical(y_res, num_classes = 2)

    for i in range(len(X_res)):
        height, width, channels = 150, 150, 3
        X_test_rus = X_res.reshape(len(X_res), height, width, channels)
        
    print(X_test_rus.shape)
    print(y_test_rus.shape)
    #sns.countplot(np.argmax(y_test_rus, axis=1)).set_title('TEST (undersampled)')

    test_loss, test_acc = model.evaluate(X_test_rus, y_test_rus, verbose=0)
    print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

    # plots
    plot_learning_curves(history, MODEL_NAME)
    y_pred = model.predict(X_test_rus)
    # to get the prediction, we pick the class with with the highest probability
    y_pred_classes = np.argmax(y_pred, axis = 1) 
    y_true = np.argmax(y_test_rus, axis = 1) 

    conf_mtx = confusion_matrix(y_true, y_pred_classes) 
    plot_confusion_matrix(conf_mtx, target_names = ['NORMAL', 'PNEUMONIA'], cmap='Greens', normalize=False, model_name=MODEL_NAME)
    plot_roc_curve(MODEL_NAME, NUM_CLASSES, y_test_rus, y_pred)

if __name__ == '__main__':
    main()