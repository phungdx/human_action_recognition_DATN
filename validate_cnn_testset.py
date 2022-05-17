"""
Classify test images set through our CNN.
Use keras 2+ and tensorflow 1+
It takes a long time for hours.
"""
import numpy as np
import operator
import random
import glob
from data import DataSet
from processor import process_image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

data = DataSet()
def main():
    # CNN model evaluate

    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_num = 697865 #the number of test images
    batch_size = 32
    test_generator = test_data_gen.flow_from_directory('/data/d14122793/ucf101_full/test/', target_size=(299, 299),
                                                       batch_size=batch_size, classes=data.classes,
                                                       class_mode='categorical')
    # load the trained model that has been saved in CNN_train_UCF101.py, your model name maybe is not the same as follow
    model = load_model('/data/d14122793/ucf101_full/checkpoints/inception.036-1.60.hdf5')
    results = model.evaluate(test_generator, steps=test_data_num // batch_size)
    print(results)
    print(model.metrics)


if __name__ == '__main__':
    main()