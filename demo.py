"""
Given a video path and a saved model (checkpoint), produce classification
predictions.
Note that if using a model that requires features to be extracted, those
features must be extracted first.
"""
from keras.models import load_model
from data import DataSet
import numpy as np


def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape, class_limit=class_limit)
    
    # Extract the sample from the data.
    sample = data.get_frames_by_filename(video_name, data_type)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    saved_model = '/data/d14122793/ucf101_full/checkpoints/******.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 101

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    # video_name = 'v_Archery_g04_c02'
    video_name = 'v_Basketball_g07_c04'
    #video_name = 'v_test_g01_c01'
    print ("Current video name:", video_name)
    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See data.py for options.")

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)

if __name__ == '__main__':
    main()