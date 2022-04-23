"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from keras.models import load_model
from data import DataSet
import argparse


def validate(data_type, model, seq_length=40, saved_model=None,
             class_limit=None, image_shape=None):
    # Creating train generator with 8596 samples.
    # Creating test generator with 3418 samples.
    # Total 12041 samples
    test_data_num = 3418
    batch_size = 32

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    test_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    #model = load_model(saved_model)

    # Evaluate!
    #results = rm.model.evaluate_generator(
     #   generator=val_generator,
      #  val_samples=3200)
    results = rm.model.evaluate_generator(generator=test_generator, steps=test_data_num // batch_size)
    print(results)
    print(rm.model.metrics_names)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="Select a model to train (conv_3d, c3d, lrcn, lstm, mlp)")
    args = parser.parse_args()

    # Fetch model selection
    model = args.model

    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (299, 299, 3)
        if model == 'conv_3d':
            saved_model = '/data/d14122793/ucf101_full/checkpoints/conv_3d-images.****.hdf5'
            validate(data_type, model, saved_model=saved_model,
                     image_shape=image_shape, class_limit=101)

        elif model == 'c3d':
            saved_model = '/data/d14122793/ucf101_full/checkpoints/c3d-images.****.hdf5'
            validate(data_type, model, saved_model=saved_model,
                     image_shape=image_shape, class_limit=101)

        else:
            saved_model = '/data/d14122793/ucf101_full/checkpoints/lrcn-images.033-3.831.hdf5'
            validate(data_type, model, saved_model=saved_model,
                     image_shape=image_shape, class_limit=101)

    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None

        if model == 'lstm':
            saved_model = '/data/d14122793/ucf101_full/checkpoints/lstm-features.025-1.044.hdf5'
            validate(data_type, model, saved_model=saved_model,
                     image_shape=image_shape, class_limit=101)

        else:
            saved_model = '/data/d14122793/ucf101_full/checkpoints/mlp-features.006-1.030.hdf5'
            validate(data_type, model, saved_model=saved_model,
                     image_shape=image_shape, class_limit=101)
    else:
        raise ValueError("Invalid model. Please choose one of them: conv_3d, c3d, lrcn, lstm, mlp.")


if __name__ == '__main__':
    main()