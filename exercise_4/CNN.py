"""
ELEC-E5500 Speech Processing -- Autumn 2021 Python Exercise 4
Speaker Recognition.
"""

from random import seed
from numpy import *
import argparse
import os.path as path
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from keras import optimizers
from keras.losses import categorical_crossentropy

import functions

np.set_printoptions(threshold=sys.maxsize)


def main(opt):

    if opt.train:
        print(
            "|             Training a CNN based Speaker Verification System                           |"
        )
        print(
            " ******************************************************************************************\n"
        )

        training_filename = "training.lst"
        training_list = open(training_filename, "r")
        show_names, show_labels = functions.read_file(training_list)

        # To encode target labels with value between 0 and n_classes-1
        label_encoder = LabelEncoder()
        data_labels = label_encoder.fit_transform(show_labels)
        opt.n_classes = len(np.unique(data_labels))
        print("Number of classes", len(np.unique(data_labels)))

        # Binarize labels in a one-vs-all fashion
        binarize = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        data_labels = binarize.fit_transform(data_labels)

        training_data = functions.load_data(opt, show_names)
        training_data = np.array(training_data)

        # Partition the dataset into training and evaluation. Allocate 90% of the data for training and 10% of it for evaluation.
        # Use train_test_split which is a scikit-learn's function which helps us to split train and test audio data kept in the same folders.
        # You need to partiton both the train and development labels
        train, val, train_labels, val_labels = train_test_split(
            training_data,
            data_labels,
            train_size=0.9,
            test_size=0.1,
            random_state=opt.seed,
        )

        train = np.array(train)
        train_labels = np.array(train_labels)
        val = np.array(val)
        val_labels = np.array(val_labels)

        n_frames = opt.window_size
        n_features = 80
        n_channels = 1

        # Define the input shape. Use 350 by 80. We use 350 since we are taking only 3.5 seconds of the speech and 80 is the number of features. The channel is 1.
        # The input shape has the form of [window_size, number of features, number of channels]
        input_shape = [n_frames, n_features, n_channels]

        model = functions.cnn(opt, 2, n_filters=[16, 32], input_shape=input_shape)
        model.summary()

        optm = optimizers.Adam(
            lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
        )

        # Compile the model. Use categorical_crossentropy as a loss function.
        model.compile(optimizer=optm, loss=categorical_crossentropy)

        # Train the model for 50 epochs. Shuffle the data.
        model.fit(
            train,
            train_labels,
            epochs=opt.max_epochs,
            validation_data=(val, val_labels),
            shuffle=True,
        )

        # Save the trained model as “Speaker_Recognition_Exercise.hf5”
        model.save(filepath="Speaker_Recognition_Exercise.hf5")

    if opt.predict:
        print(" -------------------------------------------------")
        print(
            "|          Prediciting using trained CNN based Speaker Verification Model                            |"
        )
        print(
            "******************************************************************************************************\n"
        )

        validation_trials = "trials.lst"
        validation_list = open(validation_trials, "r")
        validation_names = functions.read_trials(validation_list)

        # Load the saved “Speaker_Recognition_Exercise.hf5” model.
        # Load model
        model_name = "Speaker_Recognition_Exercise.hf5"
        model = load_model(model_name)

        score_file = "scores_VoxCeleb-1"
        functions.predict_by_model(
            opt, model, validation_names, score_file, "Embedding"
        )
        print(".... Done prediction with model : %s" % model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A CNN based Speaker Verification System."
    )

    parser.add_argument(
        "--train", type=int, default=1, help="1 for trainning, 0 for predicting"
    )
    parser.add_argument(
        "--predict", type=int, default=0, help="0 for trainning, 1 for predicting"
    )

    # paths
    parser.add_argument(
        "--spec_path", type=str, default="./wav/", help="spectrograms path"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./", help="where model is saved"
    )

    # optmization:
    parser.add_argument(
        "--window_size", type=int, default=350, help="Number of frames in a sample"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="number of sequences to train on in parallel",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="number of full passes through the training data",
    )
    parser.add_argument(
        "--activation_function", type=str, default="relu", help="Activation function"
    )
    parser.add_argument("--n_classes", type=int, help="Number of classes")
    parser.add_argument(
        "--seed", type=int, default=3435, help="random number generator seed"
    )

    params = parser.parse_args()
    np.random.seed(params.seed)

    main(params)

