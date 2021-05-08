import numpy as np
from preprocessDataset import preprocessDataset
from distributeDataset import distributeDatset
from trainModel import trainModel
from testModel import testModel

if __name__ == "__main__":

    # Training Variables
    MARGIN = 25
    RESNET_SIZE = "ResNet34"  # "ResNet34" or "ResNet50" or "ResNet101" or "ResNet152"
    MIN_AGE = 13  # Inclusive
    MAX_AGE = 116  # Inclusive
    M_array = [2, 4, 8, 16, 32, 64]   # Number of different bin configurations
    L_array = [8, 16, 32, 64]  # Number of bins in each configuration
    SOCIAL_MEDIA_SEGMENTS = np.array([25, 35, 50])  # (MIN_AGE,24), (25,34), (35,49), (50,MAX_AGE)

    # Raw Images Paths
    DATASETS_PATH = "../Datasets/"


    # Preprocessed Images Paths
    PREPROCESSED_FOLDER_PATH = DATASETS_PATH + "UTKFace-Preprocessed-" + str(MARGIN) + "/"
    PREPROCESSED_IMAGES_PATH = PREPROCESSED_FOLDER_PATH + "Images/"
    PREPROCESSED_CSV_PATH = PREPROCESSED_FOLDER_PATH + "CSVs/"

    # preprocessDataset(MARGIN, DATASETS_PATH, PREPROCESSED_FOLDER_PATH, PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH)
    # distributeDatset(PREPROCESSED_CSV_PATH, MIN_AGE, MAX_AGE, SOCIAL_MEDIA_SEGMENTS)

    best_age_mae, best_ML = -1, (0, 0)
    for M in M_array:
        for L in L_array:
            # Output Path
            OUTPUT_FOLDER_NAME = "LabelDiversity-" + RESNET_SIZE + "-" + str(M) + "X" + str(L)
            OUT_PATH = "../TrainedModels/" + OUTPUT_FOLDER_NAME + "/"

            validLoss = trainModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L)
            age_mae = testModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, SOCIAL_MEDIA_SEGMENTS, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L)

            if age_mae < best_age_mae or best_age_mae == -1:
                best_age_mae = validLoss
                best_ML = (M, L)
                print("Current ML:\t", (M, L), "\tCurrent MAE:\t", age_mae)
                print("Best ML:\t", best_ML, "\tBest MAE:\t", best_age_mae)
