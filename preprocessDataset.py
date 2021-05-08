import cv2
import face_recognition
import pandas as pd
import numpy as np
from PIL import Image
import os


def preprocessImage(img, nTimesToUpsample, margin):
    detectedFaces = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), number_of_times_to_upsample=nTimesToUpsample, model="cnn")
    # Filter images with multiple/no detected faces
    if len(detectedFaces) != 1:
        return detectedFaces, None
    # Modify face bounding rectangle to become bounding square if possible
    img_h, img_w, _ = img.shape
    y1, x2, y2, x1 = detectedFaces[0]
    width = x2 - x1 - 1
    height = y2 - y1 - 1
    diff = height - width
    shift = int(margin * max(width, height))
    if (diff > 0):
        if diff % 2 == 0:  # symmetric
            top = max(y1 - shift, 0)
            bottom = min(y2 + shift, img_h - 1)
            left = max(x1 - shift - int(diff / 2), 0)
            right = min(x2 + shift + int(diff / 2), img_w - 1)
        else:
            top = max(y1 - shift, 0)
            bottom = min(y2 + shift, img_h - 1)
            left = max(x1 - shift - int((diff - 1) / 2), 0)
            right = min(x2 + shift + int((diff + 1) / 2), img_w - 1)
    elif (diff <= 0):
        if diff % 2 == 0:  # symmetric
            top = max(y1 - shift + int(diff / 2), 0)
            bottom = min(y2 + shift - int(diff / 2), img_h - 1)
            left = max(x1 - shift, 0)
            right = min(x2 + shift, img_w - 1)
        else:
            top = max(y1 - shift + int((diff - 1) / 2), 0)
            bottom = min(y2 + shift - int((diff + 1) / 2), img_h - 1)
            left = max(x1 - shift, 0)
            right = min(x2 + shift, img_w - 1)

    face = img[top:bottom, left:right, :]
    face = np.array(Image.fromarray(np.uint8(face)).resize((224, 224), Image.ANTIALIAS))

    return detectedFaces, face


def preprocessDataset(MARGIN, DATASETS_PATH, PREPROCESSED_FOLDER_PATH, PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH):

    # PATHS
    UTKFACE_IMAGES_PATH = DATASETS_PATH + "Original/UTKFace/"
    if not os.path.exists(PREPROCESSED_FOLDER_PATH):
        os.mkdir(PREPROCESSED_FOLDER_PATH)
    if not os.path.exists(PREPROCESSED_IMAGES_PATH):
        os.mkdir(PREPROCESSED_IMAGES_PATH)
    if not os.path.exists(PREPROCESSED_CSV_PATH):
        os.mkdir(PREPROCESSED_CSV_PATH)


    # PREPROCESSING VARIABLES
    nTimesToUpsample = 1 # for low quality images


    # Read the filtered imdb and wiki CSVs and create final dataframe that will contain all preprocessed images
    preprocessed_df = pd.DataFrame(columns=["genders", "ages", "img_paths"])


    # Preprocess UTKFace
    count = 0
    for filename in os.listdir(UTKFACE_IMAGES_PATH):
        # Read image and detect faces
        img = cv2.imread(UTKFACE_IMAGES_PATH + filename)
        # Preprocess image
        detectedFaces, face = preprocessImage(img, nTimesToUpsample, MARGIN)
        # Filter images with multiple/no detected faces
        if len(detectedFaces) != 1:
            continue
        # Append row to preprocessed_df
        age = filename.split("_")[0]
        gender = str(1 - int(filename.split("_")[1]))
        preprocessed_df = preprocessed_df.append({"genders": gender, "ages": age, "img_paths": filename}, ignore_index=True)
        # Write preprocessed image to Preprocessed Dataset
        cv2.imwrite(PREPROCESSED_IMAGES_PATH + filename, face)

        count += 1

        if count % 100 == 0:
            print("UTKFace: ", count)

    preprocessed_df.to_csv(PREPROCESSED_CSV_PATH + "preprocessedDataset.csv", index=False)
