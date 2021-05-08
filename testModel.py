import os
import numpy as np
import numpy.random as random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from customDataset import CustomDataset
from resnetModel import ResNet
from computeStats import compute_detailed_stats, writeStatsToExcel

def testModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, SOCIAL_MEDIA_SEGMENTS, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L):

    TRAIN_CSV_PATH = PREPROCESSED_CSV_PATH + "train_dataset.csv"
    VALID_CSV_PATH = PREPROCESSED_CSV_PATH + "valid_dataset.csv"
    TEST_CSV_PATH = PREPROCESSED_CSV_PATH + "test_dataset.csv"

    ##########################
    # SETTINGS
    ##########################
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    # Make results reproducible
    torch.backends.cudnn.deterministic = True

    NUM_WORKERS = 8  # Number of processes in charge of preprocessing batches
    DATA_PARALLEL = True
    CUDA_DEVICE = 0
    if DATA_PARALLEL:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda:" + str(CUDA_DEVICE))

    # Hyperparameters
    BATCH_SIZE = 256

    ############################
    # GENERATE DUMMY BINS INNER EDGES
    ############################
    dummy_bins_inner_edges = torch.from_numpy(np.sort(random.randint(low=MIN_AGE+1, high=MAX_AGE+1, size=(M, L-1))))
    # This will be overwritten when when the model loads and we don't need it for the dataset anyway

    ###################
    # Dataset
    ###################
    custom_transform = transforms.Compose([transforms.Resize((240, 240)),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])

    train_dataset = CustomDataset(csv_path=TRAIN_CSV_PATH,
                                  img_dir=PREPROCESSED_IMAGES_PATH,
                                  bins_inner_edges=dummy_bins_inner_edges,
                                  transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.ToTensor()])

    test_dataset = CustomDataset(csv_path=TEST_CSV_PATH,
                                 img_dir=PREPROCESSED_IMAGES_PATH,
                                 bins_inner_edges=dummy_bins_inner_edges,
                                 transform=custom_transform2)

    valid_dataset = CustomDataset(csv_path=VALID_CSV_PATH,
                                  img_dir=PREPROCESSED_IMAGES_PATH,
                                  bins_inner_edges=dummy_bins_inner_edges,
                                  transform=custom_transform2)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

    ##########################
    # MODEL
    ##########################
    model = ResNet(RESNET_SIZE, MIN_AGE, MAX_AGE, dummy_bins_inner_edges)
    model.load_state_dict(torch.load(os.path.join(OUT_PATH, 'best_model.pt')))
    if DATA_PARALLEL:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    ##########################
    # GET DETAILED MODEL STATS
    ##########################
    model.eval()
    with torch.set_grad_enabled(False):
        print("Training Dataset...")
        writeStatsToExcel(SOCIAL_MEDIA_SEGMENTS, OUT_PATH + 'TrainRes.xlsx', *compute_detailed_stats(model, train_loader, DEVICE, SOCIAL_MEDIA_SEGMENTS))
        print("Validation Dataset...")
        writeStatsToExcel(SOCIAL_MEDIA_SEGMENTS, OUT_PATH + 'ValidRes.xlsx', *compute_detailed_stats(model, valid_loader, DEVICE, SOCIAL_MEDIA_SEGMENTS))
        print("Testing Dataset...")
        age_mae, age_mse, age_seg_acc, age_stats = compute_detailed_stats(model, test_loader, DEVICE, SOCIAL_MEDIA_SEGMENTS)
        writeStatsToExcel(SOCIAL_MEDIA_SEGMENTS, OUT_PATH + 'TestRes.xlsx', age_mae, age_mse, age_seg_acc, age_stats)
        print("Finished!")

    return age_mae