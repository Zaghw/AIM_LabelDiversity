import os
import time
import torch
import numpy as np
import numpy.random as random
import torch.nn as nn
import sys
import math
from torch.utils.data import DataLoader
from torchvision import transforms


from customDataset import CustomDataset
from resnetModel import ResNet
from computeStats import compute_summarized_stats

def trainModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L):

    TRAIN_CSV_PATH = PREPROCESSED_CSV_PATH + "train_dataset.csv"
    VALID_CSV_PATH = PREPROCESSED_CSV_PATH + "valid_dataset.csv"

    ##########################
    # SETTINGS
    ##########################
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    # Make results reproducible
    torch.backends.cudnn.deterministic = True
    RANDOM_SEED = 1

    # GPU settings
    NUM_WORKERS = 8  # Number of processes in charge of preprocessing batches
    DATA_PARALLEL = True
    CUDA_DEVICE = 0
    if torch.cuda.is_available():
        if DATA_PARALLEL:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cuda:" + str(CUDA_DEVICE))
    else:
        DEVICE=torch.device("cpu")




    # Hyperparameters
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 10000
    BATCH_SIZE = 256


    ##########################
    # LOGGING
    ##########################
    LOGFILE = os.path.join(OUT_PATH, 'training.log')

    header = []
    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Output Path: %s' % OUT_PATH)
    header.append('Script: %s' % sys.argv[0])
    header.append('ResNetSize: %s' % RESNET_SIZE)
    header.append('Batch Size: %s' % BATCH_SIZE)

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()

    ############################
    # GENERATE BINS INNER EDGES
    ############################
    bins_inner_edges = torch.from_numpy(np.sort(random.randint(low=MIN_AGE+1, high=MAX_AGE+1, size=(M, L-1))))
    # Lowest edge will be MIN_AGE+1 and highest edge will be MAX_AGE

    ###################
    # Dataset
    ###################
    custom_transform = transforms.Compose([transforms.Resize((240, 240)),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor()])

    train_dataset = CustomDataset(csv_path=TRAIN_CSV_PATH,
                                  img_dir=PREPROCESSED_IMAGES_PATH,
                                  bins_inner_edges=bins_inner_edges,
                                  transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.ToTensor()])

    valid_dataset = CustomDataset(csv_path=VALID_CSV_PATH,
                                    img_dir=PREPROCESSED_IMAGES_PATH,
                                    bins_inner_edges=bins_inner_edges,
                                    transform=custom_transform2)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    ##########################
    # MODEL
    ##########################
    model = ResNet(RESNET_SIZE, MIN_AGE, MAX_AGE, bins_inner_edges)
    if DATA_PARALLEL:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    ###########################################
    # COST FUNCTIONS
    ###########################################
    def age_loss_fn(age_label_matrices, age_pred_matrices):
        return -torch.sum(age_label_matrices * torch.log2(age_pred_matrices))

    ###########################################
    # OPTIMIZER
    ###########################################
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ###########################################
    # TRAINING & VALIDATION
    ###########################################
    start_time = time.time()
    # Initialize validation stats
    best_mae, best_rmse, best_overall_acc, best_epoch, best_valid_loss = 0, 0, 0, 0, -1
    # Initialize counter since last improvement on validation
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):

        # Training
        model.train()
        for batch_idx, (images, age_labels, age_labels_matrices) in enumerate(train_loader):

            images = images.to(DEVICE)
            age_labels_matrices = age_labels_matrices.to(DEVICE)

            # FORWARD AND BACK PROP
            age_preds, age_preds_matrices = model(images)
            loss = age_loss_fn(age_labels_matrices, age_preds_matrices)
            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Training Loss: %.4f' % (epoch + 1, NUM_EPOCHS, batch_idx, len(train_dataset) // BATCH_SIZE, loss.item()))
            if not batch_idx % 1:
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)

        # Validation
        model.eval()
        with torch.set_grad_enabled(False):
            valid_mae, valid_mse, valid_loss = compute_summarized_stats(model, age_loss_fn, valid_loader, DEVICE)

        if valid_loss < best_valid_loss or best_valid_loss == -1:
            best_mae, best_rmse, best_epoch, best_valid_loss = valid_mae, math.sqrt(valid_mse), epoch + 1, valid_loss
            early_stop_counter = 0

            ########## SAVE MODEL #############
            if DATA_PARALLEL:
                torch.save(model.module.state_dict(), os.path.join(OUT_PATH, 'best_model.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(OUT_PATH, 'best_model.pt'))
        else:
            early_stop_counter += 1
            if early_stop_counter > EARLY_STOPPING_PATIENCE:
                with open(LOGFILE, 'a') as f:
                    s = "IMPROVEMENT ON VALIDATION SET HAS STALLED...INITIATING EARLY-STOPPING"
                    print(s)
                    f.write('%s\n' % s)
                    return best_valid_loss


        s = 'STATS: | Current Valid: MAE=%.4f,MSE=%.4f, VALID_LOSS=%.4f, EPOCH=%d | ' \
            'Best Valid :MAE=%.4f,MSE=%.4f, VALID_LOSS=%.4f, EPOCH=%d | LAST IMPROVEMENT=%d EPOCHS' % (
            valid_mae, math.sqrt(valid_mse), valid_loss, epoch + 1, best_mae,
            best_rmse, best_valid_loss, best_epoch + 1, early_stop_counter)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.4f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    return best_valid_loss

# if __name__ == "__main__":
#
#     # Training Variables
#     MARGIN = 25
#     RESNET_SIZE = "ResNet34"  # "ResNet34" or "ResNet50" or "ResNet101" or "ResNet152"
#     MIN_AGE = 13  # Inclusive
#     MAX_AGE = 116  # Inclusive
#     M = 5   # Number of different bin configurations
#     L = 10  # Number of bins in each configuration
#     SOCIAL_MEDIA_SEGMENTS = np.array([25, 35, 50])  # (MIN_AGE,24), (25,34), (35,49), (50,MAX_AGE)
#
#     # Raw Images Paths
#     DATASETS_PATH = "../Datasets/"
#
#
#     # Preprocessed Images Paths
#     PREPROCESSED_FOLDER_PATH = DATASETS_PATH + "Preprocessed-" + str(MARGIN) + "/"
#     PREPROCESSED_IMAGES_PATH = PREPROCESSED_FOLDER_PATH + "Images/"
#     PREPROCESSED_CSV_PATH = PREPROCESSED_FOLDER_PATH + "CSVs/"
#
#
#     # Output Path
#     OUTPUT_FOLDER_NAME = "LabelDiversity-" + RESNET_SIZE + "-" + str(M) + "X" + str(L)
#     OUT_PATH = "../TrainedModels/" + OUTPUT_FOLDER_NAME + "/"
#
#
#     # preprocessDataset(MARGIN, DATASETS_PATH, PREPROCESSED_FOLDER_PATH, PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH)
#     # distributeDatset(PREPROCESSED_CSV_PATH, MIN_AGE, MAX_AGE, SOCIAL_MEDIA_SEGMENTS)
#     validLoss = trainModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L)
#     # print("Margin: ", Margin, " returned Validation Cost: ", validCost)
#     # testModel(PREPROCESSED_IMAGES_PATH, PREPROCESSED_CSV_PATH, OUT_PATH, SOCIAL_MEDIA_SEGMENTS, RESNET_SIZE, MIN_AGE, MAX_AGE, M, L)
