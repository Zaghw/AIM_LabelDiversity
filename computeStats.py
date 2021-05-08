import torch
import numpy as np
import xlsxwriter
import math

def compute_summarized_stats(model, age_loss_fn, data_loader, device):

    cumul_abs_error, cumul_sq_error, num_images, loss = 0, 0, 0, 0
    for batch_idx, (images, age_labels, age_labels_matrices) in enumerate(data_loader):

        images = images.to(device)
        age_labels = age_labels.to(device)
        age_labels_matrices = age_labels_matrices.to(device)

        age_preds, age_preds_matrices = model(images)
        age_loss = age_loss_fn(age_labels_matrices, age_preds_matrices)
        loss += age_loss

        num_images += age_labels.size(0)
        cumul_abs_error += torch.sum(torch.abs(age_preds - age_labels))
        cumul_sq_error += torch.sum((age_preds - age_labels) ** 2)

    age_mae = cumul_abs_error.item() / num_images
    age_mse = cumul_sq_error.item() / num_images

    return age_mae, age_mse, loss



def compute_detailed_stats(model, data_loader, device, SOCIAL_MEDIA_SEGMENTS):
    cumul_abs_error, cumul_sq_error, cumul_correct_count, num_examples = 0, 0, 0, 0
    age_stats = []

    for i in range(len(SOCIAL_MEDIA_SEGMENTS) + 1):
        age_stats.append({"total_count": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0})

    for batch_idx, (images, age_labels, age_labels_matrices) in enumerate(data_loader):

        # Move batch to device
        images = images.to(device)
        age_labels = age_labels.to(device)

        # Get model predictions
        age_preds, age_preds_matrices = model(images)

        # Get Age Segments from Age Labels and Predictions
        age_labels_segments = torch.from_numpy(np.digitize(age_labels, SOCIAL_MEDIA_SEGMENTS))
        age_preds_segments = torch.from_numpy(np.digitize(age_preds, SOCIAL_MEDIA_SEGMENTS))

        # Get True Positives, False Positives, and False Negatives
        for j in range(len(SOCIAL_MEDIA_SEGMENTS) + 1):
            age_stats[j]["total_count"] += torch.sum(age_labels_segments == j)
            age_stats[j]["true_positives"] += torch.sum(torch.logical_and(age_labels_segments == j, age_preds_segments == j))
            age_stats[j]["false_positives"] += torch.sum(torch.logical_and(age_labels_segments != j, age_preds_segments == j))
            age_stats[j]["false_negatives"] += torch.sum(torch.logical_and(age_labels_segments == j, age_preds_segments != j))
        correct_age_preds = age_labels_segments == age_preds_segments

        num_examples += age_labels.size(0)
        cumul_abs_error += torch.sum(torch.abs(age_preds - age_labels))
        cumul_sq_error += torch.sum((age_preds - age_labels)**2)
        cumul_correct_count += torch.sum(correct_age_preds)

    age_mae = cumul_abs_error.item() / num_examples
    age_mse = cumul_sq_error.item() / num_examples
    age_seg_acc = cumul_correct_count.item() / num_examples

    for i in range(len(SOCIAL_MEDIA_SEGMENTS) + 1):
        age_stats[i]["precision"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_positives"])
        age_stats[i]["recall"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_negatives"])
        age_stats[i]["f1_score"] = 2 * (age_stats[i]["precision"] * age_stats[i]["recall"]) / (age_stats[i]["precision"] + age_stats[i]["recall"])

    return age_mae, age_mse, age_seg_acc, age_stats



def writeStatsToExcel(SOCIAL_MEDIA_SEGMENTS, OUTPUT_FILENAME ,age_mae, age_mse, age_seg_acc, age_stats):

    # WRITE RESULTS TO EXCEL
    workbook = xlsxwriter.Workbook(OUTPUT_FILENAME)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    # Prepare column headings
    worksheet.write(row, col + 1, "Precision")
    worksheet.write(row, col + 2, "Recall")
    worksheet.write(row, col + 3, "F1-Score")
    worksheet.write(row, col + 4, "Support")
    row += 1

    # Write Age Results
    for i in range(len(SOCIAL_MEDIA_SEGMENTS) + 1):
        worksheet.write(row, col, "CLASS" + str(i) + ":")
        worksheet.write(row, col + 1, age_stats[i]["precision"].__float__())
        worksheet.write(row, col + 2, age_stats[i]["recall"].__float__())
        worksheet.write(row, col + 3, age_stats[i]["f1_score"].__float__())
        worksheet.write(row, col + 4, age_stats[i]["total_count"].__float__())
        row += 1
    worksheet.write(row, col, "Age Acc:")
    worksheet.write(row, col + 1, age_seg_acc)
    worksheet.write(row, col + 2, "Age MAE:")
    worksheet.write(row, col + 3, age_mae)
    worksheet.write(row, col + 4, "Age RMSE:")
    worksheet.write(row, col + 5, math.sqrt(age_mse))
    row += 2

    # Close xlsx
    workbook.close()