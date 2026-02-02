import os
import numpy as np
from .align_eval.eval import main_eval_one_file, writeCsv

METRICS = ['Average absolute error', 'Median absolute error', 'Percentage of correct segments',
            'Percentage of correct onsets with tolerance']

def eval_submission(input_folder, output_folder, year, submission, dataset):
    means = []
    medians = []
    percentages_correct = []
    percentages_tolerance = []
    csv_results = [['Track'] + METRICS]
    output_path = os.path.join(output_folder, str(year), dataset, submission + '.csv')
    for file in os.listdir(os.path.join(input_folder, 'Ground-Truth', dataset)):
        gt_path = os.path.join(input_folder, 'Ground-Truth', dataset, file)
        pred_path = os.path.join(input_folder, str(year), submission, dataset, file)
        if not os.path.exists(pred_path):
            pred_path = os.path.join(input_folder, str(year), submission, dataset, file.replace('.lab', '.txt'))
        mean, median, percentage_correct, percentage_tolerance = main_eval_one_file(["dummy", gt_path, pred_path, 0.3])
        if percentage_correct:
            perc_correct = '{:.6f}'.format(percentage_correct)
        else:
            perc_correct = 'NaN'
        means.append(mean)
        medians.append(median)
        percentages_correct.append(0.0 if percentage_correct is None else percentage_correct)
        percentages_tolerance.append(percentage_tolerance)
        csv_results.append([file, '{:.6f}'.format(mean),
                            '{:.6f}'.format(median),
                            perc_correct,
                            '{:.6f}'.format(percentage_tolerance)])
    os.makedirs(os.path.basename(output_path), exist_ok=True)
    writeCsv(output_path, csv_results)
    avg_mean = sum(means) / len(means)
    avg_median = sum(medians) / len(medians)
    avg_percentage_correct = sum(percentages_correct) / len(percentages_correct)
    avg_percentage_tolerance = sum(percentages_tolerance) / len(percentages_tolerance)
    return avg_mean, avg_median, avg_percentage_correct, avg_percentage_tolerance

def main(input_folder, output_folder, year):
    all_results = {}
    for dataset in os.listdir(os.path.join(input_folder, 'Ground-Truth')):
        all_results[dataset] = {'scores': {}, 'metrics': METRICS, 'submissions': []}
        for submission in os.listdir(os.path.join(input_folder, str(year))):
            all_results[dataset]['submissions'].append(submission)
            avg_mean, avg_median, avg_percentage_correct, avg_percentage_tolerance = eval_submission(input_folder, output_folder, year, submission, dataset)
            all_results[dataset]['scores'][(submission, 'Average absolute error')] = str(round(avg_mean, 3))
            all_results[dataset]['scores'][(submission, 'Median absolute error')] = str(round(avg_median, 3))
            all_results[dataset]['scores'][(submission, 'Percentage of correct segments')] = str(round(avg_percentage_correct, 3))
            all_results[dataset]['scores'][(submission, 'Percentage of correct onsets with tolerance')] = str(round(avg_percentage_tolerance, 3))
    return all_results

