import importlib.metadata
from packaging import version
import os
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
# Get installed version of mir_eval
mir_eval_version = importlib.metadata.version("mir_eval")
# Assert that it's at least 0.8.2
assert version.parse(mir_eval_version) >= version.parse("0.8.2"), f"mir_eval {version} is too old, please upgrade to >=0.8.2"
print(f"mir_eval {version} is OK (>=0.8.2)")
import mir_eval.beat
from .create_view import create_view

def do_job(command):
    subprocess.run(command)
def evaluate_beat(ref_folder, pred_folder, output_file):
    files = os.listdir(ref_folder)
    def perform_evaluation(file):
        ref_file = os.path.join(ref_folder, file)
        pred_file = os.path.join(pred_folder, file.replace('.txt', '.lab'))
        print('Evaluating', ref_file, pred_file)
        reference_beats = mir_eval.io.load_events(ref_file)
        try:
            estimated_beats = mir_eval.io.load_events(pred_file)
        except:
            print('Warning: Could not load', pred_file)
            estimated_beats = np.array([])
        if os.path.basename(pred_folder) == 'CD1':
            # To keep consistency with the previous years, we round the beat times to 4 decimal places
            estimated_beats = np.round(estimated_beats, 4)
        evaluation = mir_eval.beat.evaluate(reference_beats, estimated_beats)
        evaluation['id'] = file
        return evaluation
    all_scores = Parallel(n_jobs=-1)(delayed(perform_evaluation)(file) for file in tqdm(files))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    f = open(output_file, 'w')
    keys = ['id'] + [key for key in all_scores[0].keys() if key != 'id']
    f.write(','.join(keys) + '\n')
    for i, scores in enumerate(all_scores):
        line = [str(scores[key]) for key in keys]
        f.write(','.join(line) + '\n')
    # Average the scores
    avg_result = ['#avg']
    for key in keys[1:]:
        avg_result.append(str(sum([scores[key] for scores in all_scores]) / len(all_scores)))
    f.write(','.join(avg_result) + '\n')

def main(input_folder, output_folder, year):
    datasets = os.listdir(os.path.join(input_folder, str(year)))
    for dataset in datasets:
        for submission in os.listdir(os.path.join(input_folder, str(year), dataset)):
            submission_folder = os.path.join(input_folder, str(year), dataset, submission)
            ground_truth_folder = os.path.join(input_folder, 'Ground-Truth', dataset)
            output_file = os.path.join(output_folder, str(year), dataset, submission + '.csv')
            evaluate_beat(
                ref_folder=ground_truth_folder,
                pred_folder=submission_folder,
                output_file=output_file
            )
    results = {}
    for dataset in datasets:
        results[dataset] = create_view(os.path.join(output_folder, str(year), dataset))
    return results