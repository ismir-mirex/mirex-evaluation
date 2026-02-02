import os
import uuid
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm
from .create_view import create_view
import sys


EXECUTABLE = 'MusOOEvaluator.exe' if sys.platform == 'win32' else 'MusOOEvaluator'
def do_job(command):
    subprocess.run(command)
def evaluate_chord(ref_folder, pred_folder, output_path):
    # Generate a uuid for the temp file using [ref_folder, pred_folder]
    os.makedirs('temp', exist_ok=True)
    file_name = os.path.join('temp', str(uuid.uuid5(uuid.NAMESPACE_DNS, ref_folder + pred_folder)) + '.txt')
    files = os.listdir(ref_folder)
    f = open(file_name, 'w')
    for file in files:
        if file.endswith('.lab'):
            f.write(file[:-4] + "\n")
    f.close()
    jobs = []
    for evaluation_metrics in [
        'MirexMajMin',
        'MirexMajMinBass',
        'MirexRoot',
        'MirexSevenths',
        'MirexSeventhsBass',
        'Inner'
    ]:
        for output_format in ['csv', 'txt']:
            command = [
                os.path.join('bin', EXECUTABLE),
                '--chords' if evaluation_metrics != 'Inner' else '--segmentation', evaluation_metrics,
                '--list', file_name.replace('\\', '/'),
                '--refdir', ref_folder.replace('\\', '/'),
                '--refext', '.lab',
                '--testdir', pred_folder.replace('\\', '/'),
                '--testext', '.lab',
            ]
            if output_format == 'csv':
                command += ['--csv']
            output_name = 'Results' + evaluation_metrics if evaluation_metrics != 'Inner' else 'Segmentation'

            os.makedirs(os.path.dirname(output_path % output_name), exist_ok=True)
            command += [
                '--output', output_path % output_name + '.' + output_format,
            ]
            jobs.append(command)
    Parallel(n_jobs=-1)(delayed(do_job)(job) for job in tqdm(jobs))
    try:
        os.unlink(file_name)
    except:
        pass

def main(input_folder, output_folder, year):
    datasets = os.listdir(os.path.join(input_folder, str(year)))
    for dataset in datasets:
        for submission in os.listdir(os.path.join(input_folder, str(year), dataset)):
            submission_folder = os.path.join(input_folder, str(year), dataset, submission)
            output_path = os.path.join(output_folder, str(year), dataset, '%s', submission)
            ground_truth_folder = os.path.join(input_folder, 'Ground-Truth', dataset)
            evaluate_chord(
                ref_folder=ground_truth_folder,
                pred_folder=submission_folder,
                output_path=output_path
            )
    results = {}
    for dataset in datasets:
        results[dataset] = create_view(os.path.join(output_folder, str(year), dataset))
    return results