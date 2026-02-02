import os
from .eval_script import evaluate

def main(input_folder, output_folder, year):
    datasets = os.listdir(os.path.join(input_folder, str(year)))
    all_results = {'HarmonixSet': {'scores': {}, 'metrics': ['ACC', 'HR.5_F', 'HR3_F'], 'submissions': []}}
    for submission in os.listdir(os.path.join(input_folder, str(year))):
        results = evaluate(gt_file=os.path.join(input_folder, 'harmonixset.corrected.20250821.jsonl'),
                           pred_file=os.path.join(input_folder, str(year), submission))
        submission_name = submission[:-6]
        all_results['HarmonixSet']['submissions'].append(submission_name)
        for metric in all_results['HarmonixSet']['metrics']:
            all_results['HarmonixSet']['scores'][(submission_name, metric)] = str(round(results[metric], 2))
    return all_results