import argparse
import os
from datetime import datetime
from utils import print_table, create_mediawiki_table

from audio_chord_estimation.evaluate import main as audio_chord_estimation_evaluate
from audio_beat_detection.evaluate import main as audio_beat_detection_evaluate
from audio_key_detection.evaluate import main as audio_key_detection_evaluate
from music_structure_analysis.evaluate import main as music_structure_analysis_evaluate
from lyrics_alignment.evaluate import main as lyrics_alignment_evaluate

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('task_name', type=str, help='MIREX task name, e.g., audio_chord_estimation')
    args.add_argument('input_folder', type=str, help='Input folder containing the dataset')
    args.add_argument('--output_folder', type=str, help='Output folder to save the results', default='output')
    args.add_argument('--year', type=int, help='Year of the evaluation', default=datetime.now().year)
    args.add_argument('--output_wiki', action='store_true', help='If set, outputs the results in MediaWiki format')
    args = args.parse_args()
    # Load the task from the submodule
    evaluate = globals().get(f'{args.task_name}_evaluate', None)
    if evaluate is None:
        raise ValueError(f'Unknown task name: {args.task_name}')
    os.makedirs(args.output_folder, exist_ok=True)
    results = evaluate(input_folder=args.input_folder, output_folder=os.path.join(args.output_folder, args.task_name), year=args.year)
    for dataset in results:
        if args.output_wiki:
            print(create_mediawiki_table(dataset, results[dataset]['scores'], results[dataset]['metrics'], results[dataset]['submissions']))
        else:
            print_table(dataset, results[dataset]['scores'], results[dataset]['metrics'], results[dataset]['submissions'])