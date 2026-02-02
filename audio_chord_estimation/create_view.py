import os

CHORD_METRICS = ['MirexRoot', 'MirexMajMin', 'MirexMajMinBass', 'MirexSevenths', 'MirexSeventhsBass', 'MeanSeg', 'UnderSeg', 'OverSeg']
def create_view(output_folder):
    submissions = []
    scores = {}
    for metric in os.listdir(os.path.join(output_folder)):
        metric_folder = os.path.join(output_folder, metric)
        for file in os.listdir(metric_folder):
            if file.endswith('.txt'):
                submission_id = file[:-4]
                if 'Chordino' in submission_id or 'ISMIR2019' in submission_id:
                    submission_id = 'Baseline: ' + submission_id
                if submission_id not in submissions:
                    submissions.append(submission_id)
                f = open(os.path.join(metric_folder, file), 'r')
                lines = [line.strip() for line in f.readlines() if line.strip()]
                f.close()
                for line in lines:
                    if line.startswith('Average score: '):
                        score = float(line[len('Average score: '):-1])
                        score_key = (submission_id, metric.replace('Results', ''))
                        scores[score_key] = score
                    elif line.startswith('Average combined Hamming measure (harmonic): '):
                        score = float(line[len('Average combined Hamming measure (harmonic): '):])
                        score_key = (submission_id, 'MeanSeg')
                        scores[score_key] = score * 100
                    elif line.startswith('Average under-segmentation: '):
                        score = float(line[len('Average under-segmentation: '):])
                        score_key = (submission_id, 'UnderSeg')
                        scores[score_key] = score * 100
                    elif line.startswith('Average over-segmentation: '):
                        score = float(line[len('Average over-segmentation: '):])
                        score_key = (submission_id, 'OverSeg')
                        scores[score_key] = score * 100
    submissions.sort(key=lambda x: (x.startswith('Baseline: '), x))
    for score_key in scores:
        scores[score_key] = str(round(scores[score_key], 2))
    return {'scores': scores, 'submissions': submissions, 'metrics': CHORD_METRICS}