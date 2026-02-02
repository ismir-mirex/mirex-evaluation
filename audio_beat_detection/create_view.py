import os

BEAT_METRICS = [
    ['F1', 'F-measure'],
    ['Cemgil', 'Cemgil'],
    ['Goto', 'Goto'],
    ['P-score', 'P-score'],
    ['CMLc', 'Correct Metric Level Continuous'],
    ['CMLt', 'Correct Metric Level Total'],
    ['AMLc', 'Any Metric Level Continuous'],
    ['AMLt', 'Any Metric Level Total'],
]
def create_view(output_folder):
    submissions = []
    scores = {}
    for file in os.listdir(output_folder):
        if file.endswith('.csv'):
            submission_id = file[:-4]
            if 'CD1' in submission_id or 'BeatThis' in submission_id:
                submission_id = 'Baseline: ' + submission_id
            if submission_id not in submissions:
                submissions.append(submission_id)
            f = open(os.path.join(output_folder, file), 'r')
            lines = [line.strip() for line in f.readlines() if line.strip()]
            f.close()
            keys = lines[0].split(',')
            for line in lines:
                if line.startswith('#avg'):
                    data = line.split(',')
                    for metric in BEAT_METRICS:
                        index = keys.index(metric[1])
                        score = float(data[index])
                        scores[(submission_id, metric[0])] = score * 100
    submissions.sort(key=lambda x: (x.startswith('Baseline: '), x))
    for score_key in scores:
        scores[score_key] = str(round(scores[score_key], 2))
    return {'scores': scores, 'submissions': submissions, 'metrics': [m[0] for m in BEAT_METRICS]}
