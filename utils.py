import pandas as pd

def create_mediawiki_table(dataset, scores, metrics, submissions):
    result_lines = []
    result_lines.append(f'== {dataset} ==')
    result_lines.append('')
    result_lines.append('{| class="wikitable" style="text-align:right;')
    result_lines.append('|- style="font-weight:bold; text-align:left;"')
    result_lines.append('! style="vertical-align:bottom;" | Group')
    for metric in metrics:
        result_lines.append(f'! style="vertical-align:bottom;" | {metric}')
    for submission in submissions:
        result_lines.append('|- style="vertical-align:bottom;"')
        result_lines.append(f'| style="text-align:left;" | {submission}')
        for metric in metrics:
            if (submission, metric) in scores:
                result_lines.append(f'| {scores[(submission, metric)]}')
            else:
                result_lines.append('| N/A')
    result_lines.append('|}')
    result_lines.append('')
    return result_lines

def build_dataframe(scores, submissions, metrics=None):
    data = []
    for submission in submissions:
        row = {"Group": submission}
        for metric in metrics:
            row[metric] = scores.get((submission, metric))
        data.append(row)

    df = pd.DataFrame(data)
    df.set_index("Group", inplace=True)
    return df

def print_table(dataset, scores, metrics, submissions, decimals=2):
    df = build_dataframe(scores, submissions, metrics)

    print()
    print(dataset)
    print("=" * len(dataset))

    # Round for display
    df_disp = df.round(decimals)

    # Replace NaN with "N/A" for console output
    df_disp = df_disp.fillna("N/A")

    print(df_disp.to_string())
    print()
