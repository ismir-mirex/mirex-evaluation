# MIREX Evaluation

![MIREX Logo](https://music-ir.org/mirex/common/mirex_logo.png)

This is the open-source repo for all tasks evaluated by MIREX. It contains the following content:

* The evaluation code for each task (excluding those with proprietary code).
* Information about the baselines used in each task.
* Information about the datasets used in each task.

The repo will be updated each year to reflect changes in the evaluation process. If you want to see a specific year's codebase, please check out to the branch for each year (TBA).

## Installation

1. Clone the repository
2. Install python 3.10
3. Install pandas: ``pip install pandas>=1.5.3``
4. Indivudual tasks might need additional dependencies. Please see task-specific READMEs. Common ones include: ``numpy>=1.26.4``, ``mir_eval>=0.8.2``

## Usage

```bash
python main.py <task_name> <input_folder> [--output_folder] [--year=<year>] [--output_wiki]
```

Please refer to task-specific README for details on the folder structure of `<input_folder>` and baselines.

## Supported Tasks

| Task Name                | README                                     | Evaluation | Baselines |
|--------------------------|--------------------------------------------|------------|-----------|
| audio_chord_estimation   | [Link](audio_chord_estimation/README.MD)   | ✅          | ✅         |
| audio_beat_detection     | [Link](audio_beat_detection/README.MD)     | ✅          | ✅         |
| audio_downbeat_detection | [Link](audio_downbeat_detection/README.MD) | TBA        | TBA         |
| audio_key_detection      | [Link](audio_key_detection/README.MD)      | TBA        | TBA       |
| music_structure_analysis | [Link](music_structure_analysis/README.MD) | ✅          | ✅         |
| lyrics_alignment         | [Link](lyrics_alignment/README.MD)         | ✅          | ✅         |

## Example Usages

```bash
python main.py audio_chord_estimation ../MIREXEval/ace-output --year=2025 
```

Output:
```
Billboard2012
=============
                   MirexRoot MirexMajMin MirexMajMinBass MirexSevenths MirexSeventhsBass MeanSeg UnderSeg OverSeg
Group                                                                                                            
BMACE                  58.47        9.15            9.02          2.92              2.87   69.55    92.02   57.41
MD1                    85.14       84.01           82.79         74.14             73.14    88.8    88.63   89.78
YK1                     85.9       84.67           81.83         77.22             74.46   88.43    87.88   89.58
wu-ensemble            78.28       77.17           75.63         60.01             58.81   84.42    87.98   82.57
wu-single              79.25       78.23            76.8         60.25             59.09   84.87    87.19   84.24
Baseline: Chordino     74.21       72.29           70.28         55.43             53.47   83.69    85.33   83.48
```

## Contact

For general questions, please contact us via

* Email: future-mirex@googlegroups.com
* Discord: https://discord.gg/vC2YWX29sC

MIREX 2025 Organizers:

* Junyan Jiang, New York University
* Gus Xia, MBZUAI
* Akira Maezawa, Yamaha
* Ziyu Wang, New York University
* Yixiao Zhang, ByteDance Inc.
* Ruibin Yuan, Hong Kong University of Science and Technology
* J. Stephen Downie, University of Illinois
