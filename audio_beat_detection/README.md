## Folder structure

Use the following folder structure for evaluation:
```
input_folder
├── Ground-Truth
│   ├── Dataset1
│   │   ├── track1.lab
│   │   └── ...
│   └── ...
├── 2025 (args.year)
│   ├── Submission1
│   │   ├── Dataset1
│   │   │   ├── track1.lab
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

## Baselines
### Baseline 1: Code Name `Beat This!`

#### Installation

```bash
git clone https://github.com/CPJKU/beat_this.git
cd beat_this
pip install https://github.com/CPJKU/beat_this/archive/main.zip
```

#### Usage

```bash
beat_this path/to/whole_directory/ -o path/to/output_directory
```

### Baseline 2: CD1 (vamp plugin)
```bash
sonic-annotator \
  -d vamp:qm-vamp-plugins:qm-tempotracker:beats \
  *.wav \
  -w csv --csv-basedir output/
```
