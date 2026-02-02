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

## Evaluation tool

Do not use the prebuilt binaries for MIREX 2015 (they have been changed). Use either the binary in this repo or build from source.
```bash
git config --global url."https://github.com/".insteadOf "git://github.com/"
git clone --recursive https://github.com/jpauwels/MusOOEvaluator
```

## References

https://github.com/jpauwels/MusOOEvaluator