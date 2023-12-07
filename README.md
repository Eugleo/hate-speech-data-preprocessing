# Data Processing Pipeline for Hate Speech Recognition

Make sure you are running Python 3.11.2 for full reproducibility. The recommended way to manage different Python versions is [pyenv](https://github.com/pyenv/pyenv). Steps to reproduce the processing are then:

1. Download `all_DEFR_comments_27062022.csv` from PolyBox and put it into a folder named `data`.
2. Run `bash run-processing.sh`. This should download all the necessary dependencies and run the processing script.

After these steps the `out` folder will contain the processed files, tagged (in the filename) with a version number.

## Explanation of the generated data splits

The code generates multiple different data splits, some of them hierarchical. See below for details.

- `evaluation_expert` contains the 500 expert-annotated data. This is intended for evaluation only, as a held-out test set.
- `evaluation_representative` contains the first ~25 000 comments which were randomly sampled. This is intended for evaluation only, as a held-out test set.
- `full_main` contains the remaining comments from the dataset that were classified as being german by `lingua`, the library we use for language detection. This set should be used for model development.
  - `training_main` contains 85% of `full_main` and is intended for training and hyperparameter tuning
  - `evaluation_main` contains 15% of `full_main` and is intended for hyperparameter tuning and model comparison with other teams. You shouldn't optimize too hard for this set to prevent over-inflating your scores and keep the comparison with other teams realistic.

Finally, `everything` literally contains everything — all languages and all comments, including the expert and representative evaluation sets. It

## Making changes

**IMPORTANT:** Whenever you make changes to the code, before running it, bump the `VERSION` variable in `process_data.py` by 1. This ensures that we never accidentally use differently processed data, and prevents data leakage and other problems.
