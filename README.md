# Data Processing Pipeline for Hate Speech Recognition

Process to reproduce:

1. Get `all_comments_lang.csv` from Dropbox. This file is the same as `all_DEFR_comments_27062022.csv`, only with lang labels generated by `langdetect` added as an additional column.
2. Place `all_comments_lang.csv` into folder `data`.
3. Make sure you are running Python 3.11.2 for full reproducibility. The recommended way to manage different Python versions is [pyenv](https://github.com/pyenv/pyenv).
4. Run `bash run-processing.sh`. This should download all the necessary dependencies and run the processing script.
5. Check the `data` folder. It should have the files `processed_comments_{train,evaluation,test}_v*.csv`.

## Explanation of the generated data splits

The code generates three different data splits.

- `train` is the **only** set that should be used for training. I.e. even if your training process involves (cross) validation, use only this dataset to perform it.
- `evaluation` should be used for evaluating your models and comparison with other teams' models. You shouldn't optimize too hard for this set to prevent over-inflating your scores and keep the comparison with other teams realistic.
- `test` is an actual held-out test set. During the modelling phase, it should only be looked at sparsely or not at all to prevent even accidental data leakage. It should only be used near the end of the project to estimate the real-world performance of the models we trained and selected using the other two data splits.

## Making changes

**IMPORTANT:** Whenever you make changes to the code, before running it, bump the `VERSION` variable in `process_data.py` by 1. This ensures that we never accidentally use differently processed data, and prevents data leakage and other problems.
