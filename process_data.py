# %%
import numpy as np
import polars as pl
import polars.selectors as cs
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lingua import Language, LanguageDetectorBuilder
from polars import col as c

# %%
path = "data/all_DEFR_comments_27062022.csv"
df = pl.read_csv(path, dtypes={"ArticleID": pl.Utf8, "ID": pl.Utf8})


# %%
def implies(a, b):
    return (c(a) == 0) | (c(b) == 1)


df = (
    df.drop_nulls().unique()
    # Drop unnecessary columns
    .drop("ArticleID", "ID", "kommentar", "toxische_sprache")
    # Standardize column names
    .rename(
        {
            # Note that we use the original comments, not the preprocessed ones
            "kommentar_original": "comment",
            "geschlecht": "gender",
            "alter": "age",
            "sexualitaet": "sexuality",
            "nationalitaet": "nationality",
            "beeintraechtigung": "disability",
            "sozialer_status": "social_status",
            "politik": "political_views",
            "aussehen": "appearance",
            "andere": "other",
        }
    )
    # Fix the toxicity labels
    .rename({"label": "toxic"})
    # Create a new column to indicate if the toxicity is targeted (number of targets > 0)
    .with_columns(
        targeted=pl.sum_horizontal(cs.all().exclude(["toxic", "comment"])) > 0
    )
    # Enforce label consistency (toxic is superset of targeted)
    .filter(implies("targeted", "toxic"))
    # Cast all the yes/no variables to long
    .cast({cs.numeric(): pl.Int64, cs.boolean(): pl.Int64})
    # Add a column with the row id
    .with_row_count("id")
)


# %%
# Check: toxic is superset of targeted
assert ((df["toxic"] == 0) & (df["targeted"] == 1)).sum() == 0

# Check: no comment has targets but is not labeled as targeted
assert (
    df.filter(
        (
            c("gender")
            + c("age")
            + c("sexuality")
            + c("religion")
            + c("nationality")
            + c("disability")
            + c("social_status")
            + c("political_views")
            + c("appearance")
            + c("other")
        )
        == 0
    )["targeted"].sum()
    == 0
)

# %%
LANGUAGES = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.ITALIAN]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()


def detect_language(comment):
    lang = DETECTOR.detect_language_of(comment)
    return lang.name if lang is not None else "UKNOWN"


with open("data/language_list.csv", "w") as f:
    f.write("language\n")
    for i, comment in enumerate(df["comment"]):
        if i % 10000 == 0:
            print(i)
        f.write(detect_language(comment) + "\n")

# %%
language_df = pl.read_csv("data/language_list.csv")
df = df.with_columns(language=language_df["language"]).filter(c("language") == "GERMAN")


# %%
# Split data into train, eval, test
SEED = 42

# Randomly split to 20 folds, one fold is thus 5% of the data
stratifier = MultilabelStratifiedKFold(20, shuffle=True, random_state=SEED)

comments = df.select("comment")
labels = df.select(
    "toxic",
    "gender",
    "age",
    "sexuality",
    "religion",
    "nationality",
    "disability",
    "social_status",
    "political_views",
    "appearance",
    "other",
)
splits = [s for _, s in stratifier.split(comments, labels)]  # type: ignore

# Train = 14*5% = 70%, Eval = 3*5% = 15%, Test = 3*5% = 15% of the original dataset
train_split_idx = np.concatenate(splits[:14])
evaluation_split_idx = np.concatenate(splits[14:17])
test_split_idx = np.concatenate(splits[17:])

train_data = df[train_split_idx]
evaluation_data = df[evaluation_split_idx]
test_data = df[test_split_idx]


# %%
# You can manually check that the ratios for label combinations are similar
# among train/eval/test sets, e.g.:
#   check_ratio(train_data, ["gender", "age"])
#   check_ratio(evaluation_data, ["gender", "age"])
#   check_ratio(test_data, ["gender", "age"])
def check_ratio(data: pl.DataFrame, labels: list[str]):
    combination = "+".join(labels)
    return (
        data.with_columns(
            (pl.sum_horizontal(c(labels)) == len(labels)).alias(combination)
        )
        .select(combination)
        .to_series()
        .value_counts()
        .with_columns(ratio=c("counts") / len(data))
        .sort(by=combination)
    )


# %%
# Bump VERSION if we do any changes to the code
VERSION = 2
df.write_csv(f"data/processed_comments_all_v{VERSION}.csv")
train_data.write_csv(f"data/processed_comments_train_v{VERSION}.csv")
evaluation_data.write_csv(f"data/processed_comments_evaluation_v{VERSION}.csv")
test_data.write_csv(f"data/processed_comments_test_v{VERSION}.csv")

# %%
