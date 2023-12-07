# %%
import numpy as np
import polars as pl
import polars.selectors as cs
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from lingua import LanguageDetectorBuilder  # type: ignore
from polars import col as c


# %%
def deduplicate_by_majority_vote(df):
    columns = [
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
    ]

    return df.group_by("comment").agg(
        cs.by_name(columns).mean().round().cast(int), cs.all().exclude(columns).first()
    )


def implies(a, b):
    return (c(a) == 0) | (c(b) == 1)


def load_main_dataset(path="data/all_DEFR_comments_27062022.csv"):
    df = pl.read_csv(path, dtypes={"ArticleID": pl.Utf8, "ID": pl.Utf8})

    df = (
        df.drop("ArticleID", "ID", "toxische_sprache")
        .drop_nulls()
        # Standardize column names
        .rename(
            {
                # Note that we use the original comments, not the preprocessed ones
                "kommentar_original": "comment",
                "kommentar": "comment_preprocessed_legacy",
                "geschlecht": "gender",
                "alter": "age",
                "sexualitaet": "sexuality",
                "religion": "religion",
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
    )

    df = deduplicate_by_majority_vote(df)

    non_target_columns = ["toxic", "comment", "comment_preprocessed_legacy"]

    return (
        df.with_columns(
            targeted=pl.sum_horizontal(cs.all().exclude(non_target_columns)) > 0
        )
        # Enforce label consistency (toxic is superset of targeted)
        .filter(implies("targeted", "toxic"))
        # Cast all the yes/no variables to long
        .cast({cs.numeric(): pl.Int64, cs.boolean(): pl.Int64})
    )


def load_evaluation_representative(path="data/set0_AK_SK_BW.xlsx"):
    df = pl.read_excel(path)

    df = (
        df.drop(
            "ArticleID",
            "ID",
            "Toxische Sprache",
            "Check_HS",
            "Lang",
            "TopicDD",
            "Text",
            "Title",
        )
        .drop_nulls()
        # Standardize column names
        .rename(
            {
                "Kommentar": "comment",
                "Geschlecht": "gender",
                "Alter": "age",
                "Sexualität": "sexuality",
                "Religion": "religion",
                "Nationalität/Hautfarbe/Herkunft": "nationality",
                "Geistige/körperliche Beeinträchtigung": "disability",
                "Sozialer Status/Bildung/Einkommen/Berufsgruppe": "social_status",
                "Politische Einstellung": "political_views",
                "Aussehen/körperliche Merkmale": "appearance",
                "Andere": "other",
            }
        )
        # Fix the toxicity labels
        .rename({"IsHateSpeech": "toxic"})
    )

    df = deduplicate_by_majority_vote(df)

    non_target_columns = ["toxic", "comment"]

    return (
        df.with_columns(
            targeted=pl.sum_horizontal(cs.all().exclude(non_target_columns)) > 0
        )
        # Enforce label consistency (toxic is superset of targeted)
        .filter(implies("targeted", "toxic"))
        # Cast all the yes/no variables to long
        .cast({cs.numeric(): pl.Int64, cs.boolean(): pl.Int64})
    )


def load_evaluation_expert(path="data/experts_for_annots_16062023_master.xlsx"):
    df = pl.read_excel(path)

    df = (
        df.drop(
            "ArticleID",
            "ID",
            "Titel",
            "Text",
            "Hate Speech_KD",
            "Target Group_KD",
            "Hate Speech_FG",
            "Target Group_FG",
            "Hate Speech_SK",
            "Target Group_SK",
            "Initial Agreement",
            "Difficult case?",
        )
        # Standardize column names
        .rename(
            {
                "Kommentar": "comment",
                "Konsensus Target 1": "target_1",
                "": "target_2",
            }
        )
        # Fix the toxicity labels
        .rename({"Konsensus HS": "toxic"})
        .drop_nulls(["comment", "toxic"])
    )

    def targeted_to(target):
        return (
            pl.when((c("target_1") == target) | (c("target_2") == target))
            .then(1)
            .otherwise(0)
        )

    df = df.with_columns(
        gender=targeted_to("Geschlecht"),
        age=targeted_to("Alter"),
        sexuality=targeted_to("Sexualität"),
        religion=targeted_to("Religion"),
        nationality=targeted_to("Nationalität/Hautfarbe/Herkunft"),
        disability=targeted_to("Geistige/körperliche Beeinträchtigung"),
        social_status=targeted_to("Sozialer Status/Bildung/Einkommen/Berufsgruppe"),
        political_views=targeted_to("Politische Einstellung"),
        appearance=targeted_to("Aussehen/körperliche Merkmale"),
        other=targeted_to("Andere"),
    )

    non_target_columns = ["toxic", "comment"]

    return (
        df.drop("target_1", "target_2")
        .with_columns(
            targeted=pl.sum_horizontal(cs.all().exclude(non_target_columns)) > 0
        )
        # Enforce label consistency (toxic is superset of targeted)
        .filter(implies("targeted", "toxic"))
        # Cast all the yes/no variables to long
        .cast({cs.numeric(): pl.Int64, cs.boolean(): pl.Int64})
    )


# %%
df_everything = load_main_dataset()
df_evaluation_representative = load_evaluation_representative()
df_evaluation_expert = load_evaluation_expert()

# Make sure df_everything actually contains everything, keep the expert opinions
df_everything = pl.concat(
    [
        df_everything.with_columns(source=pl.lit("main")).join(
            df_evaluation_expert, on="comment", how="anti"
        ),
        df_evaluation_representative.with_columns(source=pl.lit("representative")),
        df_evaluation_expert.with_columns(source=pl.lit("expert")),
    ],
    how="diagonal",
)

# %%
# Check: toxic is superset of targeted
assert ((df_everything["toxic"] == 0) & (df_everything["targeted"] == 1)).sum() == 0

# Check: no comment has targets but is not labeled as targeted
assert (
    df_everything.filter(
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
def add_language_column(df: pl.DataFrame):
    def language(confidence_list):
        top = [c for c in sorted(confidence_list, key=lambda c: c.value, reverse=True)]
        return top[0].language.name, top[0].value

    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    detector_confidence_list = detector.compute_language_confidence_values_in_parallel(
        df["comment"]
    )

    languages, confidences = [], []
    for confidence_list in detector_confidence_list:
        lang, confidence = language(confidence_list)
        languages.append(lang)
        confidences.append(confidence)

    return df.with_columns(
        language=pl.Series(languages),
        language_confidence=pl.Series(confidences),
    )


df_everything = add_language_column(df_everything)
# We know that the expert evaluation is in German, so we can just overwrite the language
df_everything = df_everything.with_columns(
    language=pl.when(c("source") != "main")
    .then(pl.lit("GERMAN"))
    .otherwise(c("language")),
    language_confidence=pl.when(c("source") != "main")
    .then(pl.lit(1.0))
    .otherwise(c("language_confidence")),
)

# %%
# Remove the expert evaluation and the representative evaluation from the main dataset
df_full_main = df_everything.join(
    df_evaluation_representative, on="comment", how="anti"
).join(df_evaluation_expert, on="comment", how="anti")
assert (df_full_main["source"] == "main").all()
# Remove non-german comments
df_full_main = df_full_main.filter(c("language") == "GERMAN")

# %%
# Split data into train, eval, test
SEED = 42

# Randomly split to 20 folds, one fold is thus 5% of the data
stratifier = MultilabelStratifiedKFold(20, shuffle=True, random_state=SEED)

comments = df_full_main.select("comment")
labels = df_full_main.select(
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

# Train = 17*5% = 85%, Eval = 3*5% = 15% of the original dataset
train_split_idx = np.concatenate(splits[:17])
evaluation_original_split_idx = np.concatenate(splits[17:])

df_training_main = df_full_main[train_split_idx]
df_evaluation_main = df_full_main[evaluation_original_split_idx]


# %%
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


# You can manually check that the ratios for label combinations are similar
# among train/eval/test sets, e.g.:
#   check_ratio(train_data, ["gender", "age"])
#   check_ratio(evaluation_data, ["gender", "age"])
#   check_ratio(test_data, ["gender", "age"])

# %%
# Bump VERSION if we do any changes to the code
VERSION = 4

df_everything.write_csv(f"out/processed_everything_v{VERSION}.csv")

df_full_main.write_csv(f"out/processed_full_main_v{VERSION}.csv")
df_training_main.write_csv(f"out/processed_training_main_v{VERSION}.csv")
df_evaluation_main.write_csv(f"out/processed_evaluation_main_v{VERSION}.csv")

df_evaluation_expert.write_csv(f"out/processed_evaluation_expert_v{VERSION}.csv")
df_evaluation_representative.write_csv(
    f"out/processed_evaluation_representative_v{VERSION}.csv"
)
