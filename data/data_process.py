import kagglehub
import os
import pandas as pd
from my_logger import logger
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Download latest version
path = kagglehub.competition_download("titanic")
logger.info(f"Path to competition files: {path}")

train_csv = os.path.join(path, "train.csv")
pd_raw = pd.read_csv(train_csv)


def preview_dataset():
    files = os.listdir(path)

    logger.info(type(files))

    # ['test.csv', 'train.csv', 'gender_submission.csv']
    logger.info(files)

    train_csv = os.path.join(path, "train.csv")
    pd_train = pd.read_csv(train_csv)

    # check pd
    rows = len(pd_train)
    logger.info(f"rows total cnt: {rows}")

    missing = pd.isna(pd_train).sum()
    logger.info("pd_trian missiong -> %s", missing)

    #  Cabin, Age, Embarked are missing
    ages = pd_train["Age"].unique()
    cabins = pd_train["Cabin"].unique()
    embarks = pd_train["Embarked"].unique()

    logger.info(f"ages -> {ages}")
    logger.info(f"cabins -> {cabins}")
    logger.info(f"embark -> {embarks}")

    logger.info(type(pd_train))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    logger.info(f" head 1 => {pd_train.head(2)}")

    logger.info(f" columns => {pd_train.columns}")
    print(pd_train.head(10).to_markdown())


# preview dataset here
preview_dataset()


def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_csv = os.path.join(path, "train.csv")
    pd_train = pd.read_csv(train_csv)

    pd_train, pd_validate = train_test_split(
        pd_train, test_size=0.2, random_state=42, shuffle=True
    )

    test_csv = os.path.join(path, "test.csv")
    pd_test = pd.read_csv(test_csv)

    return pd_train, pd_validate, pd_test


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    #  map dataFrame to matrix
    df.drop("Survived", axis=1)
    df.drop("PassengerId", axis=1)
    df.drop("Cabin", axis=1)
    df.drop("Name", axis=1)
    df.drop("Ticket", axis=1)

    age_median = df["Age"].median()
    logger.info(f"age_median: {age_median}")
    df["Age"] = df["Age"].fillna(age_median)

    # df["HasCabin"] = df["Cabin"].apply(lambda v: 0 if pd.isna(v) else 1 )
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    df["Sex"] = df["Sex"].apply(lambda s: 0 if s == "male" else 1)

    pd.get_dummies(
        df,
        columns=[
            "Sex",
            "Embarked",
        ],
    )

    return df


# test
processed = process_features(pd_raw)
print(processed.head(4).to_markdown())



def write_sub_csv(map: dict[int, int], force: bool = False):

    if not force:
        out_file_name = "submission.csv"
        cur = os.path.curdir
        out_file = os.path.join(cur, out_file_name)

        if out_file:
            logger.info("submission file alread exit, skip override")
            return

    else:
        logger.info(f"force override submission file, map len: ${len(map)}")
        submissions = pd.DataFrame(
            {
                "PassengerId": map.keys,
                "Survived": map.values,
            }
        )
        submissions.to_csv("submission.csv", index=False)
