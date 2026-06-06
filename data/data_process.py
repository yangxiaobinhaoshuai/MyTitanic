import kagglehub
import os
import pandas as pd
from my_logger import logger
from sklearn.model_selection import train_test_split


# Download latest version
path = kagglehub.competition_download("titanic")
logger.info(f"Path to competition files: {path}")

files = os.listdir(path)
logger.info(type(files))
# ['test.csv', 'train.csv', 'gender_submission.csv']
logger.info(files)

train_csv = os.path.join(path, "train.csv")
pd_raw = pd.read_csv(train_csv)


def preview_raw_dataset(pd_param: pd.DataFrame):
    # check pd
    rows = len(pd_param)
    logger.info(f"rows total cnt: {rows}")

    missing = pd.isna(pd_param).sum()
    logger.info("pd_trian missiong -> %s", missing)

    #  Cabin, Age, Embarked are missing
    ages = pd_param["Age"].unique()
    cabins = pd_param["Cabin"].unique()
    embarks = pd_param["Embarked"].unique()

    logger.info(f"ages -> {ages}")
    logger.info(f"cabins -> {cabins}")
    logger.info(f"embark -> {embarks}")

    logger.info(type(pd_param))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    logger.info(f" head 1 => {pd_param.head(2)}")

    logger.info(f" columns => {pd_param.columns}")
    print(pd_param.head(4).to_markdown())


# preview raw dataset here
preview_raw_dataset(pd_raw)


def calculate_raw_median(pd: pd.DataFrame, column_name: str) -> int:
    return int(pd[column_name].median())


def process_features(df: pd.DataFrame, median_value: int) -> pd.DataFrame:
    df = df.copy()

    # df["HasCabin"] = df["Cabin"].apply(lambda v: 0 if pd.isna(v) else 1 )
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    #  map dataFrame to matrix
    df = df.drop(
        columns=["Survived", "PassengerId", "Cabin", "Name", "Ticket"], errors="ignore"
    )

    logger.info(f"age_median: {median_value}")
    df["Age"] = df["Age"].fillna(median_value)

    # onehot encoder
    df = pd.get_dummies(
        df,
        columns=[
            "Sex",
            "Embarked",
        ],
        dtype=int,
    )

    return df


def preview_processed_dataset(pdf: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"shape :{pdf.shape}")
    logger.info(f"head 4 : {pdf.head(4)}")
    logger.info(f"Age column: {pdf['Age'].isna().sum()}")
    logger.info(f"Embark column: {pdf['Embarked_S'].isna().sum()}")
    return pdf


def split_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pd_train, pd_val = train_test_split(
        raw, test_size=0.3, random_state=42, shuffle=True
    )

    test_csv = os.path.join(path, "test.csv")
    pd_test = pd.read_csv(test_csv)

    return pd_train, pd_val, pd_test


pd_train, pd_val, pd_test = split_dataset(raw=pd_raw)

#  labelv
pd_y_train = pd_train["Survived"]
pd_y_val = pd_val["Survived"]


train_age_median = calculate_raw_median(pd_train, "Age")

processed_train = process_features(pd_train, train_age_median)
processed_val = process_features(pd_val, train_age_median)
processed_test = process_features(pd_test, train_age_median)

# 缺的列补 0， 多余的丢掉
processed_val = processed_val.reindex(columns=processed_train.columns, fill_value=0)
processed_test = processed_test.reindex(columns=processed_train.columns, fill_value=0)

preview_processed_dataset(processed_train)
preview_processed_dataset(processed_val)
preview_processed_dataset(processed_test)


pd_x_train = processed_train
pd_x_val = processed_val
pd_x_test = processed_test


def get_processed_data() -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame
]:
    return pd_x_train, pd_y_train, pd_x_val, pd_y_val, pd_x_test


# TODO
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
