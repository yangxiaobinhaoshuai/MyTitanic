import kagglehub
import os
import pandas as pd
from my_logger import logger

# Download latest version
path = kagglehub.competition_download("titanic")

logger.info(f"Path to competition files: {path}")


files = os.listdir(path)

logger.info(type(files))
# ['test.csv', 'train.csv', 'gender_submission.csv']
logger.info(files)

train_csv = os.path.join(path, "train.csv")
pd_train = pd.read_csv(train_csv)
logger.info(type(pd_train))
logger.info(pd_train.head(1))

test_csv = os.path.join(path, "test.csv")
pd_test = pd.read_csv(test_csv)
logger.info(type(pd_test))
logger.info(pd_test.head(1))


def get_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:

    train_csv = os.path.join(path, "train.csv")
    pd_train = pd.read_csv(train_csv)

    test_csv = os.path.join(path, "test.csv")
    pd_test = pd.read_csv(test_csv)

    return pd_train, pd_test


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
