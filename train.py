from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from data.data_process import get_processed_data
from my_logger import logger

lr = 0.1


# Useless for now
def write_sub_csv(test_passenger_id: pd.Series, y_test: pd.Series, force: bool = False):

    submission_path: Path = Path("submission.csv")

    if not force:
        if submission_path.exists():
            logger.info("submission file already exist, skip override")
            return

    #  Force override branch
    logger.info("New or override submission file.")

    res_dict = {"PassengerId": test_passenger_id, "Survived": y_test}

    pd.DataFrame(res_dict).to_csv(submission_path, index=False)


def train():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        test_passenger_id,
    ) = get_processed_data()

    logger.info(f"X_train shape:{X_train.shape}")
    logger.info(f"y_train shape:{y_train.shape}")
    logger.info(f"X_test shape:{X_test.shape}")

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=lr)
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)

    acc = accuracy_score(y_val, pred_val)
    logger.info(f"val acc : {acc}")

    pred_test = model.predict(X_test)
    logger.info(f"y_test prediction: {pred_test}")

    # Write to csv
    write_sub_csv(test_passenger_id, pred_test, force=True)




# Redundant for now
def convert_to_map(y_test: pd.Series, test_passenger_id: pd.Series) -> dict[int, int]:
    return dict(zip(test_passenger_id, y_test))


if __name__ == "__main__":
    train()