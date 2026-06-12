from data.data_process import get_processed_data
from my_logger import logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd

lr = 0.1


def train():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        test_passender_id,
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
    logger.info(f"test acc : {pred_test}")


# TODO
# train()


#  TODO

# def convert_to_map(
#     y_test: pd.Series, test_passenger_id: pd.Series
# ) -> dict[int, int]: 
#     map = {}
#     for idx,y in y_test.items():
#         map[test_passenger_id.iloc[idx]] = y

#         return map
        


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
        #  Force override branch
        logger.info(f"force override submission file, map len: ${len(map)}")
        submissions = pd.DataFrame(
            {
                "PassengerId": map.keys,
                "Survived": map.values,
            }
        )
        submissions.to_csv("submission.csv", index=False)
