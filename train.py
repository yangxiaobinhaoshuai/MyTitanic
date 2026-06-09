from data.data_process import get_processed_data
from my_logger import logger
from xgboost import XGBClassifier


lr = 0.001


def train():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
    ) = get_processed_data()

    logger.info(f"X_train shape:{X_train.shape}")
    logger.info(f"y_train shape:{y_train.shape}")
    logger.info(f"X_test shape:{X_test.shape}")


    # model = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=lr)

    # model.fit(
    #     pd_train,
    # )

    # optimizer = Adam()

    pass


train()