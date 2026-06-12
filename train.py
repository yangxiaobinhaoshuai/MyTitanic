from data.data_process import get_processed_data
from my_logger import logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


lr = 0.1


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

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=lr)
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, pred_val)
    logger.info(f"val acc : {acc}")

    # pred_test = model.predict(X_test)
    # logger.info(f"test acc : {pred_test}")


train()
