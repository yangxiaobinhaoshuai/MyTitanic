from data.data_process import get_dataset
from my_logger import logger
from model.my_model import TiModel
from xgboost import XGBClassifier


lr = 0.001


def train():
    
    pd_train, pd_validate, pd_test = get_dataset()

    logger.info(f"pd_train shape:{pd_train.shape}")
    logger.info(f"pd_test shape:{pd_test.shape}")

    model = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=lr)

    # model.fit(
    #     pd_train,
    # )

    # optimizer = Adam()

    pass
