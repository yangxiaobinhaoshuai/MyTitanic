from data.data_process import get_dataset
from my_logger import logger
from model.my_model import TiModel
from torch.optim import Adam


eopches = 100
lr = 0.001


def train():

    pd_train, pd_test = get_dataset()
    logger.info(f"pd_train shape:{pd_train.shape}")
    logger.info(f"pd_test shape:{pd_test.shape}")

    model = TiModel()
    

    # optimizer = Adam()

    pass
