import argparse

from data.data_process import write_sub_csv

from train import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    train()

    
    
    # Test 
    # write_sub_csv({})
