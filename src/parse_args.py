import argparse

paser = argparse.ArgumentParser(description="Paser for Arguments",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# global setting
paser.add_argument("-data_path", dest="data_path", default="./data/", help="Path of data")
paser.add_argument("-save_path", dest="save_path", default="./checkpoint/", help="Path of checkpoint")
paser.add_argument("-log_path", dest="log_path", default="./logs/", help="Path of log")
paser.add_argument("-seed", dest="seed", default=3407, help="Set Random Seed")
paser.add_argument("-gpu", dest="gpu", default=0, help="Device of gpu")

# model setting
paser.add_argument("-lr", dest="lr", default=1e-3, help="learning rate for models")
paser.add_argument("-num_epoch", dest="num_epoch", default=3, help="Num of epoch")
paser.add_argument("-batch_size", dest="batch_size", default=8, help="Batch_size of data")
paser.add_argument("-debug", dest="debug", default=False, help="debug or not")

args = paser.parse_args()