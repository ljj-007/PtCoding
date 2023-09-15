# PtCoding

## Introduction
A framework for quick start to write codes for **P**y**t**orch **Coding**.

The goal of this project is to simplify the coding for deep learning of pytorch, and let developers more focus on data processing and model designing.

Meanwhile, this is a canonical deep learning code framework for Pytorch.

version 1.0.

## Folder Tree

```shell
|-- PtCoding
    |-- main.py
    |-- README.md
    |-- data
    |-- logs
    |-- src
        |-- parse_args.py
        |-- train.py
        |-- test.py
        |-- utils.py
        |-- data_load
            |-- data_loader.py
            |-- MyData.py
        |-- model
            |-- model_process.py
            |-- MyModel

```

## Quick Start for Coding

- Rewrite **./src/data_load/MyData.py** to define your own data.

- Rewrite **./src/data_load/data_loader.py** to load your data into class [Dataset].

- Rerite **./src/model/MyModel.py** to define your own model.

- Rewrite **./src/model/model_process.py** to define the process of train, valid, and test in an epoch.

- Rewrite **./train.py** and **./test.py** to define the  train process (train and valid) and test process (test) in an epoch.

- Rewrite **./src/parse_args.py** to set the parameters.

- Rewrite **./main.py** to define the whole process.

## Run

```shell
python main.py
```

