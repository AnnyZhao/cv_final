import argparse
import pickle
import gc
import os
import sys
import numpy as np
import torch
import lib.base_model as base_model
import lib.gaussian_model as gaussian_model
from lib.training import run_train
import typing
import sys
from tee import StdoutTee
from datetime import datetime
from pytz import timezone
from tzlocal import get_localzone
import pathlib

now_utc = datetime.now(timezone('UTC'))
# Convert to local time zone
now_local = now_utc.astimezone(get_localzone())
dt_string = now_local.strftime("%Y-%m-%d %H:%M")
log_dir = f"logs/{dt_string}"
pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
args = parser.parse_args()
device = torch.device("cuda:0")


def make_base_model():
    resnet18 = base_model.ResNet18(pretrained=True)
    model = base_model.MyModel(resnet18).to(device)
    return model


def make_gaussian_model():
    resnet18 = gaussian_model.ResNet18(pretrained=True)
    model = gaussian_model.GaussianModel(resnet18).to(device)
    return model


epochs = args.epochs


with StdoutTee(f"{log_dir}/log-gaussian.txt", mode='w', buff=1):
    run_train(model_name='gaussian', model_f=make_gaussian_model, EPOCHS=epochs, log_dir=log_dir)
gc.collect()
with StdoutTee(f"{log_dir}/log-base.txt", mode='w', buff=1):
    run_train(model_name='base', model_f=make_base_model, EPOCHS=epochs, log_dir=log_dir)
