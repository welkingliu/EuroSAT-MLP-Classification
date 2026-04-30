import pandas as pd
import os

path = "E:\hw1\code\logs\detailed_curves"
files = os.listdir(path)

loss_upper = 0
loss_lower = 10
acc_upper = 0
acc_lower = 10
for file in files:
    df = pd.read_csv(os.path.join(path, file))
    if df["loss"].max() >= loss_upper:
        loss_upper = df["loss"].max()
    if df["loss"].min() <= loss_lower:
        loss_lower = df["loss"].min()
        
    if df["train_acc"].max() >= acc_upper:
        acc_upper = df["train_acc"].max()
    if df["train_acc"].min() <= acc_lower:
        acc_lower = df["train_acc"].min()

    if df["val_acc"].max() >= acc_upper:
        acc_upper = df["val_acc"].max()
    if df["val_acc"].min() <= acc_lower:
        acc_lower = df["val_acc"].min()

print(loss_upper, loss_lower)
print(acc_upper, acc_lower)
