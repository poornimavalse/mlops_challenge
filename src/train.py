import pandas as pd
import argparse
import glob
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.sklearn import autolog

def main(args):
    autolog()

    df = get_csvs_df(args.training_data)

    X_train, X_test, y_train, y_test = split_data(df)

    train_model(args.reg_rate,X_train,X_test,y_train,y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"cannot use path")
    csv_files=glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"no csv file")
    return pd.concat((pd.read_csv(f) for f in csv_files),sort=False)

def split_data(df,test_size=0.2):
    X=df.drop("Diabetic",axis=1)
    y=df["Diabetic"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    return X_train,X_test,y_train,y_test

def train_model(reg_rate,X_train,X_test,y_train,y_test):
    LogisticRegression(C=1/reg_rate,solver="liblinear").fit(X_train,y_train)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data",dest="training_data",type=str)
    parser.add_argument("--reg_rate",dest="reg_rate",type=float,default=0.01)
    args = parser.parse_args()
    return args

if __name__ ==  "__main__":
    print("*" * 60)
    args = parse_args()
    main(args)

