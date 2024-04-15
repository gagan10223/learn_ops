import wandb
import argparse
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error,accuracy_score



def go(args):
    run = wandb.init(job_type='test_model')
    model_path = run.use_artifact(args.model).download()
    data = run.use_artifact(args.datasample).file()
    print(data)
    df = pd.read_csv(data)
    target = df.pop('price')
    model = mlflow.sklearn.load_model(model_path)
    y_pred = model.predict(df)
    mae = mean_absolute_error(target,y_pred)
    r2 = model.score(df,target)
    run.summary['r2'] = r2
    run.summary['mae'] = mae
    
    
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('datasample')
    arg = args.parse_args()
    
    go(arg)