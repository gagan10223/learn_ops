import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split

def go(args):
    run = wandb.init(job_type='splitting')
    run.config.update(args)
    data = run.use_artifact(args.data_sample).file()
    df = pd.read_csv(data)
    train,test = train_test_split(df,test_size=float(args.test_size),random_state=int(args.random_seed),stratify=df[args.stratify_by] if args.stratify_by != 'none' else None)
    
    for df,k in zip([train,test],['train','test']):
            with tempfile.NamedTemporaryFile('w') as fp:
                df.to_csv(fp.name)
                artifact = wandb.Artifact(k,type=k)
                artifact.add_file(fp.name)
                run.log_artifact(artifact)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_sample')
    parser.add_argument('test_size')
    parser.add_argument('random_seed')
    parser.add_argument('stratify_by')
    args = parser.parse_args()
    go(args)