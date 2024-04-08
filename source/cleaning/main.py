import wandb
import pandas as pd
import os
import logging
import argparse

def go(args):
    run = wandb.init(job_type='cleaning')
    run.config.update(args)
    art = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(art)
    ids = df['price'].between(float(args.min_price),float(args.max_price))
    new_df = df[ids].copy()
    new_df['last_review'] = pd.to_datetime(new_df['last_review'])
    ids = new_df['longitude'].between(-74.25,-73.50 ) & new_df['latitude'].between(40.5,41.2)
    final_df = new_df[ids].copy()
    
    final_df.to_csv('clean_sample.csv', index=False)
    artifact = wandb.Artifact(args.output_artifact, type=args.output_type,description=args.output_description)
    artifact.add_file('clean_sample.csv')
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('parsing')
    parser.add_argument('input_artifact')
    parser.add_argument('output_artifact')
    parser.add_argument('output_type')
    parser.add_argument('output_description')
    parser.add_argument('min_price')
    parser.add_argument('max_price')
    args = parser.parse_args()
    go(args)