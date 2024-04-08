import wandb
import argparse
import logging
import os

def go(args):
    run = wandb.init(job_type='download_file')
    artifact = wandb.Artifact(args.artifact_name,type=args.artifact_type,description=args.artifact_description)
    artifact.add_file(os.path.join('../../',args.sample))
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('downloading it in wandb')
    parser.add_argument('sample')
    parser.add_argument('artifact_name')
    parser.add_argument('artifact_type')
    parser.add_argument('artifact_description')
    args = parser.parse_args()
    go(args)
    print(args)
    print('g1')
