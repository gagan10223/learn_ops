print('gagan')


from omegaconf import DictConfig
import hydra
import os
import mlflow
import wandb
import json
import tempfile

_steps = ['download','cleaning','data_check','data_split','train_random']

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']
    steps = config['main']['steps'].split(',') if config['main']['steps'] != 'all' else _steps
    print('-------------------')
    print(hydra.utils.get_original_cwd())
    print('-------------------')
    
    print(os.environ['WANDB_RUN_GROUP'])
    print(os.environ['WANDB_PROJECT'])
    
    with tempfile.TemporaryDirectory() as tmp:
        if 'download' in steps:
            _ = mlflow.run(
                f"{hydra.utils.get_original_cwd()}/source/download",
                'main',
                parameters={
                    'sample': config['etl']['sample'],
                    'artifact_name': 'track_data',
                    'artifact_type':'downloading',
                    'artifact_description':'downloading'
                }
            )
            
        if 'cleaning' in steps:
            _ = mlflow.run(
                f"{hydra.utils.get_original_cwd()}/source/cleaning",
                'main',
                parameters={
                    'input_artifact': 'track_data:latest',
                    'output_artifact': 'clean_data',
                    'output_type':'cleaned_data',
                    'output_description':'cleaned_data',
                    'min_price':config['etl']['min_price'],
                    'max_price':config['etl']['max_price']
                }
            )
        if 'data_check' in steps:
            _ = mlflow.run(
                f"{hydra.utils.get_original_cwd()}/source/data_check",
                'main',
                parameters={
                    'csv':'clean_data:latest',
                    'ref':'clean_data:reference',
                    'kl_threeshold':config['data_check']['kl_threshold'],
                    'min_price':config['etl']['min_price'],
                    'max_price':config['etl']['max_price']
                }
            )
        if 'data_split' in steps:
            _ = mlflow.run(
                f"{hydra.utils.get_original_cwd()}/source/split_data",
                'main',
                parameters={
                    'data_sample':'clean_data:latest',
                    'test_size':config['modeling']['test_size'],
                    'random_seed':config['modeling']['random_seed'],
                    'stratify_by':config['modeling']['stratify_by']
                }
            
            )
        if 'train_random' in steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            with open(rf_config,'r') as fp:
                print('-------   ----------')

                print(fp.read())
            
            _= mlflow.run(
                f"{hydra.utils.get_original_cwd()}/source/train",
                'main',
                parameters={
                    'train_art':'train:latest',
                    'val_size':config['modeling']['val_size'],
                    'random_seed':config['modeling']['random_seed'],
                    'stratify_by':config['modeling']['stratify_by'],
                    'rf_config': rf_config,
                    'max_tfidf':config['modeling']['max_tfidf_features'],
                    'output_art':'pkl'
                }
            
                
            )


    
if __name__ == '__main__':
    go()