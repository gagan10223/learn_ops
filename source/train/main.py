import argparse
import wandb
import os
import logging
import shutil
import matplotlib.pyplot as plt
import mlflow
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

logging = logging.getLogger()
def feature_dates(dates):
    data = pd.DataFrame(dates).apply(pd.to_datetime)
    return data.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()
    
def go(args):
    run = wandb.init(job_type='training')
    run.config.update(args)
    with open(args.rf_config,'r') as file:
        configs = json.load(file)
    configs['random_state'] = int(args.random_seed)
    path = run.use_artifact(args.train_art).file()
    data = pd.read_csv(path)
    y = data.pop('price')
    x_train,y_train,x_test,y_test = train_test_split(data,y,random_state=int(args.random_seed),stratify=data[args.stratify_by],test_size=float(args.val_size))

    pipe,columns = inference(configs,args.max_tfidf,x_train)

    pipe.fit(x_train,x_test)
    scored = pipe.score(y_train,y_test)
    logging.info(f"score:{scored}")
    preds = pipe.predict(y_train)
    error = mean_absolute_error(preds,y_test)
    logging.info(f"error:{error}")

    if os.path.exists('random_forest_dir'):
        shutil.rmtree('random_forest_dir')
    signature = mlflow.models.infer_signature(y_train,y_test)
    print(signature)
    mlflow.sklearn.save_model(
        pipe,
        'random_forest_dir',
        signature=signature,
        input_example=x_train.iloc[:5]
    )
    #artifact = wandb.Artifact(args.output_art,type='model_export',description='trained',metadata=configs)
    #artifact.add_dir('random_forest_dir')
    #run.log_artifact(artifact)
    print('----------------------------r======================')
    print(pipe['rand'].feature_importances_)
    r2 = pipe.score(y_train,y_test)
    mae = mean_absolute_error(pipe.predict(y_train),y_test)
    run.summary['r2'] = r2
    run.summary['mae'] = mae
    run.finish()
    
def inference(config,max_feat,x_train):
    ordinal_cat = ['room_type']
    non_cat = ['neighbourhood_group']
    ordinal_encoder = OrdinalEncoder()
    non_pre = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())
    nums = ['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365','longitude','latitude']
    zero_imputer = SimpleImputer(strategy='constant',fill_value=0)

    date_imputer = make_pipeline(SimpleImputer(strategy='constant',fill_value='2010-01-01'),FunctionTransformer(feature_dates,check_inverse=False,validate=False))
    tfidf = make_pipeline(SimpleImputer(strategy='constant',fill_value=''),FunctionTransformer(np.reshape,kw_args={'newshape':-1}),TfidfVectorizer(
        binary=False,max_features=int(max_feat),stop_words='english'
    ))

    preprocessor = ColumnTransformer(
        transformers=[
            ('first',ordinal_encoder,ordinal_cat),
            ('second',non_pre,non_cat),
            ('third',zero_imputer,nums),
            ('fourth',tfidf,['name']),
            ('fifth',date_imputer,['last_review'])
        ],
        remainder='drop'
    )
    
    features = ordinal_cat + non_cat + nums + ['last_review','name']
    random_forest = RandomForestRegressor(**config)
    pipeline = Pipeline(
        steps = [
            ('pre',preprocessor),
            ('rand',random_forest)
        ]
    )
    x = preprocessor.fit_transform(x_train)
   
    return pipeline,features
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_art')
    parser.add_argument('val_size')
    parser.add_argument('random_seed')
    parser.add_argument('stratify_by')
    parser.add_argument('rf_config')
    parser.add_argument('max_tfidf')
    parser.add_argument('output_art')
    args = parser.parse_args()

    
    go(args)