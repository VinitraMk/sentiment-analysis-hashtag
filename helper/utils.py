from datetime import datetime
from os import path, getenv
import yaml
import pandas as pd
import numpy as np
import joblib
import torch
import json

def get_filename(filename):
    now = datetime.now().strftime('%d%m%Y-%H%M%S')
    return f'{filename}_{now}'

def get_config_path():
    return path.join(getenv('ROOT_DIR'),'args.yaml')

def get_azure_config():
    config = get_config()
    return read_json(config["resource_config_path"])

def get_all_args():
    config_path = get_config_path()
    all_args = {}
    with open(config_path) as file:
        all_args = yaml.full_load(file)
    return all_args

def get_config():
    all_args = get_all_args()
    return all_args['config']

def read_json(path):
    with open(path) as fp:
        json_obj = json.load(fp)
        return json_obj
    return None

def get_model_params(ensemble = False, model_name = ''):
    if not(ensemble):
        all_args = get_all_args()
        return all_args['model_args']
    else:
        all_args = get_all_args()
        model_args = all_args['model_args']['ensembler_args'][model_name]
        return model_args

def get_preproc_params():
    all_args = get_all_args()
    return all_args['preproc_args']

def get_validation_params():
    all_args = get_all_args()
    return all_args['validation_args']

def break_date(date):
    if date != '':
        return (date.day, date.month, date.year)
    return (-1,-1,-1)

def extract_date(datestring):
    date = ''
    try:
        date = datetime.strptime(datestring,'%d-%m-%Y')
    except ValueError:
        try:
            date = datetime.strptime(datestring, '%Y-%m-%d')
        except ValueError:
            try:
                date = datetime.strptime(datestring,'%m-%d-%Y')
            except ValueError:
                return ''
    except TypeError:
        return ''
    return date

def is_null(value):
    return pd.isnull(value) or pd.isna(value)

def save_fig(file_name, plt):
    config = get_all_args()['config']
    plt.savefig(f'{config["visualizations_path"]}/{file_name}.png')

def save_model(model, model_path, model_name, is_nn = False):
    if not(is_nn):
        with open(f'{model_path}/{model_name}_model.sav', 'wb') as f:
                joblib.dump(model, f)
    else:
        with open(f'{model_path}/{model_name}_model.pt', 'wb') as f:
            torch.save(model, f)

def save_model_supports(model_obj, model_path, filename):
    output_path = f'{model_path}/{filename}'
    torch.save(model_obj, output_path)

def save_tensor(tensor_obj, tensor_name, filepath = ''):
    config = get_all_args()["config"]
    filename = f"{config['processed_io_path']}\\input\\tensor_{tensor_name}.pt" if filepath == '' else f"{filepath}\\tensor_{tensor_name}.pt"
    torch.save(tensor_obj, filename)

def download_model(model_path):
    return joblib.load(model_path)

def init_weights(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
    elif classname.find('EmbeddingBag') != -1 or classname.find('Embedding') != -1:
        config = get_config()
        weights = torch.load(f"{config['processed_io_path']}\\input\\tensor_word_embeddings_weights.pt")
        #model_arg = get_model_params()
        #n = model_arg["embed_dim"]
        #y = 1.0/np.sqrt(n)
        m.weight.data = weights

def download_model(model_path):
    return joblib.load(model_path)

def get_target_cols():
    model_args = get_model_params()
    return model_args['target_cols']