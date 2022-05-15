#python imports
import json
import os
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from azure.storage.blob import BlobServiceClient, BlobClient
from datetime import date
import pandas as pd
from constants.types.validation_method import ValidationMethod

#custom imports
from helper.utils import get_config, get_filename, get_model_params, get_preproc_params, get_validation_params, save_model 
from helper.preprocessor import Preprocessor
#from models.ann import ANN
from experiments.validate import Validate
from constants.model_enums import Model
#from experiments.experiment import Experiment


def get_models_for_ensembler(model_args):
    all_models = model_args['ensembler_args'].keys()
    models = []
    for model in all_models:
        model_arg = model_args['ensembler_args'][model]
        model_clf = make_model(model_arg, True)
        models.append((model, model_clf))
    return models

def make_model(args, ensemble = False):
    if args['model'] == Model.SVM:
        svc_classifier= SVM(ensemble)
        return svc_classifier.get_model()
    elif args['model'] == Model.XGB:
        xgb_classifier = XGB(ensemble)
        return xgb_classifier.get_model()
    elif args['model'] == Model.DECISION_TREE:
        decision_tree_clf = DecisionTree(ensemble)
        return decision_tree_clf.get_model()
    elif args['model'] == Model.CTB:
        catboost_classifier = CTB(ensemble)
        return catboost_classifier.get_model()
    elif args['model'] == Model.RFA:
        rfa = RFA(ensemble)
        return rfa.get_model()
    elif args['model'] == Model.ENSEMBLER:
        models = get_models_for_ensembler(args)
        ensembler_classifer = Ensembler(models)
        return ensembler_classifer.get_model()
    elif args['model'] == Model.ANN:
        ann = ANN(ensemble)
        return ann.get_model(ann)
    else:
        print('Invalid model name :-( \n')
        exit()

def make_azure_res():
    print('\nConfiguring Azure Resources...')
    # Configuring workspace
    print('\tConfiguring Workspace...')
    today = date.today()
    todaystring = today.strftime("%d-%m-%Y")
    ws = Workspace.from_config()

    print('\tConfiguring Environment...\n')
    user_env = Environment.get(workspace=ws, name="vinazureml-env")
    user_env.docker.enabled = True
    user_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    env_vars = dict()
    env_vars['AZUREML_COMPUTE_USE_COMMON_RUNTIME'] = 'false'
    user_env.environment_variables = env_vars
    experiment = AzExperiment(workspace=ws, name=f'{todaystring}-experiments')
    
    return experiment, ws, user_env
    

def check_and_upload_input_data(azws, upload_input = False):
    #STORAGEACCOUNTURL= 'https://mlintro1651836008.blob.core.windows.net/'
    #blob_service_client = BlobServiceClient(account_url= STORAGEACCOUNTURL, credential = os.environ['AZURE_STORAGE_CONNECTIONKEY'])
    #container_client = blob_service_client.get_container_client('azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b')
    #myblobs = container_client.list_blobs(name_starts_with="input/inputs")
    #all_blobs = []
    #for s in myblobs:
        #all_blobs.append(s.name)
    def_blob_store = azws.get_default_datastore()
    if upload_input:
        def_blob_store.upload(src_dir='./processed_io', target_path='input/', overwrite=True)
    input_data = Dataset.File.from_files(path=(def_blob_store,'/input'))
    input_data = input_data.as_named_input('input').as_mount()
    output = OutputFileDatasetConfig(destination=(def_blob_store, '/output'))
    return input_data, output


def train_model_in_azure(azexp, azws, azuserenv, model_name, epoch_index, validation_k, model_args_string, is_nn = False, preproc_string = '', save_preds = False, filename = '', features = '', label_dict = '', pseudo_labeling_run = -1, upload_input = False):
    input_data, output = check_and_upload_input_data(azws, upload_input)
    print(input_data)
    if is_nn:
        config = ScriptRunConfig(
            source_directory='./models/training_scripts',
            script='train_nn.py',
            arguments=[input_data, output, model_name, save_preds, epoch_index, validation_k, filename, model_args_string, preproc_string, features, label_dict, pseudo_labeling_run],
            compute_target='mikasa',
            environment=azuserenv)
    else:
        config = ScriptRunConfig(
            source_directory='./models/training_scripts',
            script='train.py',
            arguments=[input_data, output, model_name, save_preds, epoch_index, validation_k, filename, model_args_string, preproc_string, features, label_dict, pseudo_labeling_run],
            compute_target='mikasa',
            environment=azuserenv)
    run = azexp.submit(config)
    log_path = get_config()['experimental_output_path']
    #run = azexp.start_logging(snapshot_directory=f"{log_path}/azure_experiment_logs")
    run.wait_for_completion(show_output=True, raise_on_error = True)
    aml_url = run.get_portal_url()
    print(aml_url)

def preprocess_data():
    preprocessor = Preprocessor()
    return preprocessor.start_preprocessing()

def download_blob(local_filename, blob_client_instance):
    with open(local_filename, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)

def download_output(filename, model_name, pseudo_labeling_run = -1):
    config = get_config()
    STORAGEACCOUNTURL= 'https://mlintro1651836008.blob.core.windows.net/'
    LOCALCSVPATH = f'{config["output_path"]}\\{filename}.csv' if pseudo_labeling_run == -1 else f'{config["internal_output_path"]}\\new_data_preds.csv'
    LOCALLOGPATH = f'{config["experimental_output_path"]}\\{filename}.txt'
    MAINCONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
    CSVCONTAINER = f'{MAINCONTAINER}/results'
    LOGCONTAINER = f'{MAINCONTAINER}/experiment_logs'
    CSVBLOB = f'{filename}.csv' if pseudo_labeling_run == -1 else 'new_data_preds.csv'
    LOGBLOB = f'{filename}.txt'
    EXPLOGBLOB = f'{model_name}_log.txt'
    blob_service_client = BlobServiceClient(account_url= STORAGEACCOUNTURL, credential = os.environ['AZURE_STORAGE_CONNECTIONKEY'])
    blob_client_csv = blob_service_client.get_blob_client(CSVCONTAINER, CSVBLOB, snapshot = None)
    if pseudo_labeling_run == -1:
        blob_client_log = blob_service_client.get_blob_client(LOGCONTAINER, LOGBLOB, snapshot = None)
        blob_client_explog = blob_service_client.get_blob_client(MAINCONTAINER, EXPLOGBLOB, snapshot = None)
        download_blob(LOCALLOGPATH, blob_client_log)
        blob_client_explog.delete_blob()
    download_blob(LOCALCSVPATH, blob_client_csv)
    

def test_model(model):
    config_params = get_config()
    X = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\train_X.csv')
    y = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\train_y.csv')
    test_X = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\test_X.csv')
    test_ids = pd.read_csv(f'{config_params["processed_io_path"]}\\input\\test_ids.csv')
    print(X.shape, y.shape)
    print(X.head())
    model = model.fit(X, y)
    print('Test X shape' ,test_X.shape)
    print(test_X.head())
    ypreds = model.predict(test_X)
    print(ypreds[:5])
    ypreds = pd.DataFrame(ypreds)
    preds_df = test_ids.join(ypreds)
    print(preds_df.head())

def start_validation(data, test_ids, test_X, label_dict, new_data = None, features = [], model = None, is_nn = False):
    args = get_model_params()
    validation_args = get_validation_params()
    preproc_args = get_preproc_params()
    azexp, azws, azuserenv = make_azure_res()
    print('Starting experiment...')
    validate = Validate(data, test_X, test_ids, label_dict, new_data)
    model_args_string = json.dumps(args)
    preproc_args_string = json.dumps(get_preproc_params())
    filename = get_filename(args['model'])
    if validation_args['validation_type'] == ValidationMethod.NORMAL_SPLIT:
        print('\n\n******* Validation Run ***********')
        if not(preproc_args['apply_pseudo_labeling']):
            validate.prepare_validation_dataset()
            #test_model(model)
            if is_nn:
                train_model_in_azure(azexp, azws, azuserenv, args['model'], 0, 1, model_args_string, is_nn, preproc_args_string, False, filename, '', '', -1, True)
                train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , 0, model_args_string, is_nn, preproc_args_string, True, filename, '', str(label_dict))
            else:
                train_model_in_azure(azexp, azws, azuserenv, args['model'], 0, 1, model_args_string, is_nn, preproc_args_string, False, filename, '', '', -1, True)
                train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , 0, model_args_string, is_nn, preproc_args_string, True, filename, '', str(label_dict))
            download_output(filename, args['model'], -1)
        else:
            validate.prepare_validation_data_in_runs(0)
            #test_model(model)
            train_model_in_azure(azexp, azws, azuserenv, args['model'], -1, 0, model_args_string, is_nn, preproc_args_string, False, filename, '', '', 0, True)
            download_output(filename, args['model'], 1)
            validate.prepare_validation_data_in_runs(1)
            train_model_in_azure(azexp, azws, azuserenv, args['model'], 0, 1, model_args_string, is_nn, preproc_args_string, False, filename, '', '', 0, True)
            train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , 0, model_args_string, is_nn, preproc_args_string, True, filename, '', str(label_dict))
            download_output(filename, args['model'])
    elif validation_args['validation_type'] == ValidationMethod.K_FOLD:
        for i in range(validation_args['k']):
            print('\n*************** Run', i,'****************')
            validate.prepare_validation_dataset()
            train_model_in_azure(azexp, azws, azuserenv, args['model'], i , validation_args['k'], model_args_string, is_nn, preproc_args_string, True, filename, '', '', -1, True)
        print('\n\n*************** Final Run ****************')
        filename = get_filename(args['model'])
        validate.prepare_full_dataset()
        train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , validation_args['k'], model_args_string, is_nn, preproc_args_string, True, filename,'', str(label_dict), -1, True)
        download_output(filename, args['model'])
    elif validation_args['validation_type'] == ValidationMethod.STRATIFIED_K_FOLD:
        train_indices, valid_indices = validate.get_stratified_kfold_indices()
        for i, e in enumerate((train_indices, valid_indices)):
            print('\n*************** Run', i,'****************')
            validate.prepare_stratified_kfold_dataset(e[0], e[1])
            train_model_in_azure(azexp, azws, azuserenv, args['model'], i , validation_args['k'], model_args_string, is_nn, preproc_args_string, True, filename, '', '', -1, True)
        print('\n\n*************** Final Run ****************')
        filename = get_filename(args['model'])
        validate.prepare_full_dataset()
        train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , validation_args['k'], model_args_string, is_nn, preproc_args_string, True, filename, str(label_dict), -1, True)
        download_output(filename, args['model'])
    else:
        raise ValueError(f'{validation_args["validation_type"]} is an invalid Validation method')

def read_args():
    args = get_model_params()
    config = get_config()
    preproc_args = get_preproc_params()
    train, test_X, test_ids, features, label_list = preprocess_data()
    '''
    model = make_model(args)
    model_path = f'{config["processed_io_path"]}/models'
    is_nn = args['model'] == Model.ANN
    save_model(model, model_path, args['model'], is_nn)
    if preproc_args['apply_pseudo_labeling']:
        start_validation(train, test_ids, test_X, label_dict, new_data, features, model, is_nn)
    else:
        start_validation(train, test_ids, test_X, label_dict, None, features, model, is_nn)
    '''

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()