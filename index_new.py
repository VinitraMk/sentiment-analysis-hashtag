import os
from sklearn.utils import shuffle
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from datetime import date
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd
from torchtext.data.functional import to_map_style_dataset
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from azureml.core.authentication import InteractiveLoginAuthentication
import mlflow

#custom imports
from helper.preprocessor import Preprocessor
from constants.model_enums import Model
from helper.augmenter import generate_new_data
from models.rnn import RNN
from experiments.WeatherDataset import WeatherDataset
from helper.utils import get_config, get_filename, get_model_params, get_preproc_params, init_weights, read_json, save_fig, save_model, save_tensor, get_target_cols, get_azure_config

class Index:

    model_args = None
    preproc_args = None
    config = None
    preprocessor = None
    train_dataset = None
    test_dataset = None
    model_details = dict()
    no_of_batches = 0
    split_train_dataset = None
    current_model_obj = None
    model_state = None
    blob_service_client = None

    def __init__(self):
        if not(os.getenv('ROOT_DIR')):
            os.environ['ROOT_DIR'] = os.getcwd()

    def start_program(self):
        self.__define_args()
        self.__preprocess_data()
        self.__prepare_datasets()
        self.__get_word2vec_embeddings()
        self.__make_model()
        self.__make_azure_resources()
        print("############# Starting training with training data #############################\n\n")
        self.__start_training(self.test_loader)
        
    def __define_args(self):
        print('\nDefining configs')
        self.model_args = get_model_params()
        self.config = get_config()
        self.azure_config = get_azure_config()
        self.preproc_args = get_preproc_params()
        STORAGE_ACCOUNT_URL = 'https://mlintro1651836008.blob.core.windows.net/'
        self.blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=os.environ["AZURE_STORAGE_CONNECTIONKEY"])

    def __preprocess_data(self):
        print('Setting up preprocessor')
        self.preprocessor = Preprocessor()
        self.train_dataset = WeatherDataset(f'{self.config["input_path"]}\\train.csv')
        self.test_dataset = WeatherDataset(f'{self.config["input_path"]}\\test.csv', False)
        self.preprocessor.start_preprocessing(self.train_dataset)
    
    def __prepare_datasets(self):
        print('Preparing dataset')
        num_train = int(len(self.train_dataset) * self.preproc_args["train_validation_split"])
        num_valid = len(self.train_dataset) - num_train
        self.no_of_batches = int(num_train / self.model_args["batch_size"]) + 1
        self.valid_batches_count = int(num_valid / self.model_args["batch_size"]) + 1
        split_train_, split_valid_ = random_split(self.train_dataset, [num_train, num_valid])
        self.split_train_dataset = split_train_
        self.train_loader = DataLoader(split_train_, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        self.valid_loader = DataLoader(split_valid_, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.model_args["batch_size"], shuffle=False, collate_fn=self.preprocessor.collate_batch)

    def __get_word2vec_embeddings(self):
        io_path = self.config["processed_io_path"]
        embedding_model_path = f'{io_path}\\models\\word_embeddings.model'
        if not(os.path.exists(embedding_model_path)):
            print('Getting word embeddings')
            print('\tGetting tokens')
            train_tokens = self.preprocessor.get_tokens_for_dataset(self.train_dataset)
            print('\tInitializing Word2Vec')
            self.embedding_model = Word2Vec(sentences = train_tokens, vector_size=self.model_args["embed_dim"], window=self.model_args["window_size"], min_count = 1)
            print('\tTraining Word2Vec')
            self.embedding_model.train(train_tokens, total_examples=len(train_tokens), epochs=self.model_args["word2vec_epochs"])
            print('\tSaving Word2Vec embeddings')
            self.embedding_model.save(embedding_model_path)

    def __make_model(self):
        print('Making model')
        print('\tInitializing with weights from word embeddings')
        self.weight_matrix = self.preprocessor.build_vocab_weights()
        save_tensor(self.weight_matrix, 'word_embeddings_weights')
        print('\tInitializing model')
        if (self.model_args["model"] == Model.RNN):
            self.model = RNN(self.model_args["lstm_size"], self.model_args["embed_dim"], self.model_args["num_layers"], self.preprocessor.get_vocab_size(), self.model_args['num_classes'], self.model_args["dropout"])
        self.model.apply(init_weights)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args['lr'], momentum = self.model_args["momentum"])
        model_obj = {
            'model_state': self.model.state_dict(),
            'criterion': criterion,
            'optimizer_state': optimizer.state_dict()
        }
        #print('model state', self.model.state_dict())
        if self.model_args["model"] == Model.RNN:
            prev_state_h, prev_state_c = self.model.init_state(self.model_args["text_max_length"])
            model_obj["rnn_prev_state_h"], model_obj["rnn_prev_state_c"] = prev_state_h, prev_state_c
            self.model_state = (prev_state_h, prev_state_c)
        model_path = f"{self.config['processed_io_path']}\\models"
        print('\tSaving model')
        save_model(model_obj, model_path, self.model_args["model"], True)
    
    def __make_azure_resources(self):
        print('\nConfiguring Azure Resources')
        # Configuring workspace
        print('\tConfiguring Workspace')
        today = date.today()
        todaystring = today.strftime("%d-%m-%Y")
        interactive_auth = InteractiveLoginAuthentication(tenant_id=self.azure_config["tenant_id"])
        self.azws = Workspace.from_config(self.config["resource_config_path"], auth=interactive_auth)
        mlflow.set_tracking_uri(self.azws.get_mlflow_tracking_uri())
        
        print('\tConfiguring Environment')
        self.azenv = Environment.get(workspace=self.azws, name="vinazureml-env")
        self.azexp = AzExperiment(workspace=self.azws, name=f'{todaystring}-experiments')
        print('\tGetting default blob store\n')
        self.def_blob_store = self.azws.get_default_datastore()

    def __copy_model_chkpoint(self):
        local_model_path = f"{self.config['internal_output_path']}/{self.model_args['model']}_model.pt"
        upload_model_path = f"{self.config['processed_io_path']}/models/{self.model_args['model']}_model.pt"
        if (os.path.isfile(local_model_path)):
            os.system(f"copy {local_model_path} {upload_model_path}")

    def __merge_output_with_training(self):
        new_train_dataset = torch.utils.data.ConcatDataset([self.split_train_dataset, self.new_dataset])
        self.train_loader = DataLoader(new_train_dataset, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        
    def __train_model_in_azure(self, is_first_batch = False, is_last_batch = False):
        print('\tBuilding config for experiment run')
        input_data = Dataset.File.from_files(path=(self.def_blob_store, '/input'))
        input_data = input_data.as_named_input('input').as_mount()
        output = OutputFileDatasetConfig(destination=(self.def_blob_store, '/output'))
        model_args_string = json.dumps(self.model_args)
        config = ScriptRunConfig(
            source_directory='./models/training_scripts',
            script='train_nn_new.py',
            arguments=[input_data, output, self.model_args["model"], self.preprocessor.get_vocab_size(), model_args_string,
            self.no_of_batches, self.valid_batches_count],
            compute_target="mikasa",
            environment=self.azenv
        )

        run = self.azexp.submit(config)
        run.wait_for_completion(show_output = False, raise_on_error = True)

    def __upload_batch_data(self):
        self.__copy_model_chkpoint()
        print('\t\tUploading data to blob storage')
        self.def_blob_store.upload(src_dir='./processed_io', target_path="input/", overwrite=True, show_progress = False)

    def __download_output(self):
        config = get_config()
        MAIN_OUTPUT_CONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
        MODEL_CONTAINER = f'{MAIN_OUTPUT_CONTAINER}/models'
        #MODEL_DETAILS_CONTAINER = f'{MAIN_OUTPUT_CONTAINER}/internal_output'
        LOCAL_MODELDET_PATH = f"{config['experimental_output_path']}\\{self.model_args['model']}_model_details.json"
        LOCAL_MODEL_PATH = f"{config['internal_output_path']}\\{self.model_args['model']}_model.pt"
        blob_client_model = self.blob_service_client.get_blob_client(MODEL_CONTAINER, f'{self.model_args["model"]}_model.pt', snapshot=None)
        blob_client_modeldet = self.blob_service_client.get_blob_client(MODEL_CONTAINER, f'{self.model_args["model"]}_model_details.json', snapshot=None)
        self.__download_blob(LOCAL_MODEL_PATH, blob_client_model)
        self.__download_blob(LOCAL_MODELDET_PATH, blob_client_modeldet)
        blob_client_modeldet.delete_blob()
        with open(LOCAL_MODELDET_PATH) as fp:
            self.model_details = json.load(fp)


    def __download_blob(self, local_filename, blob_client_instance):
        with open(local_filename, "wb") as my_blob:
            blob_data = blob_client_instance.download_blob()
            blob_data.readinto(my_blob)

    def __remove_old_inputs(self):
        print('\nDeleting old inputs')
        ROOT_CONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b'
        FOLDER_CONTAINER = f"{ROOT_CONTAINER}/input/input"
        LOCAL_PATH = self.config["internal_output_path"]
        blob_container_client = self.blob_service_client.get_container_client(ROOT_CONTAINER)
        all_blobs = blob_container_client.list_blobs()
        for b in all_blobs:
            if "tensor" in b.name:
                blob_name = b.name.split('/')[2]
                blob_client_model = self.blob_service_client.get_blob_client(FOLDER_CONTAINER, blob_name, snapshot=None)
                print(f'\tDeleting tensor {blob_name}')
                blob_client_model.delete_blob()
                local_blob_path = f"{self.config['internal_output_path']}\\{blob_name}"
                if os.path.exists(local_blob_path):
                    os.remove(local_blob_path)

    def __plot_loss_accuracy(self, filename):
        #plot for loss and accuracy
        print("\nPlotting loss and accuracy for all epochs")
        x = np.arange(1, self.model_args["num_epochs"]+1)
        y1, y2 = [], []
        avg_acc, avg_loss = 0, 0
        for i in range(1, self.model_args["num_epochs"]+1):
            y1.append(self.model_details[f"epoch_{i}"]["loss"])
            y2.append(self.model_details[f"epoch_{i}"]["accuracy"] * 100)
            avg_acc+=self.model_details[f"epoch_{i}"]["accuracy"]
            avg_loss+=self.model_details[f"epoch_{i}"]["loss"]
        avg_acc/=self.model_args["num_epochs"]
        avg_acc*=100
        avg_loss/=self.model_args["num_epochs"]
        print("\tAverage accuracy:", avg_acc)
        print("\tAverage loss:", avg_loss, "\n")
        model_logs = {
            'average_accuracy': avg_acc,
            'average_loss': avg_loss,
            'model_filename': filename,
            'model_params': json.dumps(self.model_args),
            'preprocessing_params': json.dumps(self.preproc_args)
        }
        exp_out_path = self.config['experimental_output_path']
        log_output_path = f"{exp_out_path}\\{filename}.json"
        with open(log_output_path, 'w+') as f:
            json.dump(model_logs, f)

        plt.plot(x, y1, color="red", marker='o', linewidth=3, markersize=5)
        plt.plot(x, y2, color="green", marker="*", linewidth=3, markersize=5)
        save_fig("loss_accuracy_plot", plt)
        plt.clf()

    def __evaluate_model(self):
        print('\nGetting test data predictions')
        internal_path = self.config['internal_output_path']
        model_name = self.model_args['model']
        out_path = self.config['output_path']
        model_state_path = f"{internal_path}\\{model_name}_model.pt"
        model_object = torch.load(model_state_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(model_object['model_state'])
        self.model.eval()
        target_cols = get_target_cols()
        results_df = pd.DataFrame([], columns=['id'] + target_cols)
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if self.model_args["model"] == Model.RNN:
                    prev_state= (model_object["rnn_prev_state_h"], model_object["rnn_prev_state_c"])
                    predicted_probs, _ = self.model(batch[1])
                predicted_labels = predicted_probs[-1].type(torch.float)
                actual_labels = batch[0].type(torch.float)
                predicted_df = pd.DataFrame(predicted_labels.numpy(), columns=target_cols)
                predicted_ids = pd.DataFrame(batch[2].numpy(), columns=['id'])
                predicted_df = pd.concat([predicted_ids, predicted_df], axis = 1)
                results_df = results_df.append(predicted_df, ignore_index = True)
            filename = get_filename(model_name)
            csv_output_path = f"{out_path}\\{filename}.csv"
            results_df.to_csv(csv_output_path, index = False)
            self.__plot_loss_accuracy(filename)
        
    def __start_training(self, test_loader, is_pseudo_test = False):
        print("Vocabulary size: ", self.preprocessor.get_vocab_size(), '\n')
        #self.model_details["current_epoch"] = epoch
        #self.model_details[f"epoch_{epoch}"] = dict()
        if self.model_args['refresh_data']:
            self.__remove_old_inputs()
            print('\nBuilding all input tensors for training')
            for i, batch  in enumerate(self.train_loader):
                print(f'\tBuilding batch {i} for training')
                #out, _ = self.model(batch[1], self.model_state)
                #print('predicted probs', out)
                #print('predicted probs size', out.size())
                #print('actual output', batch[0])
                #print('actual output size', batch[0].size())
                #exit()
                save_tensor(batch[1], f'train_batch_{i}_texts')
                save_tensor(batch[0], f'train_batch_{i}_labels')
                save_tensor(batch[3], f'train_batch_{i}_offsets')
            for i, batch in enumerate(self.valid_loader):
                print(f'\tBuilding batch {i} for validation')
                save_tensor(batch[1], f'valid_batch_{i}_texts')
                save_tensor(batch[0], f'valid_batch_{i}_labels')
                save_tensor(batch[3], f'valid_batch_{i}_offsets')
            self.__upload_batch_data()
        print('\nTraining model')
        self.__train_model_in_azure()
        print('\nDownloading output')
        self.__download_output()
        self.__evaluate_model()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    index = Index()
    index.start_program()