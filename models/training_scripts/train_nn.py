import sys
from pandas.core.indexing import is_label_like
import torch.nn as nn
from torch.nn import Conv1d, ReLU, MaxPool1d, Sigmoid, EmbeddingBag, Linear, Dropout, Embedding
import math
import torch
import torch
import os
import json

class RNN(nn.Module):
    embedding_dim = None
    num_layers = None
    dropout = None
    lstm_size = None

    def __init__(self, lstm_size, embedding_dim, num_layers, num_words, num_classes, dropout):
        super(RNN, self).__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.lstm_size, num_layers = self.num_layers, dropout = dropout, batch_first = True)
        self.fc = nn.Linear(self.lstm_size, num_classes)
        #self.fc1 = nn.Linear(num_words, num_classes)

    def forward(self, x, prev_state):
        embedded = self.embedding(x)
        output, state = self.lstm(embedded, prev_state)
        logits = self.fc(output)
        #logits = self.fc1(logits)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size), torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class ExperimentTrain:

    def __init__(self, mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch, device, model_args):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.is_first_batch = is_first_batch
        self.is_last_batch = is_last_batch
        self.device = device
        self.model_args = model_args
        self.model_details = dict()
        self.prev_state_h = None
        self.prev_state_c = None

    def __load_data_and_model(self):
        data_input_path = f"{self.mounted_input_path}/input"
        text_tensor_path = f"{data_input_path}/tensor_texts.pt"
        label_tensor_path = f"{data_input_path}/tensor_labels.pt"
        offset_tensor_path = f"{data_input_path}/tensor_offsets.pt"

        self.text_tensor = torch.load(text_tensor_path, map_location=self.device)
        self.label_tensor = torch.load(label_tensor_path, map_location=self.device)
        self.offset_tensor = torch.load(offset_tensor_path, map_location=self.device)
        self.text_tensor.to(self.device)
        self.label_tensor.to(self.device)
        self.offset_tensor.to(self.device)
        
        if self.is_first_batch:
            model_input_path = f"{self.mounted_input_path}/models/{self.model_args['model']}_model.pt"
        else:
            model_input_path = f"{self.mounted_output_path}/internal_output/{self.model_args['model']}_model.pt"
            model_details_path = f"{self.mounted_output_path}/internal_output/model_details.json"
            if os.path.exists(model_details_path):
                with open(model_details_path) as f:
                    self.model_details = json.load(f)

        #self.model = CNN(self.vocab_size, self.model_args["embed_dim"], 2, model_args["out_size"], model_args["stride"], model_args["text_max_length"], model_args["final_output_size"])
        self.model = RNN(self.model_args["lstm_size"], self.model_args["embed_dim"], self.model_args["num_layers"], self.vocab_size, self.model_args['num_classes'], self.model_args["dropout"])
        model_object = torch.load(model_input_path, map_location=self.device)
        self.model.load_state_dict(model_object["model_state"])
        self.model.to(self.device)
        if self.model_args["model"] == 'rnn':
            self.prev_state_h = model_object["rnn_prev_state_h"]
            self.prev_state_c = model_object["rnn_prev_state_c"]

        self.criterion = model_object["criterion"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args["lr"], momentum = self.model_args["momentum"])
        self.optimizer.load_state_dict(model_object["optimizer_state"])

    def __train_batch(self):
        self.model.train()
        self.optimizer.zero_grad()
        if self.model_args["model"] == 'ann':
            predicted_labels = self.model(self.text_tensor, self.offset_tensor)
        elif self.model_args["model"] == 'cnn':
            predicted_labels = self.model(self.text_tensor)
        elif self.model_args["model"] == 'rnn':
            predicted_labels, (state_h, state_c) = self.model(self.text_tensor, (self.prev_state_h, self.prev_state_c))
            self.prev_state_h, self.prev_state_c = state_h.detach(), state_c.detach()
        else:
            print('\nInvalid model name...')
            print('Exiting program...')
        
        self.label_tensor = self.label_tensor.type(torch.FloatTensor).to(self.device)
        print('actual output', predicted_labels[:5])
        print('actual output squeeze', predicted_labels[:5].squeeze())
        print('expected outupt', self.label_tensor[:5])
        loss = self.criterion(predicted_labels.squeeze(), self.label_tensor)
        print('loss', loss)
        if self.is_first_batch:
            self.model_details["loss"] = loss.item()
        else:
            self.model_details["loss"] += loss.item()
        loss.backward()
        self.optimizer.step()
        model_object = {
            'model_state': self.model.state_dict(),
            'criterion': self.criterion,
            'optimizer_state': self.optimizer.state_dict()
        }

        if self.model_args["model"] == 'rnn':
            model_object["rnn_prev_state_h"], model_object["rnn_prev_state_c"] = self.prev_state_h, self.prev_state_c

        if not(self.is_last_batch):
            model_output_path = f"{self.mounted_output_path}/internal_output/{self.model_args['model']}_model.pt"
        else:
            model_output_path = f"{self.mounted_output_path}/models/{self.model_args['model']}_model.pt"

        if not(os.path.exists(f"{self.mounted_output_path}/internal_output")):
            os.mkdir(f"{self.mounted_output_path}/internal_output")

        if not(os.path.exists(f"{self.mounted_output_path}/models")):
            os.mkdir(f"{self.mounted_output_path}/models")
        print('Saving model...')
        torch.save(model_object, model_output_path)
        print('Saving model details...')
        with open(f"{self.mounted_output_path}/internal_output/model_details.json", "w+") as f:
            json.dump(self.model_details, f)

    def start_experiment(self):
        self.__load_data_and_model()
        self.__train_batch()


if __name__ == "__main__":

    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    vocab_size = int(sys.argv[4])
    is_first_batch = True if sys.argv[5] == "True" else False
    is_last_batch = True if sys.argv[6] == "True" else False
    model_args_string = sys.argv[7]

    model_args = json.loads(model_args_string)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        exit()
    exp = ExperimentTrain(mounted_input_path, mounted_output_path, model_name, vocab_size, is_first_batch, is_last_batch, device, model_args)
    exp.start_experiment()