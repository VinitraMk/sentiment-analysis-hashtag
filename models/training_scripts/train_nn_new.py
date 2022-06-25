import sys
from pandas.core.indexing import is_label_like
import torch.nn as nn
from torch.nn import Conv1d, ReLU, MaxPool1d, Sigmoid, EmbeddingBag, Linear, Dropout, Embedding
import math
import torch
import os
import json
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

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
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.lstm_size, num_layers = self.num_layers,
        dropout=dropout, batch_first = True)
        self.fc = nn.Linear(self.lstm_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, state = self.lstm(embedded)
        logits = self.fc(state[-1])
        logits = torch.sigmoid(logits)
        logits = (logits > 0.5).type(torch.int8)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size), torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class ExperimentTrain:

    def __init__(self, mounted_input_path, mounted_output_path, model_name, vocab_size, device, model_args, no_of_batches):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.vocab_size = vocab_size
        #self.is_first_batch = is_first_batch
        #self.is_last_batch = is_last_batch
        self.device = device
        self.model_args = model_args
        self.model_details = dict()
        self.prev_state_h = None
        self.prev_state_c = None
        self.no_of_batches = no_of_batches

    def __load_data_and_model(self, batch_index):
        data_input_path = f"{self.mounted_input_path}/input"
        text_tensor_path = f"{data_input_path}/tensor_batch_{batch_index}_texts.pt"
        label_tensor_path = f"{data_input_path}/tensor_batch_{batch_index}_labels.pt"
        offset_tensor_path = f"{data_input_path}/tensor_batch_{batch_index}_offsets.pt"

        self.text_tensor = torch.load(text_tensor_path, map_location=self.device)
        self.label_tensor = torch.load(label_tensor_path, map_location=self.device)
        self.offset_tensor = torch.load(offset_tensor_path, map_location=self.device)
        self.text_tensor.to(self.device)
        self.label_tensor.to(self.device)
        self.offset_tensor.to(self.device)
        model_name = self.model_args['model']
        
        if batch_index == 0:
            model_input_path = f"{self.mounted_input_path}/models/{model_name}_model.pt"
        else:
            model_input_path = f"{self.mounted_output_path}/internal_output/{model_name}_model.pt"
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

    def __train_batch(self, batch_index):
        self.model.train()
        self.optimizer.zero_grad()
        model_name = self.model_args['model']
        if self.model_args["model"] == 'rnn':
            predicted_labels, (state_h, state_c) = self.model(self.text_tensor)
            self.prev_state_h, self.prev_state_c = state_h.detach(), state_c.detach()
        else:
            print('\nInvalid model name...')
            print('Exiting program...')
        
        self.label_tensor = self.label_tensor.to(self.device)
        print('actual output', predicted_labels[-1][:5])
        print('expected outupt', self.label_tensor[:5])
        predicted_labels = Variable(predicted_labels[-1].type(torch.float), requires_grad=True)
        actual_labels = Variable(self.label_tensor.type(torch.float), requires_grad=True)
        loss = self.criterion(predicted_labels, actual_labels)
        print('loss', loss)
        if batch_index == 0:
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

        if batch_index == (self.model_args["num_epochs"] - 1):
            model_output_path = f"{self.mounted_output_path}/internal_output/{model_name}_model.pt"
        else:
            model_output_path = f"{self.mounted_output_path}/models/{model_name}_model.pt"

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
        for i in range(self.no_of_batches):
            self.__load_data_and_model(i)
            self.__train_batch(i)


if __name__ == "__main__":

    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    vocab_size = int(sys.argv[4])
    #is_first_batch = True if sys.argv[5] == "True" else False
    #is_last_batch = True if sys.argv[6] == "True" else False
    model_args_string = sys.argv[5]
    no_of_batches = int(sys.argv[6])

    model_args = json.loads(model_args_string)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        exit()
    exp = ExperimentTrain(mounted_input_path, mounted_output_path, model_name, vocab_size, device, model_args, no_of_batches)
    exp.start_experiment()