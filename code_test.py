import torch
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import Dataset
from nltk.corpus import stopwords
import string
import re
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
import os

from constants.model_enums import Model
import re
from azure.storage.blob import BlobClient, BlobServiceClient
from helper.utils import get_config

config = get_config()
azure_config = config["resource_config_path"]
STORAGE_ACCOUNT_URL = 'https://mlintro1651836008.blob.core.windows.net/'
blob_service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=os.environ["AZURE_STORAGE_CONNECTIONKEY"])
container = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/input/input'
root_container = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b'
blob = 'tensor_train_batch_0_labels.pt'
blob_client = blob_service.get_blob_client(container, blob, snapshot=None)
blob_container_client = blob_service.get_container_client(root_container)
all_blobs = blob_container_client.list_blobs()
for b in all_blobs:
    if "tensor" in b.name:
        print(b)
print(blob_client.exists())
