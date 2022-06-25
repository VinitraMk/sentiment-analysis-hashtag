import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.corpus import stopwords
import string
import re
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np

#custom imports
from helper.utils import get_config, get_model_params, get_preproc_params, get_target_cols

class Preprocessor:
    preproc_args = None
    config = None
    embedded_model_wv = None

    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

    def start_preprocessing(self, data_iter):
        print('\tBuilding vocabulary')
        self.vocab = build_vocab_from_iterator(self.__yield_tokens(data_iter), specials=["<unk>"])
        print('\tSetting vocabulary index')
        self.vocab.set_default_index(1)
        print('\tBuilding text pipeline')
        self.__build_text_pipeline()
        self.label_cols = get_target_cols()

    def collate_batch(self, batch):
        label_list, text_list, id_list, offsets = [], [], [], [0]

        for sample in batch:
            label_list.append(self.label_pipeline(sample))
            id_list.append(sample["id"])
            preprocessed_text = torch.tensor(self.text_pipeline(sample["tweet"]), dtype=torch.int64)
            text_list.append(preprocessed_text)
            offsets.append(preprocessed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        id_list = torch.tensor(id_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.stack(text_list)
        
        return label_list, text_list, id_list, offsets

    def get_vocab_size(self):
        return len(self.vocab)

    def get_tokens_for_dataset(self, data_iter):
        all_tokens = []
        for i, sample in enumerate(data_iter):
            #print(self.tokenizer(self.__clean_data(data_iter[i][text])))
            all_tokens.append(self.tokenizer(self.__clean_data(data_iter[i]['tweet'])))
        return all_tokens

    def build_vocab_weights(self):
        embedded_model_path = f"{get_config()['processed_io_path']}\\models\\word_embeddings.model"
        embedded_model = Word2Vec.load(embedded_model_path)
        model_args = get_model_params()
        weight_matrix = np.zeros((len(self.vocab), model_args["embed_dim"]))
        wv = embedded_model.wv
        c = 0
        for word in self.vocab.get_itos():
            if word == "<unk>":
                continue
            wi = self.vocab.__getitem__(word)
            if wv.has_index_for(word):
                weight_matrix[wi] = wv[word]
        return torch.tensor(weight_matrix).type(torch.FloatTensor)

    def __yield_tokens(self, data_iter):
        for i, sample in enumerate(data_iter):
            #print('sample', sample)
            yield self.tokenizer(self.__clean_data(data_iter[i]['tweet']))

    def __adding_padding(self, x, max_text_len):
        if len(x) > max_text_len:
            x = x[:max_text_len]
        zeros = [0] * (max_text_len - len(x))
        x = x + zeros
        return x

    def __build_text_pipeline(self):
        max_text_len = get_model_params()['text_max_length']
        print('\t\tCleaning data')
        self.text_pipeline = lambda x: self.__adding_padding(self.vocab(self.tokenizer(self.__clean_data(x))), max_text_len)
        print('\t\tSetting label pipeline')
        self.label_pipeline = lambda x: self.__get_label_arr(x)

    def __get_label_arr(self, sample):
        all_labels = []
        for x in self.label_cols:
            all_labels.append(sample[x])
        return all_labels

    def __clean_data(self, x):
        #x = self.__apply_consistent_case(x)
        #x = self.__remove_newlinechars(x)
        x = self.__remove_extra_spaces(x)
        x = self.__remove_stop_words(x)
        x = self.__remove_punctuations(x)
        x = self.__remove_hashtags_mentions(x)
        x = self.__remove_urls(x)
        x = self.__remove_numbers(x)
        x = self.__remove_emoticons(x)
        x = self.__remove_nonenglish_alphabets(x)
        x = self.__apply_lematizer(x)
        return x

    def __apply_consistent_case(self, x):
        return str(x).lower()

    def __remove_newlinechars(self, x):
        return x.strip()

    def __remove_extra_spaces(self, x):
        return re.sub(' +', ' ', x)

    def __remove_stop_words(self, x):
        en_stopwords = set(stopwords.words('english'))
        return " ".join([word for word in str(x).split() if word not in en_stopwords])

    def __remove_punctuations(self, x):
        return x.translate(str.maketrans('', '', string.punctuation))

    def __remove_hashtags_mentions(self, x):
        #removing mentions
        x = re.sub("@([a-zA-Z0-9_]{1,50})", "", x)
        #removing hashtags
        x = re.sub("#[A-Za-z0-9_]+", "", x)
        return x

    def __remove_urls(self, x):
        return re.sub(r'http\S+', '', x)

    def __remove_numbers(self, x):
        return re.sub('([0-9]+)', '', x)

    def __remove_nonenglish_alphabets(self, x):
        is_nonenglish = lambda word: True if len(re.findall("[^A-Za-z0-9\s]+", word)) > 0 else False
        words = x.split(' ')
        new_text = " ".join([x if not(is_nonenglish(x)) else '' for x in words])
        return new_text

    def __remove_emoticons(self, x):
        emoticon_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags = re.UNICODE
        )
        return emoticon_pattern.sub(r'', x)

    def __apply_lematizer(self, x):
        lematizer = WordNetLemmatizer()
        x = "".join([lematizer.lemmatize(word) for word in x])
        return x