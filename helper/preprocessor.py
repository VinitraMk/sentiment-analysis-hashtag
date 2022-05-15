#python library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
import string
from sklearn import preprocessing
import re
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

#custom imports
from helper.utils import get_config, get_preproc_params, get_validation_params, save_fig
from helper.vectorizer import Vectorizer
from helper.augmenter import Augmenter, generate_new_data
from constants.types.label_encoding import LabelEncoding

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    preproc_args = None
    config = None
    class_dict = dict()
    train_len = 0
    features = None
    new_data = None
    new_data_ids = None
    best_words = None

    def __init__(self):
        self.config = get_config()
        self.train = pd.read_csv(f'{self.config["input_path"]}/train.csv', encoding="ISO-8859-1")
        self.train_ids = self.train['id']
        self.train_len = self.train.shape[0]
        self.test = pd.read_csv(f'{self.config["input_path"]}/test.csv', encoding="ISO-8859-1")
        self.test_ids = self.test['id']
        self.data = pd.read_csv(f'{self.config["input_path"]}/data.csv', encoding="ISO-8859-1")
        self.preproc_args = get_preproc_params()
        self.class_dict = ['0', '2', '4']
        self.target_cols = ['s1','s2','s3','s4','s5', 'w1', 'w2', 'w3', 'w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']

    def start_preprocessing(self):
        print('\nStarting preprocessing of data...')
        self.__visualize_target_distribution()
        '''
        if self.preproc_args['encoding_type'] == LabelEncoding.LABEL_ENCODING:
            self.__apply_label_encoding()
        else:
            self.__apply_onehot_encoding()
        '''
        self.__cleanup_data()
        self.__merge_hashtags_with_text()
        self.__test_train_split()
        self.__apply_vectorization()
        self.__make_new_features()
        self.__apply_normalization()
        return self.__apply_scaling()

    def __cleanup_data(self):
        self.__remove_skip_cols()
        self.__make_hashtags_col()
        self.__apply_consistent_case()
        self.__remove_stop_words()
        self.__remove_punctuations()
        self.__remove_repeated_characters()
        self.__remove_urls()
        self.__remove_numbers()
        self.__remove_emoticons()
        self.__getting_tokens_of_text()
        if self.preproc_args['apply_stemming']:
            self.__apply_stemming()
        else:
            self.__apply_lematizer()
        #self.__plot_word_clouds()

    def __merge_hashtags_with_text(self):
        self.data['tweet_hashtag'] = self.data['tweet'] + self.data['hashtags']
        self.data['tweet'] = self.data['tweet_hashtag']
        self.data = self.data.drop(columns = ['tweet_hashtag'])
        print(self.data['tweet'])
        

    def __visualize_target_distribution(self):
        print('\tDrawing target variable visualization')
        plt.figure(figsize=(12, 10))
        plt.hist(self.train[self.target_cols])
        save_fig(f'Target_plot', plt)
        plt.clf()
    
    def __apply_consistent_case(self):
        print('\tApplying consistent case to all text')
        apply_lowercase = lambda text: str(text).lower()
        self.data['tweet'] = self.data['tweet'].apply(apply_lowercase)

    def __remove_stop_words(self):
        print('\tRemoving stop words')
        en_stopwords = set(stopwords.words('english'))
        clean_stopwords = lambda text: " ".join([word for word in str(text).split() if word not in en_stopwords])
        self.data['tweet'] = self.data['tweet'].apply(clean_stopwords)

    def __remove_punctuations(self):
        print('\tRemoving punctuations')
        clean_punctuation = lambda text: text.translate(str.maketrans('', '', string.punctuation))
        self.data['tweet'] = self.data['tweet'].apply(clean_punctuation)

    def __remove_repeated_characters(self):
        print('\tRemoving repeated characters')
        remove_repetitions = lambda text: re.sub(r'(.)1+',r'1', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_repetitions)

    def __remove_urls(self):
        print('\tRemoving urls')
        remove_urls = lambda text: re.sub('((www.[^s]+)|(https?://[^s]+)|(http?//[^s]+))', ' ', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_urls)

    def __remove_numbers(self):
        print('\tRemoving numbers')
        remove_numbers = lambda text: re.sub('([0-9]+)', '', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_numbers)

    def __remove_emoticons(self):
        print('\tRemoving emoticons')
        emoticon_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags = re.UNICODE
        )
        remove_emoticons = lambda text: emoticon_pattern.sub(r'', text)
        self.data['tweet'] = self.data['tweet'].apply(remove_emoticons)

    def __remove_skip_cols(self):
        self.data = self.data.drop(columns=self.preproc_args['skip_cols'])

    def __make_hashtags_col(self):
        make_comma_separated_str = lambda arr: '.'.join(map(str, arr)) if len(arr) != 0 else ''
        extract_hashtags = lambda text: re.findall(r'#(\w+)', text)
        self.data['hashtags'] = self.data['tweet'].apply(extract_hashtags)
        self.data['hashtags'] = self.data['hashtags'].apply(make_comma_separated_str)
        print(self.data[self.data['hashtags'] != ''].shape[0])
        print(self.data['hashtags'])

    def __getting_tokens_of_text(self):
        print('\tTokenization of tweet text')
        tokenizer = RegexpTokenizer('\w+')
        self.data['tweet'] = self.data['tweet'].apply(tokenizer.tokenize)

    def __apply_stemming(self):
        print('\tApplying Porter Stemmer')
        stemmer = PorterStemmer()
        stemming = lambda text: " ".join([stemmer.stem(word) for word in text])
        self.data['tweet'] = self.data['tweet'].apply(stemming)

    def __apply_lematizer(self):
        print('\tApplying Wordnet Lemmatizer')
        lematizer = WordNetLemmatizer()
        lematizing = lambda text: " ".join([lematizer.lemmatize(word) for word in text])
        self.data['tweet'] = self.data['tweet'].apply(lematizing)

    def __plot_word_clouds(self):
        print('\tPlotting Word Clouds')
        for col_key in self.class_dict.keys():
            print(f'\t\tPlotting word cloud for {col_key}')
            col = self.class_dict[col_key]
            plt.figure(figsize=(20, 20))
            data = self.train
            data = data[data['target'] == col_key]['tweet'][:]
            wc = WordCloud(max_words = 1000, width = 1600, height = 800, collocations = False).generate(" ".join(data))
            plt.imshow(wc)
            save_fig(f'wordcloud_{col_key}', plt)
            plt.clf()

    def __test_train_split(self):
        print('\tRestore train and test split')
        self.train = self.data[self.data['id'].isin(self.train_ids)]
        self.test = self.data[self.data['id'].isin(self.test_ids)]
        self.test = self.test.drop(columns = self.target_cols)

    def __generate_new_data(self):
        config_params = get_config()
        generate_new_data(self.train['tweet'])
        self.new_data = pd.read_csv(f'{config_params["input_path"]}\\new_data.csv')
        self.new_data_ids = self.new_data['id']
        self.data = pd.concat([self.data, self.new_data])

    def __make_new_features(self):
        self.train['num_exclamation_marks'] = self.train['tweet'].apply(lambda x: x.count('!'))
        self.train['num_question_marks'] = self.train['tweet'].apply(lambda x: x.count('?'))
        self.train['num_capitals'] = self.train['tweet'].apply(lambda x: sum([1 for c in x if c.isupper()]))
        self.test['num_exclamation_marks'] = self.test['tweet'].apply(lambda x: x.count('!'))
        self.test['num_question_marks'] = self.test['tweet'].apply(lambda x: x.count('?'))
        self.test['num_capitals'] = self.test['tweet'].apply(lambda x: sum([1 for c in x if c.isupper()]))
        
        for word in self.best_words:
            col_name = f'is_{word}_present'
            self.train[col_name] = self.train['tweet'].apply(lambda x: 1 if x.count(word) > 0 else 0)
            self.test[col_name] = self.test['tweet'].apply(lambda x: 1 if x.count(word) > 0 else 0)
        print('\t\tNo of features: ', len(list(self.train.columns)) - 3)

    def __apply_scaling(self):
        print('\tApplying scaling')
        train_df = self.train.drop(columns = ['tweet', 'id' ] + self.target_cols)
        test_df = self.test.drop(columns = ['tweet', 'id'])
        train_features = train_df.values.tolist()
        test_features = test_df.values.tolist()
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        self.__convert_vectors_dataframe(train_features, test_features, None, False)
        #print('hashtags', self.train['hashtags'])
        return self.train, self.test, self.test_ids, self.features, self.class_dict
        #return self.__convert_vectors_dataframe(train_features, test_features, None, False)

    def __apply_normalization(self):
        print('\tApplying normalization')
        train_df = self.train.drop(columns = ['tweet', 'id'] + self.target_cols)
        test_df = self.test.drop(columns = ['tweet', 'id'])
        train_features = train_df.values.tolist()
        test_features = test_df.values.tolist()
        train_features = preprocessing.normalize(train_features, norm="l2")
        test_features = preprocessing.normalize(test_features, norm="l2")
        self.__convert_vectors_dataframe(train_features, test_features, None, False)

    def __convert_vectors_dataframe(self, train_features, test_features, new_data_features = None, merge = True):
        train_df = pd.DataFrame(train_features)
        test_df = pd.DataFrame(test_features)
        train_df['id'] = self.train_ids
        test_df['id'] = self.test_ids
        if merge:
            self.train = self.train.merge(train_df, on='id')
            self.test = self.test.merge(test_df, on='id')
        else:
            train_df['id'] = self.train['id']
            train_df['tweet'] = self.train['tweet']
            train_df[self.target_cols]  = self.train[self.target_cols]
            self.train = train_df
            test_df['id'] = self.test['id']
            test_df['tweet'] = self.test['tweet']
            self.test = test_df
        

    def __apply_vectorization(self):
        print('\tApplying vectorization')
        vectorizer = Vectorizer(self.train['tweet'], self.train[self.target_cols], self.class_dict, self.test['tweet'])
        train_vectors, test_vectors, self.best_words = vectorizer.apply_vectorizer()
        self.__convert_vectors_dataframe(train_vectors, test_vectors)
        self.features = vectorizer.get_features()
        print('\t\tBest words: ', self.best_words)
        print(self.train)
        exit()
