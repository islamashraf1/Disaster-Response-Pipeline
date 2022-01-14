import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, f1_score ,precision_score , accuracy_score,recall_score 
import pickle

def load_data(database_filepath):
    
    '''
    this function to extract data from SQL Database file to categories (input and response)
    input: SQL DataBase
    output: input and response arrays with categories names
    '''
    # load data from database
    conn = sqlite3.connect(database_filepath)
    #engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql('SELECT * FROM msgs_cat ', con = conn)

    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = np.array(df.iloc[:,4:].columns)
    return X,Y,category_names


def tokenize(text):
    
    '''
    this function to tokenize Text and clean it
    input: Text sentenses
    output: cleaned Text array
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
   
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok,pos='v').lower().strip()
       # if clean_tok not in stop_words:
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):


    def tokenizee(self,text):
        '''
        this function to tokenize Text and clean it
            input: Text sentenses
            output: cleaned Text array

        '''    
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        try:
            for url in detected_urls:
                text = text.replace(url, "urlplaceholder")
            text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
            tokens = word_tokenize(text)
        except:
            print('1failed\n\n')
        lemmatizer = WordNetLemmatizer()
        try:
            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)
        except:
            print('2failed\n\n')
        return clean_tokens
    
    def starting_verb(self, text):
        '''
        this function to check if 1st word is verb or Retweet
            input: Text sentenses
            output: yes or no based on 1st word

        '''           
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            
            
            try:
                s=self.tokenizee(sentence)
                if len(s)>0:
                    pos_tags = nltk.pos_tag(s)
                    #print(pos_tags)
                    first_word, first_tag = pos_tags[0]
                    if first_tag in ['VB', 'VBP'] or first_word == 'rt':
                        return True
                    else:
                        return False
                else:
                    return False
            except: 
                return False
                

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb).fillna(False).values
        
        return np.array(X_tagged).reshape(-1,1).astype(float)


def build_model():
    '''
    pipelines models and search for best parameters withen selected pipeline and classifier attributes 
    '''
    
    pipeline2 = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('starting_verb', StartingVerbExtractor())
    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__criterion': ['gini','entropy'],
              'clf__estimator__n_estimators': [10, 20, 40],
              'features__text_pipeline__vect__ngram_range' : ((1,1),(1,2)),
             'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1})}


    cv = GridSearchCV(pipeline2, param_grid=parameters, scoring='f1_micro',cv=3,verbose=10)
  
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model performance 
    input: model and Xtest data samples and esponse for test data samples and category names
    output: results of each category of output based on many parameters (F1_score, Precision, Recall)
    '''
    
    y_pred = model.predict(X_test)
    result = []
    for i in range(len(category_names)):
        f1score = f1_score(Y_test[:,i],y_pred[:,i])
        precision = precision_score(Y_test[:,i],y_pred[:,i])
        recall = recall_score(Y_test[:,i],y_pred[:,i])
        col_name = category_names[i]
        result.append([col_name,f1score,precision,recall])
    results = pd.DataFrame(result,columns=['col_name','f1score','precision','recall'])
    print(results)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()