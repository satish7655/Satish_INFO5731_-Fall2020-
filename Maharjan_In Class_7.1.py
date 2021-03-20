# Topic modeling way of extracting hidden opic from a large datset
# LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.
import re
import numpy as np
import pandas as pd
from pprint import pprint
from nltk.stem import WordNetLemmatizer

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'I', 'for', 'home', 'on', 'she', 'was', 'as', 'been', 'subject', 're', 'edu', 'use'])

# Importing the Dataset
df = pd.read_csv('Amazon_Dell_Reviews.csv')
# print(df.target_names.unique())
print("***********Dataset contains below:*************")
print(df.head(110))

# Convert to list
data = df.values.tolist()

# remove numbers from string
data = df.replace('\d+', '', regex=True, inplace=True)

# Remove new line characters
data = df.replace('\n', ' ', regex=True)

# Remove distracting single quotes
data = df.apply(lambda s: s.str.replace('"', ""))

print("***********Dataset after basic cleanup:*************")
print(data[:110])


# Tokenizing the text using simple preprocess()
def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

# print(data_words[:110])

# Create bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print("\n***********Bigram and Trigram Models***********")
print(trigram_mod[bigram_mod[data_words[0]]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Apply the Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load("en_core_web_sm")

data_lemmatized = lemmatize(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print('\n *********Displaying Lemmatized Data******')
print(data_lemmatized[:1])

# Dictionary and corpus creation required for LDA topic model
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Corpus that shows the unique id for each word
corpus = [id2word.doc2bow(text) for text in texts]

print('\n *********Creating Uniue Id for each data in the document based on word id and frequency******')
print(corpus[:110])  # (3,3) means word id 3 occured 3 times in the first document

'''[(0,
  '0.022*"ram" + 0.022*"spec" + 0.022*"mac" + 0.022*"macbook" + 0.022*"modern" '
  '+ 0.022*"nice" + 0.022*"number" + 0.022*"price" + 0.022*"pro" + '
  '0.022*"processor"')
--> 10 Keywords are produced here that contribute to the topic and weight of each topic are given accordingly'''

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
print("\n ****************Now printing the topic in LDA model************")
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

mallet_path = r'C:,Users,i24253,PycharmProjects,NLP_Learning,mallet-2.0.8,bin,mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
# Show Topics
pprint(ldamallet.show_topics(formatted=False))
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
                                                        start=2, limit=40, step=6)
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))
