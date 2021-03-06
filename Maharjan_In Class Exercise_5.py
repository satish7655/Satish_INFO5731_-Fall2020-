import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import numpy as np
import texthero as hero
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

url = 'https://www.azquotes.com/top_quotes.html?p=1'
Quote_Details = []


def get_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    return soup


def get_quotes(soup):
    quotes = soup.find_all('div', {'class': 'wrap-block'})
    try:
        for i in quotes:
            quotes_Info = {
                'text': i.find('a', {'class': 'title'}).text.strip(),
                'author': i.find('div', {'class': 'author'}).text.strip(),
            }
            Quote_Details.append(quotes_Info)

    except:
        pass


for page in range(1, 6):
    soup = get_url(f'https://www.azquotes.com/top_quotes.html?p={page}')
    get_quotes(soup)

    print(len(Quote_Details))
    if not soup.find('li', {'class': 'next inactive'}):
        pass
    else:
        break
# print(Quote_Details)
Quotes_DF = pd.DataFrame(Quote_Details)
print(Quotes_DF.head(500))
Quotes_DF.to_excel('The 500 quotes.xlsx', index=False)

count_noun = 0
count_verb = 0
count_adj = 0
count_adv = 0
count_conjunction = 0
count_numeral = 0
count_determiner = 0
count_foreign_word = 0


def pos(text):
    global count_noun
    global count_verb
    global count_adj
    global count_adv
    global count_conjunction
    global count_numeral
    global count_determiner
    global count_foreign_word
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    nouns = ["NN", "NNP", "NNPS", "NNS"]
    verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    adjs = ["JJ", "JJR", "JJS"]
    advs = ["RB", "RBR", "RBS"]
    conjunctions = ["CC"]
    cardinal_Numerals = ["CD"]
    determiners = ["DT"]
    foreign_words = ["FW"]
    for x in pos_tags:
        print(x[1])
        for noun in nouns:
            if x[1] == noun:
                count_noun = count_noun + 1
        for verb in verbs:
            if x[1] == verb:
                count_verb = count_verb + 1
        for adj in adjs:
            if x[1] == adj:
                count_adj = count_adj + 1
        for adv in advs:
            if x[1] == adv:
                count_adv = count_adv + 1
        for conjunction in conjunctions:
            if x[1] == conjunction:
                count_conjunction = count_conjunction + 1
        for cardinal_Numeral in cardinal_Numerals:
            if x[1] == cardinal_Numeral:
                count_numeral = count_numeral + 1
        for foreign_word in foreign_words:
            if x[1] == foreign_word:
                count_foreign_word = count_foreign_word + 1


# Term Frequency — Inverse Document Frequency   tf(t,d) = count of t in d / number of words in d

# Term Frequency
Term_Frequency = (Quotes_DF['text'][1:500]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
Term_Frequency.columns = ['text', 'tf1']

# Inverse Document Frequency
for i, word in enumerate(Term_Frequency['text']):
    Term_Frequency.loc[i, 'idf'] = np.log(Quotes_DF.shape[0] / (len(Quotes_DF[Quotes_DF['text'].str.contains(word)])))

# Term Frequency – Inverse Document Frequency (TF-IDF)
Term_Frequency['tfidf'] = Term_Frequency['tf1'] * Term_Frequency['idf']
Term_Frequency.to_excel('The_500_quotes_cleaned_Version.xlsx', index=False)

for column in Quotes_DF.columns:
    Quotes_DF['Count'] = Quotes_DF['text'].apply(pos)

print("**Results: Counting the POS tagging**\n")
print("Count of nouns:")
print(count_noun, "\n")
print("Count of verbs:")
print(count_verb, "\n")
print("Count of Adjectives:")
print(count_adj, "\n")
print("Count of Adverbs:")
print(count_adv, "\n")
print("Count of conjunctions:")
print(count_conjunction, "\n")
print("Count of numerals:")
print(count_numeral, "\n")
print("Count of foreign_words:")
print(count_foreign_word, "\n")

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(" My name is Satish and this class is hard")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
