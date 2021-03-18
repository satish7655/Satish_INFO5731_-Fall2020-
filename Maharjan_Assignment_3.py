import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import nltk
import numpy as np
from scipy.spatial.distance import cosine

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


def basic_clean(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


words = basic_clean(''.join(str(Quotes_DF['text'].tolist())))
print(words[:1000])
print("\n", "The Ngrams results where frequency of all the N-grams (N=3)")
print("Words   ------------------>    Frequency")
ngram = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:1000]
print(ngram)


# Break the quote in given N Gram Value
def N_Grams(text, n):
    # split in tokens
    tokens = re.split("\\s+", text)
    ngrams = []

    # Collect n-Grams
    # ngram=X-(n-1)
    # X= total number of words in a sentence, n-Ngram
    for i in range(len(tokens) - n + 1):
        temp = [tokens[j] for j in range(i, i + n)]
        ngrams.append(" ".join(temp))
    return ngrams


# test
test = "My name is satish Maharjan"
print(N_Grams(test, 3))


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


# Total Nouns Count
def NounCount(x):
    nounCount = sum(1 for word, pos in nltk.pos_tag(nltk.word_tokenize(x)) if pos.startswith('NN'))
    return nounCount


# Calculating probability of all the bigrams in the dataset
def probBigram(quote):
    print(quote)
    biGram = N_Grams(quote, 2)
    for g in biGram:
        arr = g.split(" ")
        probability = quote.count(g) / quote.count(arr[0])
        print(g)
        print(probability)


# Panda Dataframe
Quotes_DF['noun_count'] = Quotes_DF['text'].apply(NounCount)
total_nouncount = len(Quotes_DF['text'])
Quotes_DF['Probability noun phrases'] = Quotes_DF['noun_count'] / total_nouncount
Quotes_DF['Trigrams'] = Quotes_DF['text'].map(lambda x: find_ngrams(x.split(" "), 3))
Quotes_DF['Bigram'] = Quotes_DF['text'].map(lambda x: find_ngrams(x.split(" "), 2))
Quotes_DF['Proba'] = Quotes_DF['text'].map(lambda x: probBigram(x))

# Save into a file
Quotes_DF.to_excel('The 500 quotes_Assign3.xlsx', index=False)

count_noun = 0
probablilty = 0


def pos(text):
    global count_noun
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    nouns = ["NN", "NNP", "NNPS", "NNS"]
    for x in pos_tags:
        print(x[1])
        for noun in nouns:
            if x[1] == noun:
                count_noun = count_noun + 1


# Term Frequency — Inverse Document Frequency   tf(t,d) = count of t in d / total number of words in d
# Document Frequency(IDF) = log( (total number of documents)/(number of documents with term t))
# Term Frequency — Inverse Document Frequency   tf(t,d) = count of t in d / number of words in d


Term_Frequency = (Quotes_DF['text'][1:500]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
Term_Frequency.columns = ['text', 'tf1']

# Inverse Document Frequency
for i, word in enumerate(Term_Frequency['text']):
    Term_Frequency.loc[i, 'idf'] = np.log(Quotes_DF.shape[0] / (len(Quotes_DF[Quotes_DF['text'].str.contains(word)])))

# Term Frequency – Inverse Document Frequency (TF-IDF)
Term_Frequency['tfidf'] = Term_Frequency['tf1'] * Term_Frequency['idf']
Term_Frequency['cos_vec'] = (cosine(Term_Frequency['tfidf'], Term_Frequency['tfidf']))
print(Term_Frequency['cos_vec'])

Term_Frequency.to_excel('The reviews_tfidf.xlsx', index=False)
for column in Quotes_DF.columns:
    Quotes_DF['Count'] = Quotes_DF['text'].apply(pos)

print("**Results: Counting the POS tagging**\n")
print("Count of nouns:")
print(count_noun, "\n")
