import pandas as pd
import requests
from bs4 import BeautifulSoup
import string


url = 'https://www.amazon.com/Dell-Inspiron-5000-5570-Laptop/product-reviews/B07N49F51N/ref' \
      '=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1 '
Review_list = []


def get_url(url):
    # using splash to render the page
    r = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
    # create a soup object
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.title.text
    print(title)
    return soup


def get_reviews(soup):
    Dell_Review = soup.find_all('div', {'data-hook': 'review'})
    # print(Dell_Review)
    try:
        for i in Dell_Review:
            review = {
                # 'Review_title': i.find('a', {'data-hook': 'review-title'}).text.strip(),
                'Description_Detail': i.find('span', {'data-hook': 'review-body'}).text.strip(),
                # 'Name_Info': i.find('span', {'class': 'a-profile-name'}).text.strip(),
            }
            Review_list.append(review)

    except:
        pass


for p in range(1, 12):
    soup = get_url(
        f'https://www.amazon.com/Dell-Inspiron-5000-5570-Laptop/product-reviews/B07N49F51N/ref=cm_cr_arp_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber={p}')
    get_reviews(soup)
    print(len(Review_list))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break
print(Review_list)
Review_DF = pd.DataFrame(Review_list)
# print(Review_DF.head(2))
# Review_DF.to_excel('The Dell_Reviews.xlsx', index=False)

# print (string.punctuation)
stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now"]


def remove_punctuation(txt):
    # use list comphrehnesive
    txt_nopunct = "".join([i for i in txt if i not in string.punctuation])
    return txt_nopunct


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


def stemSentence(sentence):
    token_words = word_tokenize(sentence)

    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def lemmatize_text(text):
    token_words = w_tokenizer.tokenize(text)

    lemma_sentence = []
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)


count_noun = 0
count_verb = 0
count_adj = 0
count_adv = 0


def pos(text):
    global count_noun
    global count_verb
    global count_adj
    global count_adv
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    nouns = ["NN", "NNP", "NNPS", "NNS"]
    verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    adjs = ["JJ", "JJR", "JJS"]
    advs = ["RB", "RBR", "RBS"]
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


for column in Review_DF.columns:
    # 1.Remove noise, such as special characters and punctuations.
    # selecting all but number and letter with regular expression and replacing special character with space
    Review_DF['With Special Character Removed'] = Review_DF['Description_Detail'].str.replace(r'\W', " ")
    Review_DF['With punctuation Removed'] = Review_DF['Description_Detail'].apply(lambda x: remove_punctuation(x))

    # stopwords
    Review_DF['With StopWords Removed'] = Review_DF['With punctuation Removed'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))
    # remove numbers
    Review_DF['With Numbers Removed'] = Review_DF['With StopWords Removed'].str.replace('\d+', '')

    # lowercase
    Review_DF[' All Lowercase'] = Review_DF['With Numbers Removed'].str.lower()

    # Stemming
    Review_DF['Stemmed'] = Review_DF[' All Lowercase'].apply(stemSentence)

    # Lemmatization
    Review_DF['Lemmie'] = Review_DF['Stemmed'].apply(lemmatize_text)

    Review_DF['Count'] = Review_DF['Stemmed'].apply(pos)

    Review_DF.head()
    Review_DF.to_excel('The Dell_Reviews_cleaned.xlsx', index=False)

print("**Results: Counting the POS tagging**\n")
print("Count of nouns:")
print(count_noun, "\n")
print("Count of verbs:")
print(count_verb, "\n")
print("Count of Adjective:")
print(count_adj, "\n")
print("Count of Adverb:")
print(count_adv, "\n")

# tokenize the review list
words = nltk.word_tokenize(Review_list)
pos_tags = nltk.pos_tag(words)
print(pos_tags)
