import re
import string
import nltk
import spacy
import numpy as np
import math
import lexnlp.extract.en.amounts
import lexnlp.extract.en.acts
import lexnlp.extract.en.entities.nltk_re
import lexnlp.extract.en.constraints
import lexnlp.extract.en.citations
import lexnlp.extract.en.conditions
import lexnlp.extract.en.copyright
import lexnlp.extract.en.courts
import lexnlp.extract.en.cusip
import lexnlp.extract.en.dates
import lexnlp.extract.en.definitions
import lexnlp.extract.en.distances
import lexnlp.extract.en.durations
import lexnlp.extract.en.money
import lexnlp.extract.en.percents
import lexnlp.extract.en.pii
import lexnlp.extract.en.ratios
import lexnlp.extract.en.regulations
import lexnlp.extract.en.trademarks
import lexnlp.extract.en.urls

from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

pd.set_option('display.max_colwidth', 200)

# load spaCy model
nlp = spacy.load("en_core_web_sm")
list_of_lists = []
with open('titles.txt') as f:
    text = f.read().replace('\n', ' ')
    print(text)

# create a spaCy object
doc = nlp(text)

# print token, dependency, POS tag
for tok in doc:
    print(tok.text, "-->", tok.dep_, "-->", tok.pos_)

# Matcher class object
matcher = Matcher(nlp.vocab)

# define the pattern
pattern = [{'DEP': 'amod', 'OP': "?"},  # adjectival modifier
           {'POS': 'NOUN'},
           {'LOWER': 'such'},
           {'LOWER': 'as'},
           {'POS': 'PROPN'}]

matcher.add("matching_1", None, pattern)
matches = matcher(doc)

span = doc[matches[0][1]:matches[0][2]]
print(span.text)

# Extracting acts
with open(' 01-05-1 Adams v Tanner.txt,') as r:
    text2 = r.read().replace('\n', ' ')
    print(text2)

print(lexnlp.extract.en.acts.get_act_list(text2))
print(list(lexnlp.extract.en.amounts.get_amounts(text2)))
print(list(lexnlp.extract.en.entities.nltk_re.get_entities.nltk_re.get_companies(text2)))
print(list(lexnlp.extract.en.constraints.get_constraints(text2)))
print(list(lexnlp.extract.en.citations.get_citations(text2)))
print(list(lexnlp.extract.en.conditions.get_conditions(text2)))
print(list(lexnlp.extract.en.copyright.get_copyright(text2)))
print(lexnlp.extract.en.cusip.get_cusip(text2))
print(list(lexnlp.extract.en.dates.get_dates(text2)))
print(list(lexnlp.extract.en.definitions.get_definitions(text2)))
print(list(lexnlp.extract.en.distances.get_distances(text2)))
print(list(lexnlp.extract.en.durations.get_durations(text2)))
print(list(lexnlp.extract.en.money.get_money(text2)))
print(list(lexnlp.extract.en.percents.get_percents(text2)))
print(list(lexnlp.extract.en.pii.get_pii(text2)))
print(list(lexnlp.extract.en.ratios.get_ratios(text2)))
print(list(lexnlp.extract.en.regulations.get_regulations(text2)))
print(list(lexnlp.extract.en.trademarks.get_trademarks(text2)))
print(list(lexnlp.extract.en.urls.get_urls(text2)))