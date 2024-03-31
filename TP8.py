import math
import spacy
import en_core_web_sm
import json
import numpy as np

_nlp = spacy.load("en_core_web_sm")

def json_to_dic(file):
    with open(file) as f:
        data = json.load(f)
    return {d['Title']: d['Plot'] for d in data}

dic = json_to_dic('./corpus/films.json')

# Fonction test
# def dic_to_json(dic, file):
#     with open(file, 'w') as f:
#         json.dump(dic, f)

# dic_to_json(dic_nlp, './corpus/films_nlp.json')

""" Préparation du data set """

def nlp(text):
    # tokenization
    doc = _nlp(text)
    
    # Lemmatisation, normalisation, suppression des mots vides et des ponctuations
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    
    return tokens

def nlp_dico(dic):
    return {title: nlp(plot) for title, plot in dic.items()}

dic_nlp = nlp_dico(dic)


""" Calcul de la similarité de Jaccard """


def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def jaccard_similarity_from_title(title1, title2, dic):
    tokens1 = dic[title1]
    tokens2 = dic[title2]
    score = jaccard_similarity(tokens1, tokens2)
    # print(f"Le score de simiarité de Jaccard des films `{title1}` et `{title2}` est {score}")
    return score


# print(dic_nlp['The Godfather'])
# print(dic_nlp['The Godfather: Part II'])
# jaccard_similarity_from_title('The Godfather', 'The Godfather: Part II', dic_nlp)

"""Calcul de la représentation vectorielle"""


def build_vocabulary(dic):
    return set([token for tokens in dic.values() for token in tokens])

def tf(term, tokens):
    return tokens.count(term) / len(tokens)

def idf(dic):
    voc = build_vocabulary(dic)
    N = len(dic)
    idf = {}
    for term in voc:
        df = sum([1 for tokens in dic.values() if term in tokens])
        idf[term] = math.log10(N / df)
    return idf


# print(tf('crime', dic_nlp['The Godfather']))
# print(idf(dic_nlp)['crime'])


def TFIDF(dic):
    voc = build_vocabulary(dic)
    idf_values = idf(dic)
    tfidf = {}
    for title, tokens in dic.items():
        tfidf[title] = [tf(term, tokens) * idf_values[term] for term in voc]
    return tfidf


"""Recommendation"""

def cosine_similarity(list1, list2):
    dot = np.dot(list1, list2)
    norm1 = np.linalg.norm(list1)
    norm2 = np.linalg.norm(list2)
    cos = dot / (norm1 * norm2)
    return(cos)

def recommend_jaccard(title, dic):
    scores = [(t, jaccard_similarity_from_title(title, t, dic)) for t in dic.keys() if t != title]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:3]

def recommend_tfidf(title, dic):
    tfidf = TFIDF(dic)
    profile = tfidf[title]
    scores = [(t, cosine_similarity(profile, tfidf[t])) for t in dic.keys() if t != title]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:3]


film = 'The Godfather'

print(f'Les recommandations basées sur la mesure de similarité de Jaccard pour le film {film} sont :\n', recommend_jaccard(film, dic_nlp))

print()

print(f'Les recommandations basées sur la mesure de similarité de Jaccard pour le film {film} sont :\n', recommend_tfidf(film, dic_nlp))