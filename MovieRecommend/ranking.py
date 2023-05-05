from sklearn.metrics.pairwise import cosine_similarity
import operator
import nltk
from nltk.corpus import words
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import ast
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import math
from collections import defaultdict
import copy
import math

import itertools

import difflib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
filteredData = []


def preProcessing():
    data = pd.read_csv('datasets/DataWithPosters.csv')
    # data[['overview', 'genres']].head()
# data.head()
    # keywords_data = pd.read_csv('datasets/keywords.csv')
    # keywords_data.head()
    # print(keywords_data.columns)
    # data = data.drop(data[~data['id'].astype(str).str.isdigit()].index)
    # data['id'] = data['id'].astype(int)
    # print(data.columns)
    # keywords_data = keywords_data.drop(
    #     data[~data['id'].astype(str).str.isdigit()].index)
    # keywords_data['id'] = keywords_data['id'].astype(int)

    # combinedData = pd.merge(data, keywords_data, on='id', how='outer')
    # print(combinedData.columns)
    # filteredData = data[['id', 'title', 'genres', 'overview', 'keywords', 'tagline', 'popularity']]
    # filteredData.head()

    # filteredData.keywords = filteredData.keywords.fillna('[{}]')

    # keywordsList = []
    # for index, row in filteredData.keywords.iteritems():
    #     kStr = ''

    #     listofDict = ast.literal_eval(row)
    #     for dic in listofDict:
    #         if ('name' in dic.keys()):
    #             kStr = kStr+','+dic['name']
    #     kStr = kStr.strip(',')  # trim leading ;
    #     keywordsList.append(kStr)

    # tempDF = pd.DataFrame(keywordsList, columns=['keywords'])
    # filteredData.keywords = tempDF['keywords']

    # filteredData.genres = filteredData.genres.fillna('[{}]')

    # keywordsList = []
    # for index, row in filteredData.genres.iteritems():
    #     kStr = ''

    #     listofDict = ast.literal_eval(row)
    #     for dic in listofDict:
    #         if ('name' in dic.keys()):
    #             kStr = kStr+','+dic['name']
    #     kStr = kStr.strip(',')
    #     keywordsList.append(kStr)

    # tempDF = pd.DataFrame(keywordsList, columns=['genres'])
    # filteredData.genres = tempDF['genres']
    filteredData = data
    lemmatizer = WordNetLemmatizer()
    return filteredData, lemmatizer


# def remove_stopwords(text):
#     filteredData, lemmatizer = preProcessing()
#     stop_words = set(stopwords.words("english"))
#     word_tokens = word_tokenize(text)
#     filtered_text = [lemmatizer.lemmatize(
#         word.lower(), pos='v') for word in word_tokens if word not in stop_words]
#     return filtered_text


filteredData, lemmatizer = preProcessing()

# filteredData['pre-processed'] = filteredData['overview'].astype(str)
# filteredData['pre-processed'] = filteredData['pre-processed'].apply(
#     remove_stopwords)
# filteredData['genre tokens'] = filteredData['genres'].str.replace(",", " ")
# filteredData['keywords tokens'] = filteredData['keywords'].str.replace(
#     ",", " ")
# filteredData['overview tokens'] = filteredData['pre-processed'].apply(
#     lambda x: ' '.join(x))

filteredData['tokens'] = filteredData['genre_tokens'] + " " + \
    filteredData['keywords_tokens'] + " " + filteredData['overview_tokens']
filteredData['complete_string'] = filteredData['title'] + " " + \
    filteredData['overview'] + " " + " " + filteredData['tagline']

regular_punct = list(string.punctuation)


def remove_punctuation(text, punct_list):
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, ' ')

    text = ' '.join([w for w in text.split() if len(w) > 1])
    return text.strip()


a = remove_punctuation(filteredData["tokens"][0], regular_punct)
# print(a)


def convert_to_df(list_of_songs):
    set_list = list(list_of_songs)
    df = pd.DataFrame(list_of_songs, columns=['title'])
    df['Rank'] = range(1, len(list_of_songs) + 1)
    return df


# def return_matching_songs(query, search_token_song_map):
#     song_ids = set()
#     result_song_ids = set()
#     query_words = query.split(" ")
#     flag = True
#     for query_word in query_words:
#         if flag:
#             song_ids.update(search_token_song_map[query_word])
#             flag = False
#         else:
#             song_ids = song_ids.intersection(search_token_song_map[query_word])
#     song_ids = sorted(song_ids)
#     df = convert_to_df(song_ids)
#     return df


def return_matching_songs(query, search_token_song_map):
    song_ids = set()
    query_words = query.split(" ")
    flag = True
#   print(search_token_song_map)
    for query_word in query_words:
        if query_word not in search_token_song_map.keys():
            query_word_new = difflib.get_close_matches(
                query_word, search_token_song_map.keys())[0]
            search_token_song_map[query_word] = search_token_song_map[query_word_new]
            query_word = query_word_new
        if flag:
            try:
                song_ids.update(search_token_song_map[query_word])
                flag = False
            except:
                print("No such words")
        else:
            song_ids = song_ids.intersection(search_token_song_map[query_word])
    song_ids = sorted(song_ids)
    # set_list = list(song_ids)
    # df = pd.DataFrame(set_list, columns = ['song'] )
    # df['Rank'] = range(1, len(set_list) + 1)
    df = convert_to_df(song_ids)
    return df


def list_of_tokens(data):
    tokens = {}
    individual_tokens = []
    row_with_duplicate_tokens = []
    vector_magnitude = []
    words_count = []
    for row in data['tokens']:
        row_words = str(row).lower().split(' ')
        row_tokens = {}
        row_words_count = len(row_words)
        row_magnitude = 0
        list_tokens = set()
        for word in row_words:
            list_tokens.add(word)
            if word in tokens:
                tokens[word] = tokens[word]+1
            else:
                tokens[word] = 1
            if word in row_tokens:
                row_tokens[word] = row_tokens[word]+1
            else:
                row_tokens[word] = 1

        for key, val in row_tokens.items():
            row_magnitude = row_magnitude + (val*val)
            row_magnitude = math.sqrt(row_magnitude)
        vector_magnitude.append(row_magnitude)
        individual_tokens.append(list_tokens)
        words_count.append(row_words_count)
        row_with_duplicate_tokens.append(row_tokens.items())
    data['Tokens'] = individual_tokens
    data['TokensDictionary'] = row_with_duplicate_tokens
    data['Magnitude'] = vector_magnitude
    data['WordCount'] = words_count
    tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    return tokens


def build_index_token_map(data):
    index_token_map = {}
    for index, row in data.iterrows():
        for token in row['Tokens']:
            if token not in index_token_map:
                index_token_map[token] = set()
            index_token_map[token].add(row['title'])

    return index_token_map


def convert_to_df(list_of_songs):
    set_list = list(list_of_songs)
    df = pd.DataFrame(list_of_songs, columns=['title'])
    df['Rank'] = range(1, len(list_of_songs) + 1)
    return df


def song_id_token_map(data):
    song_token_map = defaultdict()
    for index, row in data.iterrows():
        res = defaultdict()
        for (k, v) in row['TokensDictionary']:
            res[k] = v
        song_token_map[row['title']] = res

    return song_token_map


def compute_TFIDF(tf, df, N):
    if tf == 0:
        return 0
    ans = 1 + math.log((tf), 10)
    ans = ans*(math.log((N/df), 10))
    return ans


def calc_all_docs_magnitude_basis_on_tfidf(data, boolean_search_index_map):
    map = {}
    total_N = len(data)
    tfidf_magnitude = []
    for index, row in data.iterrows():
        res = defaultdict()
        mag = 0
        for (k, v) in row['TokensDictionary']:
            res[k] = v
            mag = mag + \
                math.pow(compute_TFIDF(
                    v, len(boolean_search_index_map[k]), total_N), 2)
        mag = math.sqrt(mag)
        tfidf_magnitude.append(mag)
        map[row['title']] = mag
    data['TFIDF_MAGNITUDE'] = tfidf_magnitude
    return map


def get_query_dict(query):
    user_query_words = query.split(" ")
    user_query_dict = {}
    for word in user_query_words:
        if word not in user_query_dict:
            user_query_dict[word] = 0
        user_query_dict[word] = user_query_dict[word]+1
    return user_query_dict


def get_query_magn(query):
    user_query_words = query.split(" ")
    user_query_dict = {}
    ans = 0
    for word in user_query_words:
        if word not in user_query_dict:
            user_query_dict[word] = 0
        user_query_dict[word] = user_query_dict[word]+1
    for key, val in user_query_dict.items():
        ans = ans + val*val
    return math.sqrt(ans)


def calc_all_docs_magnitude_basis_on_idf(data, boolean_search_index_map):
    map = {}
    total_N = len(data)
    total_words = 0
    for index, row in data.iterrows():
        for (k, v) in row['TokensDictionary']:
            total_words = total_words+v
    return total_words

    # for index, row in data.iterrows():
    #     res = defaultdict()
    #     mag = 0
    #     for (k, v) in row['TokensDictionary']:
    #         res[k] = v
    #         mag = mag + v
    #     mag = mag/total_words
    #     bm25_magnitude.append(mag)
    #     map[row['title']] = mag
    # data['BM25_MAGNITUDE'] = bm25_magnitude
    # return map


def get_song_word_map(data):
    song_word_map = defaultdict()
    for index, row in data.iterrows():
        song_word_map[row['title']] = row['WordCount']
    return song_word_map


def calc_idf_bm25(query, map, N):
    ans = math.log10(((N-len(map[query])+0.5)/(len(map[query])+0.5))+1)
    return ans


def calc_bm25_score(tf, avgdl, D, query, map, N):
    k_1 = 1.2
    b = 0.75
    num = tf * (k_1+1)
    den = (tf+(k_1*(1-b+(b*(D/avgdl)))))
    ans = calc_idf_bm25(query, map, N)
    ans = ans * (num/den)
    return ans


def return_five_matching_songs_with_bm25(query, data):
    lemmatize_and_remove_background_tokens_part3 = list_of_tokens(data)
    boolean_search_index_map_part3 = build_index_token_map(data)

    avgdl = calc_all_docs_magnitude_basis_on_idf(
        data, boolean_search_index_map_part3)
    total_docs = len(data)
    avgdl = avgdl/total_docs

    song_token_map = song_id_token_map(data)

    song_ids_for_search = return_matching_songs(
        query, boolean_search_index_map_part3)
    current_query_dict = {}
    total_magnitudes_dict = calc_all_docs_magnitude_basis_on_tfidf(
        data, boolean_search_index_map_part3)
    song_word_map = get_song_word_map(data)
    query_mag = get_query_magn(query)
    for song in data['title'].tolist():
        score_bm25 = 0
        user_query_dict = get_query_dict(query)
        for query_word, freq in user_query_dict.items():
            # print(query_word)
            if query_word not in boolean_search_index_map_part3.keys():
                query_word_new = difflib.get_close_matches(
                    query_word, boolean_search_index_map_part3.keys())[0]
                boolean_search_index_map_part3[query_word] = boolean_search_index_map_part3[query_word_new]
                query_word = query_word_new
            # print(query_word)
            number_of_documents_with_query_word = len(
                boolean_search_index_map_part3[query_word])
            term_freq_in_cur_doc = 0
            if query_word in song_token_map[song]:
                term_freq_in_cur_doc = song_token_map[song][query_word]
            score_bm25 = score_bm25 + \
                calc_bm25_score(term_freq_in_cur_doc, avgdl,
                                song_word_map[song], query_word, boolean_search_index_map_part3, total_docs)
        current_query_dict[song] = score_bm25

    sorted_result_with_tfidf = sorted(
        current_query_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_result_with_tfidf


def bm25_search(user_query):
    user_query = user_query.lower()
    user_query.replace("and", "")
    lemmatize_and_remove_background_vocals_part3 = filteredData
    res_of_bm25 = return_five_matching_songs_with_bm25(
        user_query, lemmatize_and_remove_background_vocals_part3)
    res_of_bm25_df = pd.DataFrame(res_of_bm25)
    res_of_bm25_df

    res_of_bm25_df = res_of_bm25_df.rename(columns={0: 'title', 1: 'BM25'})
    res_of_bm25_df['Rank'] = range(1, len(res_of_bm25_df) + 1)
    songs_for_query = pd.merge(
        left=res_of_bm25_df, right=filteredData, left_on='title', right_on='title')
    songs_for_query = songs_for_query[[
        'Rank', 'BM25', 'title', 'tokens', 'popularity', 'complete_string', 'Tokens', 'id']]
    # print(songs_for_query.head(50))
    return songs_for_query.head(50)


def rankingList(user_query):
    word_list = words.words()

    df = bm25_search(user_query)
    list_tokens = list(df['Tokens'])
    list_tokens = list_tokens.extend(word_list)

    w2v = Word2Vec(sentences=list_tokens, min_count=1,
                   window=5, workers=1, seed=0)
    w2v.train(list_tokens, total_examples=w2v.corpus_count, epochs=20)
    w2v.save('word2vec.model')
    word_vectors = w2v.wv

    data_vectors = []
    for index, row in df.iterrows():
        vals = row['Tokens']
        vect = []
        for val in vals:
            if len(vect) == 0:
                vect = word_vectors[val]
            else:
                vect = list(map(operator.add, vect, word_vectors[val]))
        vect = [i/len(vals) for i in vect]
        data_vectors.append(vect)
    df['vector'] = data_vectors
    return df, word_vectors


def dot(K, L):
    if len(K) != len(L):
        return 0

    return sum(i[0] * i[1] for i in zip(K, L))


def searchQuery1(searchQuery):
    df, word_vectors = rankingList(searchQuery)
    query_list = searchQuery.split(" ")
    similarVal = defaultdict()
    for q in query_list:
        for index, row in df.iterrows():
            vect_row = row['vector']
            vect_q = word_vectors[q]

            similarVal[row['title']] = (similarVal.get(
                row['title'], 0) + cosine_similarity([vect_row], [vect_q])[0][0])
    similarVal = {k: v / len(query_list) for k, v in similarVal.items()}
    top_results = sorted(similarVal.items(), key=lambda x: x[1], reverse=True)
    res_of_dsm_df = pd.DataFrame(top_results)
    res_of_dsm_df = res_of_dsm_df.rename(columns={0: 'title', 1: 'DESM'})
    songs_for_query = pd.merge(res_of_dsm_df, df, on=['title'], how='inner')
    songs_for_query.drop_duplicates(
        subset=['title'], keep='first', inplace=True, ignore_index=True)
    return songs_for_query.head(25)


def finalRanking(user_query):
    res = searchQuery1(user_query)
    res['popularity'] = res['popularity'].astype(float)
    res['emsemble_score'] = 0.3*res['popularity'] + \
        0.7*res['BM25']+0.1*res['DESM']
    res.sort_values(by=['emsemble_score'], ascending=False)
