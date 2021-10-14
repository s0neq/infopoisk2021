from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import torch, json
import numpy as np
import pickle
from tqdm import tqdm

def bm25_similarity(my_matrix, sec_matrix):
    """
    функция с реализацией подсчета близости запросов и документов корпуса
    """
    return my_matrix.dot(sec_matrix)


def similarity(my_matrix, ques_matrix):
    return cosine_similarity(my_matrix, ques_matrix)


def count_metric(my_array, ind, metric):
    with open("preproced_corpus.json", encoding="utf8") as f:
        ans = json.load(f)

    sorted_doc_ind = np.argsort(-1 * my_array)
    res = np.array(ans)[sorted_doc_ind] 

    if ans[ind] in res[:5]:
        metric += 1
    return metric / len(ans)


def get_metric(sim):
    metric = 0
    for i,v in tqdm(enumerate(sim)):
        metric = count_metric(v,i, metric)
    return metric


def bm25get_metric(sim):
    metric = 0
    for i,v in tqdm(enumerate(sim)):
        metric = count_metric(v.toarray(),i, metric)
    return metric

def main():
    with open("file_names.json", encoding="utf8") as f:
        file_names = json.load(f)
    final = dict()
    for key in file_names:
        print("starting with", key)

        ians_name, iq_name = file_names[key][0], file_names[key][1]

        ians = torch.load(ians_name)
        iq = torch.load(iq_name)

        if key == 'Bm25':
            sim = bm25_similarity(ians, iq)
            metr = bm25get_metric(sim)
        else:
            sim = similarity(ians, iq)
            metr = get_metric(sim)
        
        final[key] =  metr
    
    # fasttext, bert
    print("now bert")
    ians_name, iq_name = "bert_matrix.pt", "bert_matrix_questions.pt"
    ians = torch.load(ians_name)
    iq = torch.load(iq_name)
    sim = similarity(ians, iq)
    metr = get_metric(sim)
    final["bert"] =  metr
    
    print("now fasttext")
    ians_name, iq_name = "fasttext_matrix.pt", "fasttext_matrix_ques.pt"
    ians = torch.load(ians_name)
    iq = torch.load(iq_name)
    sim = similarity(ians, iq)
    metr = get_metric(sim)
    final["fasttext"] =  metr
    with open("final.json", "w", encoding="utf8") as f:
        json.dump(final, f)
    
if __name__ == '__main__':
    main()
