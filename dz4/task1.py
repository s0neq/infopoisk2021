import json
import numpy as np
from bert_vec import bert_vectorizer, cls_pooling
from fasttext_vec import fasttextvec
from preproc import preproc
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import gensim


def query_indexing(my_query: str, my_vectorizer=None, bert=False, fasttext=False):
    """
    функция индексации запроса, на выходе которой посчитанный вектор запроса
    запрос может состоять из неск слов, может только из пунктуации и чисел
    """
    if not my_query:
        return False, "empty query, nothing found"

    if bert:
        # load locally saved tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained("./sberbank-ai/sbert_large_nlu_ru/saved_tokenizer/")
        model = AutoModel.from_pretrained("./sberbank-ai/sbert_large_nlu_ru/saved_model/")
        embed = bert_vectorizer([my_query], tokenizer, model, '')
        return True, np.array(embed)

    # предобработка
    prepr_query = preproc([my_query])
    if not prepr_query[0]:
        return False, "empty query after preprocessing, nothing found"

    if fasttext:
        # load locally saved model
        model = gensim.models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
        embed = fasttextvec(prepr_query, model, '')
        return True, embed

    # векторизация count or tfidf
    vec = my_vectorizer.transform(prepr_query).toarray()
    return True, vec



def similarity(my_matrix, my_vector):
    """
    функция с реализацией подсчета близости запроса и документов корпуса, 
    на выходе которой вектор, i-й элемент которого 
    обозначает близость запроса с i-м документом корпуса
    """
    return cosine_similarity(my_matrix, my_vector).T # returns a row


def lets_sort(sim, docs_names):
    """
    сортирует и печатает названия документов по убыванию релевантности
    """

    print("поиск по текстам ответов...")    
    sorted_doc_ind = np.argsort(-1 * sim)
    print("Отсортированные по убыванию соответствующие названия вопросов: ")
    res = np.array(docs_names)[sorted_doc_ind] 
    
    pprint(list(np.squeeze(res))[:5])


def lets_test():
    """
    получение запроса
    """

    my_q = "грудь маленькая, лицо не идеал. это нормально?"
    a = input("введите запрос\nлибо нажмите enter, чтобы использовать запрос по умолчанию:\n")
    if a:
        my_q = a
    print("query: ", my_q)
    return my_q


def bert_or_fasttext():
    """
    какой векторайзер использовать?
    """
    
    while True:
        try:
            a = int(input("чтобы выбрать sbert введите 0\nчтобы выбрать fasttext введите 1:"))
            if a not in set([1, 0]): raise ValueError
            return bool(a), not a
        except ValueError:
            print('так не работает, нужно ввести цифру 0 или 1\n')


def main():
    
    my_query = lets_test()
    fasttext, bert = bert_or_fasttext()

    ans_matrix = "bert_matrix.pt"
    if fasttext:
        ans_matrix = "fasttext_matrix.pt"
    
    corpus_index = torch.load(ans_matrix)
    
    # препроцессинг(кроме берта) и векторизация запроса
    is_ok, my_vector = query_indexing(my_query, 
                                      bert=bert, 
                                      fasttext=fasttext)
    
    if is_ok: # query wasnt empty
        # вектор близости: запрос и каждый документ из корпуса
        sim = similarity(corpus_index, my_vector)
        
        with open("doc_names.json", "r", encoding="utf8") as f:
            doc_names = json.load(f)
        
        lets_sort(sim, doc_names)
    else:
        print(my_vector)


if __name__ == '__main__':
    main()
