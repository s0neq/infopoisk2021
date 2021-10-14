from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import torch, json
import numpy as np
import pickle

def inverse_index_ans(my_corpus, vectorizer, name=''):
    X = vectorizer.fit_transform(my_corpus)
    torch.save(X, name)
    return name, vectorizer
    # return X, vectorizer


def inverse_index_ques(my_corpus, vectorizer, name=''):
    X = vectorizer.transform(my_corpus)
    torch.save(X, name)
    return name
    # return X


def bm25_inverse_index(my_corpus, name=''):
    """
    функция индексации корпуса, на выходе которой посчитанная матрица Document-Term 
    в качестве векторизации документов корпуса - слагаемые BM25
    """
    print("indexing in progress...")

    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(my_corpus)
    tf = tf_vectorizer.fit_transform(my_corpus) # матрица с tf
    tfidf_vectorizer.fit_transform(my_corpus) # фит векторайзера tfidf на текстах
    idf = tfidf_vectorizer.idf_

    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)

    len_d = np.squeeze(np.asarray(len_d)) # from matrix to array 

    avdl = len_d.mean()
    values = []
    rows = []
    cols = []

    for i, j in zip(*tf.nonzero()):
        word_idf = idf[j]
        word_tf = tf[i, j]

        # numerator
        A = word_idf * word_tf * (k + 1)

        # denominator
        my_const = (k * (1 - b + b * len_d[i] / avdl))
        B = word_tf + my_const
        values.append(A/B)
        rows.append(i)
        cols.append(j)


    sparce_matrix = sparse.csr_matrix((values, (rows, cols)))
    torch.save(sparce_matrix, name)
    return name, count_vectorizer
    # return sparce_matrix, count_vectorizer


def main():
    with open("preproced_corpus.json", encoding="utf8") as f:
        ans = json.load(f)

    with open("preproced_ques.json", encoding="utf8") as f:
        q = json.load(f)
    

    res = dict()

    # 1 count
    vectorizer = CountVectorizer()
    ians_name, vectorizer = inverse_index_ans(ans, vectorizer, "count_vec_ans_matr.pt")
    pickle.dump(vectorizer, open("Count_vectorizer.pickle", "wb"))
    iq_name = inverse_index_ques(q, vectorizer, "count_vec_q_matr.pt")
    res['CountVectorizer'] = [ians_name, iq_name, "Count_vectorizer.pickle"]
    print('count done')
    # 2 Tfidf
    vectorizer = TfidfVectorizer()
    ians_name, vectorizer = inverse_index_ans(ans, vectorizer, "tfidf_vec_ans_matr.pt")
    pickle.dump(vectorizer, open("Tfidf_vectorizer.pickle", "wb"))
    iq_name = inverse_index_ques(q, vectorizer, "tfidf_vec_q_matr.pt")
    res['TfidfVectorizer'] = [ians_name, iq_name, "Tfidf_vectorizer.pickle"]
    print('tfidf done')
    # 3 bm25
    ians_name, count_vectorizer = bm25_inverse_index(ans, "bm25_vec_ans_matr.pt")
    pickle.dump(count_vectorizer, open("Bm25_vectorizer.pickle", "wb"))
    iq = inverse_index_ques(q, count_vectorizer, "bm25_vec_q_matr.pt")
    res['Bm25'] = [ians_name, iq_name, "Bm25_vectorizer.pickle"]
    print('bm25 done')
    with open("file_names.json", "w", encoding="utf8") as f:
        json.dump(res, f)
    print('names of matrices and models saved in "file_names.json"')

if __name__ == '__main__':
    main()
