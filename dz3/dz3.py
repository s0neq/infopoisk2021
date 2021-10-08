import json
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import sparse
from string import punctuation

nltk.download("stopwords")
punctuation += "...-"


def get_corpus():
    """
    достает данные из файла
    возвращает два списка - с ответами на вопросы и самими вопросами 
    """

    with open("questions_about_love.jsonl", 'r', encoding='utf-8-sig') as f:  
        contents = list(f)[:50000]

    corpus = []
    doc_names = []

    for el in contents:
        element = json.loads(el)
        ans = element["answers"]

        if not ans:
            continue

        doc_text = ans[0]["text"]
        try: # in case value is empty
            the_max = int(ans[0]["author_rating"]["value"])
        except:
            the_max = -1

        for i in range(1, len(ans)):
            try: # in case value is empty
                ins_max = int(ans[i]["author_rating"]["value"])
            except:
                continue
            if ins_max > the_max:
                the_max = ins_max
                doc_text = ans[i]["text"]

        corpus.append(doc_text.strip())
        doc_names.append(element["question"].strip())
    return corpus, doc_names


def preproc(my_corpus):
    """
    Функция препроцессинга данных
    """

    res = []

    # соединяем все тексты в один, 
    # так как Mystem работает дольше при обработке нескольких текстов, чем одного большого
    alltexts = ' '.join([txt + ' endofthetext' for txt in my_corpus])

    m = Mystem()
    rus_stopwords = stopwords.words("russian")

    print("preprocessing in progress...")

    # приведение к нижнему регистру и лемматизация
    lemmas = m.lemmatize(alltexts.lower())
    print("lemmatized...")

    doc = []
    for lem in lemmas:
        # убрать стоп слова и пунктуацию (2 слепленных знака пунктуации тоже выбросить)
        if lem not in rus_stopwords \
            and lem.strip('\n ' + punctuation) not in punctuation \
                and lem.strip() != '' \
                    and lem.strip() != '\n':
            if lem == 'endofthetext': # делим тексты обратно как было
                res.append(' '.join(doc))
                doc = []
            else:
                doc.append(lem)

    return res


def inverse_index(my_corpus):
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
    return sparce_matrix, count_vectorizer


def query_indexing(my_query: str, my_vectorizer):
    """
    функция индексации запроса, на выходе которой посчитанный вектор запроса
    запрос может состоять из неск слов, может только из пунктуации и чисел
    """

    # предобработка
    preproc_query = preproc([my_query])

    if not preproc_query[0]:
        return False, "empty query after preprocessing, nothing found"

    # векторизация
    vec = my_vectorizer.transform(preproc_query).toarray()

    return True, vec


def similarity(my_matrix, my_vector):
    """
    функция с реализацией подсчета близости запроса и документов корпуса, 
    на выходе которой вектор, i-й элемент которого 
    обозначает близость запроса с i-м документом корпуса
    """

    # my_vector переворачиваю, чтобы стал столбцом
    return my_matrix.dot(my_vector.reshape(-1, 1)).T # returns a row


def lets_sort(sim, docs_names):
    """
    сортирует и печатает названия документов по убыванию релевантности
    """
    print("searching...")
    sorted_doc_ind = np.argsort(-1 * sim)
    print("Отсортированные по убыванию названия документов: ")
    print(np.array(docs_names)[sorted_doc_ind][:25]) # вывод 25 названий


def lets_test():
    """
    получение запроса
    """

    my_q = "возможно вы достались ему слишком просто. Становитесь менее предсказуемой и более загадочной и мужчина не будет так надолго пропадать."
    a = input("введите запрос\nлибо нажмите enter, чтобы использовать запрос по умолчанию:\n")
    if a:
        my_q = a
    print("query: ", my_q)
    return my_q
    
def main():
    corpus, docs_names = get_corpus()
    print("number of documents: ", len(corpus))
    my_query = lets_test()

    # препроцессинг корпуса
    res = preproc(corpus)

    # обратный индекс, в значениях: слагаемые BM25
    sparce_matrix, count_vectorizer = inverse_index(res)

    # препроцессинг и векторизация запроса
    is_ok, my_vector = query_indexing(my_query, count_vectorizer)
    
    if is_ok: # query wasnt empty
        # вектор близости: запрос и каждый документ из корпуса
        sim = similarity(sparce_matrix, my_vector)
        lets_sort(sim, docs_names)
    else:
        print(my_vector)

if __name__ == '__main__':
    main()