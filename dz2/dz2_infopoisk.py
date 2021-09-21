import os
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from string import punctuation
punctuation += "...-"


# Функция препроцессинга данных.
# Включите туда лемматизацию, приведение к одному регистру, удаление пунктуации и стоп-слов.

def preproc(my_corpus):
    res = []

    # соединяем все тексты в один, 
    # так как Mystem работает дольше при обработке нескольких текстов, чем одного большого
    alltexts = ' '.join([txt + ' endofthetext' for txt in my_corpus])

    m = Mystem()
    rus_stopwords = stopwords.words("russian")

    # приведение к нижнему регистру и лемматизация
    lemmas = m.lemmatize(alltexts.lower())

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


# функция индексации корпуса, на выходе которой посчитанная матрица Document-Term 

def inverse_index(my_corpus):
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(my_corpus) # document-term matrix
  
  return X, vectorizer

# функция индексации запроса, на выходе которой посчитанный вектор запроса

# запрос может состоять из неск слов, может только из пунктуации и чисел

def query_indexing(my_query: str, my_vectorizer):
    # предобработка
    preproc_query = preproc([my_query])

    if not preproc_query[0]:
        return False, "empty query after preprocessing, nothing found"

    # векторизация
    vec = my_vectorizer.transform(preproc_query).toarray()

    return True, vec


# функция с реализацией подсчета близости запроса и документов корпуса, 
# на выходе которой вектор, i-й элемент которого обозначает близость запроса с i-м документом корпуса

def similarity(my_matrix, my_vector):
    return cosine_similarity(my_matrix, my_vector).T # returns a row


# главная функция, объединяющая все это вместе; 
# на входе - запрос, на выходе - отсортированные по убыванию 
# имена документов коллекции

def get_doc_names(my_query):

    curr_dir = os.getcwd()
    folder = os.path.join(curr_dir, 'seasons')
    corpus = []

    # сохраняем названия документов, 
    # они соответствуют по индексам документам в corpus
    docs_names = []

    for root, dirs, _ in os.walk(folder):
        for d in dirs:
            for r, _, files in os.walk(os.path.join(root, d)):
                for name in files:
                    with open(os.path.join(r, name), 'r', encoding='utf-8-sig') as f:  
                        corpus.append(f.read())
                        # нам нужны названия документов, 
                        # чтобы выдача ранжирования была человекопонимаемая
                        docs_names.append(name)

    print("number of documents in our corpus:", len(corpus))

    # препроцессинг корпуса
    res = preproc(corpus)
    # обратный индекс, в значениях: tf-idf
    inv_index, vectorizer = inverse_index(res)
    # препроцессинг и векторизация запроса
    is_ok, my_vector = query_indexing(my_query, vectorizer)
    if is_ok: # query wasnt empty
        # вектор близости: запрос и каждый документ из корпуса
        sim = similarity(inv_index, my_vector)
        sorted_doc_ind = np.argsort(-1 * sim)
        print("Отсортированные по убыванию документы: ")
        print(np.array(docs_names)[sorted_doc_ind])
    else:
        print(my_vector)


# для примера возьмем начало файла "Friends - 1x01 - The One Where Monica Gets A Roommate.ru.txt". 
# в выдаче название этого документа печатается первым

get_doc_names("""Друзья. Как все началось

Да нечего рассказывать!
Он просто сотрудник!

Ладно тебе, ты же на свидание
с ним собралась!

Значит, он не может не быть с придурью!

Джои, веди себя прилично

Так у него горб? И парик в придачу?""")



# # пример пустой query после препроцессинга
# get_doc_names("  ,,./]- '  ")

# # выдача:
# # number of documents in our corpus: 165
# # empty query after preprocessing, nothing found
