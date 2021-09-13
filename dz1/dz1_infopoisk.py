import os
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
from string import punctuation
punctuation += "...-"

curr_dir = os.getcwd()
folder = os.path.join(curr_dir, 'seasons')
print(curr_dir, folder)
corpus = []

for root, dirs, _ in os.walk(folder):
    for d in dirs:
        for r, _, files in os.walk(os.path.join(root, d)):
            for name in files:
                with open(os.path.join(r, name), 'r', encoding='utf-8-sig') as f:  
                    corpus.append(f.read())

print("number of documents:", len(corpus))

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

res = preproc(corpus)

# Функция индексирования данных. 
# На выходе создает обратный индекс, он же матрица Term-Document.

def inverse_index(my_corpus):
  vectorizer = CountVectorizer(analyzer='word')
  X = vectorizer.fit_transform(my_corpus) # document-term matrix

  # транспонируем, чтобы получить term-document matrix
  return X.T, vectorizer

inv_index, vectorizer = inverse_index(res)

features = vectorizer.get_feature_names()

# a) какое слово является самым частотным
matrix_freq = np.asarray(inv_index.sum(axis=1)).ravel()
ind_max = np.argmax(matrix_freq)
most_freq = features[ind_max]
print("a) какое слово является самым частотным? Ответ:")
print(most_freq)

# b) какое самым редким
ind_min = np.argmin(matrix_freq)
least_freq = features[ind_min]
print("b) какое самым редким? Ответ:")
print(least_freq)

# c) какой набор слов есть во всех документах коллекции
def myfunction(row):
  return 0 not in row

all_non_zero = np.apply_along_axis(myfunction, 1, inv_index.toarray()) # boolean array
indexes = np.where(all_non_zero)[0] # indexes of non zero containing rows

print("c) какой набор слов есть во всех документах коллекции? Ответ:")
out_feat = [features[i] for i in indexes]
print(out_feat)


# d) кто из главных героев статистически самый популярный 
# (упонимается чаще всего)? Имена героев:

print("d) кто из главных героев упонимается чаще всего?")

characters = ["Моника", "Мон", 
"Рэйчел", "Рейч", "Рэйч", "Чендлер", 
"Чэндлер", "Чен", "Фиби", "Фибс", 
"Росс", "Джоуи", "Джои", "Джо"]

m = Mystem()
characters_lemmas = [lem for lem in m.lemmatize(' '.join(characters).lower()) if lem not in ' \n']
print("леммы имен главных героев:")
print(characters_lemmas)

freqs = defaultdict(int)

for name in characters_lemmas:
    n = name.lower()
    theind = vectorizer.vocabulary_.get(n)
    if not theind:
        print(n, "not in the vocabulary")
        continue
    if n == 'мон':
        n ='моника'
    elif n in ['рейч', 'рэйч']:
        n = 'рэйчел'
    elif n in ['чэндлер', 'чен']:
        n = 'чендлер'
    elif n == 'фибс':
        n = 'фиби'
    elif n in ['джо', 'джой']:
        n = 'джоуи'
    freqs[n] += inv_index[theind, :].sum()
    # суммируем по строке, соответствующей этому слову

fr_sorted = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

print("Частоты упоминаний главных героев:")

for k, v in fr_sorted:
    print(k, v)

print("Чаще всего упоминается: ", fr_sorted[0][0])
