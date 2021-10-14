import json
import numpy as np
import gensim, torch


def fasttextvec(corpus, model, file_name):
    sent_embeds = []
    for sent in corpus:
        tokens = sent.split()
        word_embeds = []
        for el in tokens:
            t = model[el]
            word_embeds.append(t)
        if len(word_embeds) == 1:
            sent_embeds.append(word_embeds[0])
        elif len(word_embeds) == 0:
            sent_embeds.append(np.zeros(300))
        else:
            sent_embeds.append(np.mean(word_embeds, axis=0))

    # print("num of embeds", len(sent_embeds))
    if not file_name:
        return np.stack(sent_embeds, axis=0)

    torch.save(np.stack(sent_embeds, axis=0), file_name)


def main():
    # with open("preproced_corpus.json", encoding="utf8") as f:
        # corpus = json.load(f)
        # file_name = "fasttext_matrix.pt"
    with open("preproced_ques.json", encoding="utf8") as f:
        corpus = json.load(f)
        file_name = "fasttext_matrix_ques.pt"
    model = gensim.models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    print("loaded lemmatized corpus and model\nstarting fasttext vectorization...")
    fasttextvec(corpus, model, file_name)


if __name__ == '__main__':
    main()
