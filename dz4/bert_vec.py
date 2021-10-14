from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import json


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


def bert_vectorizer(corpus, tokenizer, model, file_name):
    corpus = iter(corpus)
    res = []
    for el in tqdm(corpus):
        #Tokenize sentences
        encoded_input = tokenizer(el, padding=True, truncation=True, max_length=42, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
        res.append(sentence_embeddings)
    if not file_name:
       return torch.cat(res, 0)

    torch.save(torch.cat(res, 0), file_name)

def main():
    tokenizer = AutoTokenizer.from_pretrained("./sberbank-ai/sbert_large_nlu_ru/saved_tokenizer/")
    model = AutoModel.from_pretrained("./sberbank-ai/sbert_large_nlu_ru/saved_model/")

    # with open("corpus.json", "r", encoding="utf8") as f:
        # corpus = json.load(f)
        # file_name = "bert_matrix.pt"
    with open("doc_names.json", "r", encoding="utf8") as f:
        corpus = json.load(f)
        file_name = "bert_matrix_questions.pt"
    print("loaded corpus, tokenizer, model\n starting vectorization...")

    bert_vectorizer(corpus, tokenizer, model, file_name)

if __name__ == '__main__':
    main()
