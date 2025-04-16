import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time
import json

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    return sentence_embeddings

def get_embeddings(sentences):
    # Apply tokenizer
    sentences_input = retriever_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        sentences_input = sentences_input.to(device)
        with torch.no_grad():  # 如果不需要梯度，使用torch.no_grad()可以减少内存消耗  
            sentences_output = retriever_model(**sentences_input)

        # Compute token embeddings
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        
        sentences_embedding = sentences_embedding.cpu().numpy()

        # Delete variables and empty cache
        del sentences_input, sentences_output
        torch.cuda.empty_cache()

    else:
        sentences_output = retriever_model(**sentences_input) 
        sentences_embedding = mean_pooling(sentences_output[0], sentences_input['attention_mask'])
        sentences_embedding = sentences_embedding.numpy()

    return sentences_embedding

def concatenate_strings(strings_list):
    if len(strings_list) == 1:
        return strings_list[0]
    else:
        return ', '.join(strings_list)

# # nq_open
# dataset_name = 'nq_open'
# df_train = pd.read_parquet('/root/paddlejob/workspace/env_run/rag/data/train-00000-of-00001.parquet')
# df_val = pd.read_parquet('/root/paddlejob/workspace/env_run/rag/data/validation-00000-of-00001.parquet')

# # hotpotqa
# dataset_name = 'hotpotqa'
# with open('/root/paddlejob/workspace/env_run/rag/data/hotpotqa/hotpotqa_train_questions_and_answers.json', 'r', encoding='utf-8') as file:
#     data_train = json.load(file)
# with open('/root/paddlejob/workspace/env_run/rag/data/hotpotqa/hotpotqa_test_questions_and_answers.json', 'r', encoding='utf-8') as file:
#     data_test = json.load(file)
# print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

# # 2wikimultihopqa
# dataset_name = '2wikimultihopqa'
# data_train, data_test = [], []
# with open('/root/paddlejob/workspace/env_run/rag/data/2wikimultihopqa/train.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         data_train.append(json.loads(line.strip()))
# with open('/root/paddlejob/workspace/env_run/rag/data/2wikimultihopqa/dev.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         data_test.append(json.loads(line.strip()))
# print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

# # musique
# dataset_name = 'musique'
# data_train, data_test = [], []
# with open('/root/paddlejob/workspace/env_run/rag/data/musique/musique_ans_v1.0_train.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         data_train.append(json.loads(line.strip()))
# with open('/root/paddlejob/workspace/env_run/rag/data/musique/musique_ans_v1.0_dev.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         data_test.append(json.loads(line.strip()))
# print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

# ambigqa
dataset_name = 'ambigqa'
data_train, data_test = [], []
with open('./data/ambigqa/train_data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data_train.append(json.loads(line.strip()))
with open('./data/ambigqa/test_data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data_test.append(json.loads(line.strip()))
print('len(data_train): {}, len(data_test): {}'.format(len(data_train), len(data_test)))

pre_path = './data/'

#retriever_model_path = '/root/paddlejob/workspace/env_run/rag/models/facebook/contriever'

retriever_model_path = "facebook/contriever"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
retriever_model = AutoModel.from_pretrained(retriever_model_path)

index_path = pre_path+'wikipedia.contriever'
docs_path = pre_path+'psgs_w100.tsv'
#top_k_docs_path = pre_path+'ambigqa/val_top_k_docs.jsonl'
top_k_docs_path = './data/ambigqa/val_top_k_docs.jsonl'

# loading retriever model
print('*'*20)
print('loading retriever model')
# 检查GPU是否可用  
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
retriever_model = AutoModel.from_pretrained(retriever_model_path)
retriever_model = retriever_model.to(device)
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading index
print('*'*20)
print('loading index')
start_time = time.time()

# # cpu faiss
# print('using cpu index')
# index = faiss.read_index(index_path)
# gpu faiss
print('put index on gpu')
res = faiss.StandardGpuResources()  # 创建一个Faiss资源对象
read_index = faiss.read_index(index_path)  # 读取索引
index = faiss.index_cpu_to_gpu(res, 0, read_index)  # 把索引放到第一个GPU上

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading docs data
print('*'*20)
print('loading data')
start_time = time.time()
df = pd.read_csv(docs_path, sep='\t')
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# searching
print('*'*20)
print('searching')
start_time = time.time()
top_k_list = []
k=10

# # nq_open
# question_num = len(df_train)
# questions = df_train['question'].tolist()[:question_num]
# answers = df_train['answer'].apply(lambda x: ', '.join(x)).tolist()[:question_num]

# other dataset except nq_open
questions = [item['question'] for item in data_test]
if dataset_name == 'ambigqa':
    answers = [concatenate_strings(item['nq_answer']) for item in data_test]
else:
    answers = [item['answer'] for item in data_test]
print('len(questions): {}'.format(len(questions)))

batch_size = 5
iterations = len(questions) // batch_size
cur_batch = 0
for b in tqdm(range(iterations)):
    batch_answers = answers[cur_batch*batch_size: min((cur_batch+1)*batch_size, len(answers))]
    batch_questions = questions[cur_batch*batch_size: min((cur_batch+1)*batch_size, len(questions))]
    batch_questions_embeddings = get_embeddings(batch_questions)
    batch_D, batch_I = index.search(batch_questions_embeddings, k)
    cur_batch += 1
    for q in range(len(batch_questions)):
        temp_dict = {}
        question = batch_questions[q]
        answer = batch_answers[q]
        temp_dict['question'] = question
        temp_dict['answer'] = answer
        temp_dict['top_k_docs'] = [{'doc_id': str(doc_id), 'content': str(df.loc[doc_id, 'title'])+'\n' + str(df.loc[doc_id, 'text'])} for doc_id in batch_I[q]]
        top_k_list.append(temp_dict)

        if b < 5:
            print(question)
            print(answer)
            print('\t')

# 保存top-k的retrieval结果
with open(top_k_docs_path, 'w') as file:
    for dictionary in top_k_list:
        # json.dumps()函数将字典转化为json字符串
        # 每个字典写入一行
        file.write(json.dumps(dictionary) + '\n')

print('Finished!')
