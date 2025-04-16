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
from normalize_answers import *
from collections import Counter
import os
import random
from transformers import LogitsProcessorList, LogitsProcessor
from limited_tokens import get_allowed_tokens, AllowedTokensLogitsProcessor


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

def normalize_answer_final(answer):
    pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
    final_answer = normalize_answer(pre_answer)
    return final_answer

def extract_digits(input_string, K_candidate):
    input_list = input_string.split(',')
    input_list = [item.replace("Document", "") for item in input_list]

    # 创建一个空列表来存储结果
    digits_list = []
    
    # 遍历输入字符串中的每个字符
    candidate_list = [str(i) for i in range(K_candidate)]
    for char in input_list:
        # 检查字符是否为数字
        if char in candidate_list:
            # 将字符转换为整数并添加到列表中
            digits_list.append(int(char))

    # K_candidate之后的不要
    digits_list = digits_list[: K_candidate]

    # 去重复
    my_list = digits_list
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)

    # 消除不在范围内的
    return_list = []
    for item in unique_list:
        if item >= 0 and item < K_candidate:
            return_list.append(item)
    
    return return_list

def compute_scores(predict_answers, golden_answers):
    assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
    final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
    total = len(predict_answers)

    for prediction, ground_truth in zip(predict_answers, golden_answers):
        normalized_prediction = normalize_answer_final(prediction)
        normalized_ground_truth = normalize_answer_final(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
            final_metric["acc"] += 1.0

        if normalized_prediction == normalized_ground_truth:
            final_metric["em"] += 1.0

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["f1"] += f1
        final_metric["precision"] += precision
        final_metric["recall"] += recall

    for k in ['acc', 'em', 'f1', 'precision', 'recall']:
        final_metric[k] /= total

    return final_metric

def get_response(messages):
    if tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).cuda()
    else:
        conv = get_conversation_template(generator_model_path)
        for message in messages:
            conv.append_message(message["role"], message["content"])
        conv.append_message("assistant", "")
        input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # prefix_allowed_tokens_fn=lambda batch_id, prev_ids: allowed_tokens,
    )
    outputs = tokenizer.decode(
        outputs[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return outputs

def batch_get_response(model, tokenizer, messages_list, logits_processor=[]):
    # Create a list of input_ids for all messages in the batch
    input_ids_list = []
    for messages in messages_list:
        if tokenizer.chat_template is not None:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).cuda()
        else:
            conv = get_conversation_template(generator_model_path)
            for message in messages:
                conv.append_message(message["role"], message["content"])
            conv.append_message("assistant", "")
            input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
        
        input_ids_list.append(input_ids)

    # Pad input_ids to the same length and create a single tensor
    input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]  # 向量需要是一维的

    # 找出最长序列的长度
    max_length = max(input_ids.size(0) for input_ids in input_ids_list)
    # 手动左填充
    input_ids_padded = torch.stack([
        torch.cat([input_ids.new_full((max_length - input_ids.size(0),), tokenizer.eos_token_id), input_ids], dim=0)
        for input_ids in input_ids_list
    ], dim=0)

    # 创建 attention mask
    attention_masks = torch.stack([
        torch.cat([torch.zeros(max_length - input_ids.size(0), dtype=torch.long), torch.ones(input_ids.size(0), dtype=torch.long)], dim=0)
        for input_ids in input_ids_list
    ], dim=0).cuda()

    # Generate outputs for all inputs in the batch
    if len(logits_processor) != 0:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=29,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_masks,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode each generated output separately
    results = []
    for i in range(outputs.shape[0]):
        # 因为是左填充，所以生成的输出部分从填充长度位置开始
        output = tokenizer.decode(
            outputs[i, input_ids_padded[i].size(0):],
            skip_special_tokens=True,
        )
        results.append(output)

    return results

def get_selector_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n\n".format(str(doc_id), str(doc_content))

    message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(top_k_docs)-1)}, 
        {'role': 'assistant', 'content': 'Okay, I will provide the ID of candidate Documents which are helpful in answering the Question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the candidate Documents."}
    ]

    return message

def get_generator_prefix_role_prompt(question, top_k_docs):
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(top_k_docs)):
        doc_content = top_k_docs[doc_id]#['content']
        input_content = input_content + "Document {}: {}\n\n".format(str(doc_id), str(doc_content))

    if len(top_k_docs) > 0:
        input_content = input_content + "Now, answer the Question: {}, based on the above Documents".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question and the corresponding Documents."}
    ]

    elif len(top_k_docs) == 0:
        input_content = input_content + "Now, answer the Question: {}.".format(str(question))
        message = [
        {'role': 'system', 'content': "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."},
        {'role': 'assistant', 'content': 'Okay, I will provide the answer to the question. Please provide the question.'},
        {'role': 'user', 'content': input_content},
        {'role': 'assistant', 'content': "OK, I received the Question."}
    ]

    return message

def get_selector_post_role_prompt(question, top_k_docs):

    return {'role': 'user', 'content': "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: Document0,Document4,Document6,Document7.".format(len(top_k_docs)-1, str(question), len(top_k_docs)-1)}

def get_generator_post_role_prompt(top_docs):

    if len(top_docs) > 0:
        message = {'role': 'user', 'content': "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."}
    elif len(top_docs) == 0:
        message = {'role': 'user', 'content': "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer** and nothing else."}

    return message

def get_messages(question, top_docs):
    messages = get_generator_prefix_role_prompt(question, top_docs)
    messages.append(get_generator_post_role_prompt(top_docs))

    return messages

def get_selector_messages(question, top_docs):
    messages = get_selector_prefix_role_prompt(question, top_docs)
    messages.append(get_selector_post_role_prompt(question, top_docs))

    return messages

def get_qr_messages(question):
    messages = [
        {'role': 'system', 'content': "You are a professional assistant skilled at rewriting complex or unclear questions into simpler, more searchable subquestions."},
        {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
        {'role': 'user', 'content': "Please help me rewrite or decompose the given questions into sub-questions, making them easier to search for answers in a search engine. The rewritten sub-questions must have logical connections and dependencies, without being overly repetitive in meaning. Additionally, avoid using vague demonstrative pronouns and similar terms"},
        {'role': 'assistant', 'content': 'Okay, I will provide the rewritten sub-questions.'},
        {'role': 'user', 'content': "Original question is '{}'. Now rewrite or decompose the original question into sub-questions according to the above requirements, and only output the rewritten subquestions in the format of one subproblem per line without any additional content. Additionally, avoid using vague demonstrative pronouns and similar terms, avoid the duplicate subquestions.".format(question)}
    ]

    return messages

dataset_name = '2wikimultihopqa'

top_k_docs_path = './data/ambigqa/val_top_k_docs.jsonl'.format(dataset_name)
generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/{}/qr_s/ppo_1'.format(dataset_name)
# generator_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/2wikimultihopqa/qr_selector_and_generator/sft_1'

# generator_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'

# # selector_model_path = '/root/paddlejob/workspace/env_run/rag/models/LLM-Research/Meta-Llama-3-8B-Instruct'
# selector_model_path = '/root/paddlejob/workspace/env_run/rag/LLaMA-Factory/models/llama3-8b/{}/selector_and_generator/ppo_5_07'.format(dataset_name)

# loading questions-answer-topk_docs data
print('*'*20)
print('loading questions-answer-topk_docs data')
start_time = time.time()
# 创建一个空列表来存放字典
top_k_list = []
# 打开文件以读取
with open(top_k_docs_path, 'r') as file:
    for line in file:
        # json.loads() 函数将每行的json字符串转化为字典
        top_k_list.append(json.loads(line))
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# loading generator model
print('*'*20)
print('loading generator model')
model = AutoModelForCausalLM.from_pretrained(
    generator_model_path,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
model = model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# random.shuffle(top_k_list)
# top_k_list = top_k_list[:2415]

all_questions = []
all_answers = []
all_top_docs = []
for i in range(len(top_k_list)):
    cur_que_ans_docs = top_k_list[i]
    all_questions.append(cur_que_ans_docs['question'])
    all_answers.append(cur_que_ans_docs['answer'])
    all_top_docs.append([doc['content'] for doc in cur_que_ans_docs['top_k_docs']])

# answers
qr_answers = []
selector_answers = []
generator_answers = []

# *******************************query rewriter*******************************

qr_messages_list = []
for i in range(len(all_questions)):
    init_question = all_questions[i]
    qr_messages = get_qr_messages(init_question)
    qr_messages_list.append(qr_messages)

batch_size = 16
print('*'*40)
print('query rewriting.')
for i in tqdm(range(0, len(qr_messages_list), batch_size)):
    messages_batch = qr_messages_list[i: i+batch_size]
    answers_batch = batch_get_response(model, tokenizer, messages_batch)
    for answer in answers_batch:
        qr_answer_list = answer.split('\n')
        qr_answer_list = [q.strip() for q in qr_answer_list]
        qr_answers.append(qr_answer_list)

        if i < 20:
            print(qr_answer_list)
            print('\n')


#Added code for saving the file
# ✅ Save query rewrite pairs to query_rewrite_hotpotqa.json
qr_pairs = []
for i in range(len(all_questions)):
    source = all_questions[i]
    target_list = qr_answers[i]
    target = ";".join(target_list)
    qr_pairs.append({
        "source": source,
        "target": target
    })

save_path = "./data/query_rewrite_data/query_rewrite_hotpotqa.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(qr_pairs, f, ensure_ascii=False, indent=2)

print(f"✅ Saved query rewrite data to: {save_path}")

# 1. 删除模型
del model
del tokenizer
# 2. 清空缓存
torch.cuda.empty_cache()

print('*'*20)
print('retrieval documents of subquestions')
pre_path = '/root/paddlejob/workspace/env_run/rag/data/'
retriever_model_path = '/root/paddlejob/workspace/env_run/rag/models/facebook/contriever'
index_path = pre_path+'wikipedia.contriever'
docs_path = pre_path+'psgs_w100.tsv'
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

qr_all_questions = []
qr_all_subquestions_docs_dict = {}
for subq_list in qr_answers:
    for q in subq_list:
        qr_all_subquestions_docs_dict[q] = []
        qr_all_questions.append(q)
print('len(qr_all_questions)', len(qr_all_questions))
for i in tqdm(range(0, len(qr_all_questions), batch_size)):
    batch_questions = qr_all_questions[i: i+batch_size]
    batch_questions_embeddings = get_embeddings(batch_questions)
    batch_D, batch_I = index.search(batch_questions_embeddings, 10)
    for q_id in range(len(batch_questions)):
        temp_topk_docs = [str(df.loc[doc_id, 'title'])+'\n'+str(df.loc[doc_id, 'text']) for doc_id in batch_I[q_id]]
        qr_all_subquestions_docs_dict[batch_questions[q_id]] = temp_topk_docs
        # print('batch_I[q]', batch_I[q])
        # print('temp_topk_docs', len(temp_topk_docs))
        # print('\n')
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# with open('./qr_all_subquestions_docs_dict.txt', 'w', encoding='utf-8') as file:
#     json.dump(qr_all_subquestions_docs_dict, file, ensure_ascii=False, indent=4)

# # 将字典保存到一个 JSON Lines 文件中
# with open('./qr_all_subquestions_docs_dict.jsonl', 'w', encoding='utf-8') as file:
#     for key, value in qr_all_subquestions_docs_dict.items():
#         # 创建一个新的字典来表示一行数据
#         line_data = {key: value}
#         # 将字典转化为 JSON 字符串并写入文件
#         json_line = json.dumps(line_data, ensure_ascii=False)
#         file.write(json_line + '\n')


# 
shuffled_all_top_docs = []
for i in range(len(qr_answers)):
    subq_list = qr_answers[i][:4]  # 改写后的query不超过4个
    # num_docs_per_subq_dict = {'1': 10, '2': 5, '3': 5, '4': 4, '5': 3}  # 分别为10 10 15 16 15个候选文档
    num_docs_per_subq_dict = {'1': [10], '2': [5,5], '3': [4,3,3], '4': [3,3,2,2], '5': [2,2,2,2,2]}  # 分别为10 10 12 12 10个候选文档
    num_docs_per_subq = num_docs_per_subq_dict[str(len(subq_list))]

    # temp_doc_list = all_top_docs[:4]  # 原问题的前5 doc + subquestion的doc
    temp_doc_list = []  # 只有subquestion的doc

    # for subq in subq_list:
    for subq_id in range(len(subq_list)):
        subq = subq_list[subq_id]
        temp_doc_list = temp_doc_list + qr_all_subquestions_docs_dict[subq][: num_docs_per_subq[subq_id]]
    shuffled_all_top_docs.append(temp_doc_list)

    with open("./sft_1epoch_qr.txt", "a", encoding="utf-8") as file:
        file.write(str(i) + ': ' + str(len(temp_doc_list)) + ', len(subq_list): {}'.format(len(subq_list)) + '\n')


# 1. 删除索引对象
del index
# 2. 删除 CPU 索引对象
del read_index
# 3. 删除资源对象
del res

# 1. 删除模型
del retriever_model
del retriever_tokenizer
# 2. 清空缓存
torch.cuda.empty_cache()

# loading generator model
print('*'*20)
print('loading generator model')
model = AutoModelForCausalLM.from_pretrained(
    generator_model_path,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
model = model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
tokenizer.padding_side = "left"
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))


# *******************************selector*******************************

# get selector answers list
selector_messages_list = []
for i in range(len(top_k_list)):
    question = all_questions[i]
    top_docs = shuffled_all_top_docs[i]
    selector_messages = get_selector_messages(question, top_docs)
    selector_messages_list.append(selector_messages)
print("len(selector_messages_list): {}".format(len(selector_messages_list)))

batch_size = 16
print('*'*40)
print('selector generating.')
# 定义允许的token
# allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', ',', 'Document']
allowed_tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', 'Document']
allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
eos_token_id = tokenizer.eos_token_id
allowed_token_ids.append(eos_token_id)
# 创建LogitsProcessor
logits_processor = LogitsProcessorList([
    AllowedTokensLogitsProcessor(allowed_token_ids)
])
for i in tqdm(range(0, len(selector_messages_list), batch_size)):
    messages_batch = selector_messages_list[i: i+batch_size]
    answers_batch = batch_get_response(model, tokenizer, messages_batch, logits_processor)
    # for answer in answers_batch:
    for j in range(len(answers_batch)):
        answer = answers_batch[j]
        number_answer = extract_digits(answer, K_candidate=len(shuffled_all_top_docs[i+j]))
        selector_answers.append(number_answer)
        # selector_answers.append([0,1,2,3,4])

        with open("./sft_1epoch_selector.txt", "a", encoding="utf-8") as file:
            file.write(answer + " || " + str(len(shuffled_all_top_docs[i+j])) + '\n')

        if i < 20:
            print(answer, number_answer)

# *******************************generating*******************************
start_time = time.time()

generator_messages_list = []
for i in range(len(top_k_list)):
    question = all_questions[i]
    top_docs = shuffled_all_top_docs[i]
    selected_ids = selector_answers[i]
    generator_input_docs = []
    for selected_id in selected_ids:
        generator_input_docs.append(top_docs[selected_id])
    # print(i, len(generator_input_docs))
    messages = get_messages(question=question, top_docs=generator_input_docs)
    generator_messages_list.append(messages)
print("len(generator_messages_list): {}".format(len(generator_messages_list)))

batch_size = 8
print('*'*40)
print('generator generating.')
for i in tqdm(range(0, len(generator_messages_list), batch_size)):
    messages_batch = generator_messages_list[i: i+batch_size]
    answers_batch = batch_get_response(model, tokenizer, messages_batch)
    # print(answers_batch)
    # for answer in answers_batch:
    for j in range(len(answers_batch)):
        answer = answers_batch[j]
        answer = answer.replace("*", "")
        generator_answers.append(answer)

        with open("./sft_1epoch_generator.txt", "a", encoding="utf-8") as file:
            file.write(answer + '\n')

        if i < 20:
            print(answer)

# getting F1 score
print(compute_scores(generator_answers, all_answers))

save_results_path = "/root/paddlejob/workspace/env_run/rag/data/{}".format(dataset_name)
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
# 保存答案为jsonl
with open(save_results_path+'/ppo_1_ablation_qr_s.jsonl', 'w') as file: # 
    for i in range(len(generator_answers)):
        temp_dic = {'question_id': str(i), 'question': str(all_questions[i]), 'predict_answer': str(generator_answers[i]), 'golden_answer': str(all_answers[i])}
        file.write(json.dumps(temp_dic) + '\n')

end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

print('Finished!')
