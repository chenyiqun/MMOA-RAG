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
import re
import random

def re_replace(old_string):
    pattern = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5（）().,，!！？；;:：、\n -]')
    new_string = pattern.sub('', old_string)

    return new_string

def get_positive_position(list_length=10, selected_number=3):
    # 初始化一个全为0的列表
    binary_list = [0] * list_length
    # 随机选择 selected_number 个不同的索引
    indices_to_set = random.sample(range(list_length), selected_number)
    # 将选定索引位置的元素置为1
    for index in indices_to_set:
        binary_list[index] = 1

    return binary_list

def remove_punctuation(text):
    # 定义要去除的标点符号
    punctuation = set('.,!?;:"()[]{}-')
    # 过滤掉标点符号
    return ''.join(char for char in text if char not in punctuation)

def clean_and_split(text):
    # 去除标点符号
    cleaned_text = remove_punctuation(text).lower()
    # 以空格为分隔符分割单词
    words = cleaned_text.split()
    return words

def calculate_match_ratio(answer, document, i):
    # 常见词汇列表，包括介词、冠词、代词、连词、助动词、否定词和疑问词
    common_words = {
        # 常见介词、冠词、代词、连词、助动词、否定词、疑问词
        "in", "on", "at", "to", "for", "with", "by", "from", "about",
        "a", "an", "the",
        "it", "they", "we", "you", "he", "she", "i", "me", "my", "mine", "ours", "us", "your", "yours", "his", "hers", "their", "theirs",
        "and", "or", "but", "because", "if", "then", "than", "as",
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "having", "be", "been", "being",
        "not", "no", "nor", "none",
        "what", "where", "when", "who", "why", "how", "which", "whom", "whose",
        # 常用标点符号
        ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "...", "--", "/", "\\", "|", "<", ">", "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
        # 其他常见停用词
        "of", "that", "this", "these", "those", "such", "there", "here", "all", "any", "both", "each", "few", "more", "some", "most", "other", "another", "every", "either", "neither"
    }

    # 清理并分割answer和document
    answer_words = [word for word in clean_and_split(answer) if word not in common_words and len(word) > 1]
    # document_words = set(word for word in clean_and_split(document) if word not in common_words)
    document_words = remove_punctuation(document).lower()
    # document_words_2 = clean_and_split(document)

    # if i < 100 and (extra == 'yes' or extra == 'no'):
    #     print(i, answer_words, document_words)
    #     print('\n')

    # 计算answer中有多少单词出现在document中
    match_count = sum(1 for word in answer_words if word in document_words)
    # match_count += sum(1 for word in answer_words if word in document_words_2)

    # 计算比例
    if len(answer_words) == 0:
        return 0.0  # 避免除以0
    match_ratio = match_count / (2*len(answer_words))

    return match_ratio

def sort_and_classify_documents(answer, documents, i):
    # 计算每个文档的匹配比例
    document_ratios = [(document, calculate_match_ratio(answer, document, i)) for document in documents]

    # 将文档划分为两个列表：匹配比例大于0和等于0
    positive_document_ratios = [doc_ratio for doc_ratio in document_ratios if doc_ratio[1] > 0]
    negative_document_ratios = [doc_ratio for doc_ratio in document_ratios if doc_ratio[1] == 0]

    # 对匹配比例大于0的文档进行排序
    positive_document_ratios.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的文档
    sorted_positive_documents = [doc for doc, _ in positive_document_ratios]
    sorted_negative_documents = [doc for doc, _ in negative_document_ratios]

    # 返回两个列表
    return sorted_positive_documents, sorted_negative_documents

dataset_name = 'ambigqa'
top_k_docs_path = '/root/paddlejob/workspace/env_run/rag/data/{}/train_top_k_docs.jsonl'.format(dataset_name)

# loading query rewriter data
print('*'*20)
print('loading query rewriter data')
file_path = './data/query_rewrite_data/query_rewrite_hotpotqa.json'
with open(file_path, 'r', encoding='utf-8') as file:
    query_rewrite_data = json.load(file)  # 使用 json.load() 将 JSON 数据解析为字典
init_questions = []
rewritten_questions = []
query_rewrite_dict = {}
for i in range(len(query_rewrite_data)):
    init_q = query_rewrite_data[i]['source'].strip()
    rewritten_qs_list = query_rewrite_data[i]['target'].split(';')
    rewritten_qs_list = [q.strip() for q in rewritten_qs_list]
    init_questions.append(init_q)
    rewritten_questions.append(rewritten_qs_list)
    query_rewrite_dict[init_q] = rewritten_qs_list

data_list = []
file_path = '/root/paddlejob/workspace/env_run/rag/data/query_rewrite_data/top_k_qr_subq.jsonl'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)
rewritten_top_k_dict = {}
for item in data_list:
    subquestion = item['question']
    sub_top_k_list = item['top_k_docs']
    rewritten_top_k_dict[subquestion] = sub_top_k_list # {docid: content, docid: content}

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

# top_k_list = top_k_list[: int(0.5*len(top_k_list))]  # 前50%
# top_k_list = top_k_list[int(0.5*len(top_k_list)): ]  # 后50%

qr_selector_and_generator_sft_list = []
aver_num = 0.0
reject_aver_num = 0.0
selector_rm_data_list = []
for i in tqdm(range(len(top_k_list))):
    cur_que_ans_docs = top_k_list[i]
    question = cur_que_ans_docs['question']
    answer = cur_que_ans_docs['answer']
    top_k_docs = cur_que_ans_docs['top_k_docs']
    all_documents_content = [item['content'] for item in top_k_docs]

    # ****************** query rewriter ******************
    if question in query_rewrite_dict:
        qr_temp_dict = {}
        qr_temp_dict['instruction'] = "Please help me rewrite or decompose the given questions into sub-questions, making them easier to search for answers in a search engine. The rewritten sub-questions must have logical connections and dependencies, without being overly repetitive in meaning. Additionally, avoid using vague demonstrative pronouns and similar terms."

        qr_temp_dict['input'] = "Original question is '{}'. Now rewrite or decompose the original question into sub-questions according to the above requirements, and only output the rewritten subquestions in the format of one subproblem per line without any additional content. Additionally, avoid using vague demonstrative pronouns and similar terms, avoid the duplicate subquestions.".format(question)

        rewritten_qs_list = query_rewrite_dict[question]
        output_content = ""
        for q in rewritten_qs_list:
            output_content = output_content + q + '\n'
        output_content = output_content[:-1]
        qr_temp_dict['output'] = output_content

        qr_temp_dict['system'] = "You are a professional assistant skilled at rewriting complex or unclear questions into simpler, more searchable subquestions."
        qr_temp_dict['history'] = []

        qr_selector_and_generator_sft_list.append(qr_temp_dict)

        # if i < 20:
        #     print(qr_temp_dict)


    # ****************** selector ******************
    selector_temp_dict = {}
    selector_temp_dict['instruction'] = "Given the Question and {} candidate Documents, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(top_k_docs), len(top_k_docs)-1)

    input_content = "Question is: {}\n\n".format(str(question))

    if question in query_rewrite_dict:
        # rewritten_top_k_dict[subquestion] = sub_top_k_list # {docid: content, docid: content}
        all_documents_idcontent = []
        subquestions = query_rewrite_dict[question][:5]  # 截止到5个subq
        # num_docs_per_subq_dict = {'1': 10, '2': 5, '3': 5, '4': 4, '5': 3}  # 分别为10 10 15 16 15个候选文档
        num_docs_per_subq_dict = {'1': [10], '2': [5,5], '3': [4,3,3], '4': [3,3,2,2], '5': [2,2,2,2,2]}  # 分别为10 10 12 12 10个候选文档
        num_docs_per_subq = num_docs_per_subq_dict[str(len(subquestions))]
        # for subq in subquestions:
        for subq_id in range(len(subq_list)):
            subq = subq_list[subq_id]
            all_documents_idcontent += rewritten_top_k_dict[subq][: num_docs_per_subq[subq_id]]

        # all_documents_content = all_documents_content[:5] + [item['content'] for item in all_documents_idcontent]  # 原问题的前5 doc + subquestion的doc
        all_documents_content = [item['content'] for item in all_documents_idcontent]  # 只有subquestion的doc

    else:
        pass

    # get selector and generator data
    if answer == 'yes' or answer == 'no':
        active_docs_content, negative_docs_content = sort_and_classify_documents(question, all_documents_content, i)  # 如果是yes or no的问题，则用question构造selector的label.
    else:
        active_docs_content, negative_docs_content = sort_and_classify_documents(answer, all_documents_content, i)  # 如果不是yes or no，则用answer构造selector的label.

    # # get only generator data
    # active_docs_content, negative_docs_content = all_documents_content, []

    random.shuffle(active_docs_content)
    random.shuffle(negative_docs_content)
    binary_list = get_positive_position(list_length=len(all_documents_content), selected_number=len(active_docs_content))
    aver_num += len(active_docs_content)

    active_id, negative_id = 0, 0
    for b_id in range(len(binary_list)):
        if binary_list[b_id] == 0:
            doc_content = negative_docs_content[negative_id]
            input_content = input_content + "Document{}: {}\n\n".format(str(b_id), str(doc_content))
            negative_id += 1
        else:
            doc_content = active_docs_content[active_id]
            input_content = input_content + "Document{}: {}\n\n".format(str(b_id), str(doc_content))
            active_id += 1
    input_content = input_content + "Now, output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question: {}, for example, in the following format: Document0,Document4,Document6,Document7.".format(len(all_documents_content)-1, str(question))
    selector_temp_dict['input'] = input_content

    if len(active_docs_content) > 0:
        selector_output = ""
        for b_id in range(len(binary_list)):
            if binary_list[b_id] == 1:
                selector_output = selector_output + 'Document{},'.format(str(b_id))
        selector_output = selector_output[:-1]
        selector_temp_dict['output'] = selector_output
    else:
        selector_temp_dict['output'] = ''

    selector_temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0,1,2,...,{}) which are helpful in answering the Question.".format(len(all_documents_content)-1)

    selector_temp_dict['history'] = []

    # if i < 100 and (answer == 'yes' or answer == 'no'):
    #     print('****************** selector ******************')
    #     print('i: {}.'.format(i), 'output of selector: {}'.format(selector_temp_dict['output']))
    #     print('****************** selector ******************')
    #     print('\n')

    # if question in query_rewrite_dict and (len(all_documents_content)>10):
    #     print(selector_temp_dict['output'])
    #     print('\n')

    # 是否包含selector的数据
    qr_selector_and_generator_sft_list.append(selector_temp_dict)


    # ****************** selector rm data ******************
    selector_rm_temp_dict = {}
    selector_rm_temp_dict['instruction'] = selector_temp_dict['instruction']
    selector_rm_temp_dict['input'] = selector_temp_dict['input']
    selector_rm_temp_dict['chosen'] = selector_temp_dict['output']

    if len(negative_docs_content) > 0:
        selector_rejected_output = ""
        for b_id in range(len(binary_list)):
            if binary_list[b_id] == 0:
                selector_rejected_output = selector_rejected_output + '{},'.format(str(b_id))
        selector_rejected_output = selector_rejected_output[:-1]
        selector_rm_temp_dict['rejected'] = '' + selector_rejected_output + ''
    else:
        selector_rm_temp_dict['rejected'] = ''

    reject_aver_num += len(negative_docs_content)

    selector_rm_data_list.append(selector_rm_temp_dict)
    
    # if i < 200:
    #     print(i, 'chosen: {}, rejected: {}'.format(selector_rm_temp_dict['chosen'], selector_rm_temp_dict['rejected']))
    #     print('\n')

    # ****************** generator ******************

    # # only for construct ppo data
    # active_docs_content = active_docs_content + negative_docs_content

    temp_dict = {}
    input_content = "Question is: {}\n\n".format(str(question))
    for doc_id in range(len(active_docs_content)):
        doc_content = active_docs_content[doc_id]
        input_content = input_content + "Document{}: {}\n\n".format(str(doc_id), str(doc_content))

    if len(active_docs_content) > 0:
        temp_dict['instruction'] = "Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer** and nothing else."
        input_content = input_content + "Now, answer the Question: {}, based on the above Documents".format(str(question))
        temp_dict['input'] = input_content
        temp_dict['output'] = '**' + str(answer) + '**'
        temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."
        temp_dict['history'] = []

    elif len(active_docs_content) == 0:
        temp_dict['instruction'] = "Given the Question, predict the answer to the Question as briefly and accurately as possible. Only give the brief and accurate answer with the form of **answer** and nothing else."
        input_content = input_content + "Now, answer the Question: {}.".format(str(question))
        temp_dict['input'] = input_content
        temp_dict['output'] = '**' + str(answer) + '**'
        temp_dict['system'] = "You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question. If you don't know the answer to a question, please don't share false information. Answer the question as accurately as possible."
        temp_dict['history'] = []


    # if i < 200 and len(active_docs_content) == 0:
    #     print('****************** generator ******************')
    #     print(i, len(active_docs_content), temp_dict)
    #     print('****************** generator ******************')

    # if i < 2:
    #     print('****************** generator ******************')
    #     print(i, len(active_docs_content), temp_dict)
    #     print('****************** generator ******************')
    #     print('\n')

    qr_selector_and_generator_sft_list.append(temp_dict)


# if dataset not hotpotqa
# ****************** query rewriter ******************
for question, rewritten_subq_list in query_rewrite_dict.items():
    qr_temp_dict = {}
    qr_temp_dict['instruction'] = "Please help me rewrite or decompose the given questions into sub-questions, making them easier to search for answers in a search engine. The rewritten sub-questions must have logical connections and dependencies, without being overly repetitive in meaning. Additionally, avoid using vague demonstrative pronouns and similar terms."

    qr_temp_dict['input'] = "Original question is '{}'. Now rewrite or decompose the original question into sub-questions according to the above requirements, and only output the rewritten subquestions in the format of one subproblem per line without any additional content. Additionally, avoid using vague demonstrative pronouns and similar terms, avoid the duplicate subquestions.".format(question)

    rewritten_qs_list = query_rewrite_dict[question]
    output_content = ""
    for q in rewritten_qs_list:
        output_content = output_content + q + '\n'
    output_content = output_content[:-1]
    qr_temp_dict['output'] = output_content

    qr_temp_dict['system'] = "You are a professional assistant skilled at rewriting complex or unclear questions into simpler, more searchable subquestions."
    qr_temp_dict['history'] = []

    qr_selector_and_generator_sft_list.append(qr_temp_dict)


# 打乱列表
random.shuffle(qr_selector_and_generator_sft_list)

aver_num = aver_num / len(top_k_list)
reject_aver_num = reject_aver_num / len(top_k_list)
print('aver_num: {}'.format(aver_num))
print('reject_aver_num: {}'.format(reject_aver_num))

save_results_path = "/root/paddlejob/workspace/env_run/rag/data/{}".format(dataset_name)
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

# 保存答案为json
with open(save_results_path+'/{}_train_sft_qr_selector_generator_data.json'.format(dataset_name), 'w') as file:
    json.dump(qr_selector_and_generator_sft_list, file)

# # 保存答案为json
# with open(save_results_path+'/{}_train_ppo_selector_generator_data.json'.format(dataset_name), 'w') as file:
#     json.dump(qr_selector_and_generator_sft_list, file)


print('len(qr_selector_and_generator_sft_list): ', len(qr_selector_and_generator_sft_list))

print('Finished!')
