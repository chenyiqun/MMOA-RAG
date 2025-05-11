# MMOA-RAG

This repository contains the code for MMOA-RAG, a system for multi-modules optimization involving Query Rewriter, Retriever, Selector and Generator. The code is organized into several components that facilitate the deployment, training, and evaluation of the RAG system.

## Table of Contents

- [Computational Resource Requirements](#computational-resource-requirements)
- [Deploying the Retrieval Model](#deploying-the-retrieval-model)
- [Getting the SFT and MAPPO Training Data](#getting-the-sft-and-mappo-training-data)
- [Warm Start for RAG System](#warm-start-for-rag-system)
- [Multi-Agent Optimization for RAG System](#multi-agent-optimization-for-rag-system)
- [Evaluation](#evaluation)
- [Others](#others)

## Computational Resource Requirements

We used two servers, each equipped with 8 A800 GPUs (each with 80GB of memory), for training MMOA-RAG. One server was dedicated to deploying the retrieval model, while the other was used for training MARL.

Why is a separate machine needed to deploy the retrieval model? During the MARL training process, updates to the Query Rewriter are involved, and it is necessary to obtain Top-k documents in real-time during Rollout. This requires high real-time performance from the retrieval model. Therefore, we deployed the retrieval model on a separate machine using Faiss and leveraged GPU acceleration to ensure fast retrieval results.

## Deploying the Retrieval Model

The retrieval models are deployed using a specialized machine due to the multi-modules optimization that involves the training of the Query Rewriter.

To deploy the retrieval model, execute the following:

1. Ensure the code in `./flask_server.py` is properly configured.
2. Start the retrieval model API by running in one server:
   ```bash
   bash run_server.sh
   ```

## Getting the SFT and MAPPO Training Data
To generate the training data for SFT and MAPPO processes, follow these steps:

Run the following script to obtain the SFT training data:
   ```bash
   python qr_s_g_sft_data_alpaca.py
   ```

Run the following script to get the MAPPO training data for each dataset:
   ```bash
   python get_ppo_data_alpaca.py
   ```

**We developed the code of MAPPO to joint optimizing multiple modules in RAG system based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the core code can be seen at:**
   ```bash
   ./LLaMA-Factory/src/llamafactory/train/ppo/trainer_qr_s_g.py
   ```

## Warm Start for RAG System
To warm start multiple modules in the RAG system using SFT, execute:
   ```bash
   bash LLaMA-Factory/run_sft.sh
   ```

## Multi-Agent Optimization for RAG System
To perform joint learning of the multiple modules in the RAG system using MAPPO, run the following command in another server:
   ```bash
   bash LLaMA-Factory/run_mappo.sh
   ```

## Evaluation
Evaluate the performance of the RAG system by executing:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python evaluate_qr_s_g.py
   ```

## Others
Create necessary directories: 
1. `./data` for storing data sets. For example, `./data/ambigqa` is used to save the AmbigQA dataset.

2. `./models` for saving checkpoints of the retrieval model and LLMs.

