# Transformer Language Model

This project implements a character-level Transformer-based language model in PyTorch, trained on the Tiny Shakespeare dataset. It is inspired by the GPT architecture and demonstrates a minimalist approach to building a multi-head self-attention model.

## 📜 Dataset

The dataset used is [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), a corpus of Shakespeare's plays. It contains around 1 million characters.

## 🧠 Model Overview
The model uses:

-Token and positional embeddings

-Multi-head self-attention with causal masking

-Transformer blocks (attention + feedforward layers)

-Layer normalization

-Language modeling head to predict next characters

## Hyperparameters
batch_size = 16

block_size = 32

n_embd = 64

n_head = 4

n_layer = 4

learning_rate = 1e-3

dropout = 0.0

max_iters = 5000

## 📦 Architecture Components

• Head: One attention head with masking

• MultiHeadAttention: Combines multiple heads and projects the output

• FeedForward: A simple 2-layer MLP

• Block: A transformer block (layer norm + self-attn + MLP)

• BigramLanguageModel: The full model, outputs logits and optionally computes loss

## General Transformer archtecture
![image](https://github.com/user-attachments/assets/2337077e-0688-4983-9558-a31c93ee3ffe)

![image](https://github.com/user-attachments/assets/55c5219a-c577-4fae-bd90-1b73a420e137)

## 🏋️ Training
The model is trained to minimize cross-entropy loss by predicting the next character in a sequence.

• Training Loop

• Trains for 5000 iterations

• Evaluates loss on validation set every 500 steps

## 
*A PyTorch implementation inspired by the "Attention is All You Need" paper*

This notebook builds a character-level language model using a Transformer architecture, referencing the seminal paper [**"Attention is All You Need"**](https://arxiv.org/abs/1706.03762). It uses self-attention to model long-range dependencies in text without recurrence or convolution, following principles behind models like GPT.
Uses AdamW optimizer with simple learning rate decay

