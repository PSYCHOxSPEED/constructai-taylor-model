import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 30522)
        self.max_seq_len = kwargs.get('max_seq_len', 128)
        self.type_vocab_size = kwargs.get('type_vocab_size', 2)
        self.hidden_size = kwargs.get('hidden_size', 384)
        self.num_layers = kwargs.get('num_layers', 6)
        self.num_heads = kwargs.get('num_heads', 8)
        self.ffn_size = kwargs.get('ffn_size', 1536)
        self.dropout = kwargs.get('dropout', 0.1)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.embedding_dim = kwargs.get('embedding_dim', 256)
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids) + self.type_embeddings(token_type_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        return x[:, 0, :]

def load_taylor_model(repo_id="constructai/taylor-v1-128-emb", device="cuda"):
    config_path = hf_hub_download(repo_id, "config.json")
    model_path = hf_hub_download(repo_id, "pytorch_model.bin")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)
    model = EmbeddingModel(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(repo_id)
    return model, tokenizer, config

def embed_texts(model, tokenizer, texts, max_len=128, normalize=True, device="cuda"):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        embs = model(input_ids, attention_mask)
    if normalize:
        embs = F.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy()