import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "lab3")
from task1_causal_mask import create_causal_mask
from task2_cross_attention import cross_attention


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


def feed_forward(x, W1, b1, W2, b2):
    return F.relu(x @ W1 + b1) @ W2 + b2


def add_and_norm(x, sublayer_out):
    return F.layer_norm(x + sublayer_out, x.shape[-1:])


def init_encoder_params(d_model, d_ff):
    return {
        "W_q": torch.randn(d_model, d_model),
        "W_k": torch.randn(d_model, d_model),
        "W_v": torch.randn(d_model, d_model),
        "W1": torch.randn(d_model, d_ff),
        "b1": torch.zeros(d_ff),
        "W2": torch.randn(d_ff, d_model),
        "b2": torch.zeros(d_model),
    }


def encoder_block(x, params):
    Q = x @ params["W_q"]
    K = x @ params["W_k"]
    V = x @ params["W_v"]
    attn_out, _ = scaled_dot_product_attention(Q, K, V)
    x = add_and_norm(x, attn_out)
    ffn_out = feed_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
    return add_and_norm(x, ffn_out)


def init_decoder_params(d_model, d_ff, vocab_size):
    return {
        "W_q1": torch.randn(d_model, d_model),
        "W_k1": torch.randn(d_model, d_model),
        "W_v1": torch.randn(d_model, d_model),
        "W_q2": torch.randn(d_model, d_model),
        "W_k2": torch.randn(d_model, d_model),
        "W_v2": torch.randn(d_model, d_model),
        "W1": torch.randn(d_model, d_ff),
        "b1": torch.zeros(d_ff),
        "W2": torch.randn(d_ff, d_model),
        "b2": torch.zeros(d_model),
        "W_proj": torch.randn(d_model, vocab_size),
    }


def decoder_block(y, Z, params):
    seq_len = y.shape[-2]
    mask = create_causal_mask(seq_len)

    Q = y @ params["W_q1"]
    K = y @ params["W_k1"]
    V = y @ params["W_v1"]
    masked_attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    y = add_and_norm(y, masked_attn_out)

    cross_out, _ = cross_attention(Z, y)
    y = add_and_norm(y, cross_out)

    ffn_out = feed_forward(y, params["W1"], params["b1"], params["W2"], params["b2"])
    return add_and_norm(y, ffn_out)
