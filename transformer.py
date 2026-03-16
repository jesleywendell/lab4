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
