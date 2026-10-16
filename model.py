"""DNABERT-2 read encoder + DeepSets head (MetagenBERT-style).

Input:  input_ids / attention_mask of shape (B samples, R reads, L tokens)
Output: one logit per sample.
"""
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"  # pin remote code + weights


def load_dnabert2(name: str = MODEL_NAME, revision: str = MODEL_REVISION):
    """Load DNABERT-2 forced onto its plain-PyTorch attention path.

    The remote code tries `from .flash_attn_triton import ...`. With current
    Triton releases that kernel fails to compile (and Triton flash attention
    does not support the T4's sm_75 anyway). We block the `triton` import while
    loading so the remote module falls back, then null the symbol explicitly in
    case the module was already imported with Triton available.
    """
    # Passing transformers' BertConfig avoids the known config_class mismatch
    # error raised by newer transformers versions for this checkpoint.
    config = BertConfig.from_pretrained(name, revision=revision)
    saved = sys.modules.get("triton")
    sys.modules["triton"] = None  # makes `import triton` raise ImportError
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision,
                                                  trust_remote_code=True)
        encoder = AutoModel.from_pretrained(name, revision=revision, config=config,
                                            trust_remote_code=True)
    finally:
        if saved is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = saved
    attn_module = sys.modules[type(encoder.encoder.layer[0].attention.self).__module__]
    attn_module.flash_attn_qkvpacked_func = None
    encoder.pooler = None  # unused; we mean-pool token states instead
    return tokenizer, encoder


def _checkpoint_forward(layer: nn.Module) -> None:
    forward = layer.forward

    def wrapped(*args, **kwargs):
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)

    layer.forward = wrapped


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_hidden, d_out))


class DeepSets(nn.Module):
    """rho(mean_i phi(x_i)) over the set of read embeddings."""

    def __init__(self, d_in: int = 768, d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.phi = mlp(d_in, d_hidden, d_hidden, dropout)
        self.rho = mlp(d_hidden, d_hidden, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, R, d) -> (B,)
        return self.rho(self.phi(x).mean(dim=1)).squeeze(-1)


class MetagenomeClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_trainable_blocks: int = 2,
                 grad_checkpointing: bool = False, d_hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        trainable = self.encoder.encoder.layer[-n_trainable_blocks:]
        for layer in trainable:
            for p in layer.parameters():
                p.requires_grad = True
            # Frozen blocks build no autograd graph, so only these need checkpointing.
            if grad_checkpointing:
                _checkpoint_forward(layer)
        self.head = DeepSets(encoder.config.hidden_size, d_hidden)

    def embed_reads(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(N, L) token ids -> (N, d) masked-mean read embeddings."""
        hidden = self.encoder(input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, r, length = input_ids.shape
        emb = self.embed_reads(input_ids.view(b * r, length), attention_mask.view(b * r, length))
        return self.head(emb.view(b, r, -1))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
