
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        max_position_embeddings=4096,
        base=10000,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

        self.max_position_embeddings = max_position_embeddings

    def forward(self, seq_len, device):

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(
            positions,
            self.inv_freq,
        )

        emb = torch.cat(
            (
                freqs,
                freqs,
            ),
            dim=-1,
        )

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat(
        (
            -x2,
            x1,
        ),
        dim=-1,
    )


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
):

    cos = cos.unsqueeze(0).unsqueeze(0)

    sin = sin.unsqueeze(0).unsqueeze(0)

    q = (q * cos) + (rotate_half(q) * sin)

    k = (k * cos) + (rotate_half(k) * sin)

    return q, k
