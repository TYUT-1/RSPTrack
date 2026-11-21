"""Layer-wise Rectified Sparse Attention (LRSA) for joint tracking tokens.

Query-dependent block retrieval uses min/max key descriptors inspired by
Quest/ReSA. RSPTrack uses bidirectional joint encoding, ordinal local blocks,
protected class/text blocks and periodic dense layers scheduled by Fast_iTPN.
It retains the full token sequence. This reference implementation masks dense
attention logits and does not imply a sparse-kernel speed or memory gain.
"""

import math

import torch
from torch import nn


class LRSABlockSparseMask(nn.Module):
    """Apply LRSA/Quest block selection to precomputed attention logits.

    Args:
        block_size: Number of consecutive key tokens represented by one block.
        active_ratio: Fraction of key blocks selected for each query. This is
            the retained ratio before adding local and protected blocks.
        min_blocks: Minimum number of blocks selected by the top-k retrieval.
        local_blocks: Number of nearby ordinal blocks always retained for each
            query. For bidirectional visual tokens these are centred around the
            query block in the joint sequence.
    """

    def __init__(self, block_size=16, active_ratio=0.5, min_blocks=4, local_blocks=1):
        super().__init__()
        if block_size <= 0:
            raise ValueError("LRSA block_size must be positive")
        if not 0.0 < active_ratio <= 1.0:
            raise ValueError("LRSA active_ratio must be in (0, 1]")
        if min_blocks <= 0:
            raise ValueError("LRSA min_blocks must be positive")
        if local_blocks < 0:
            raise ValueError("LRSA local_blocks must be non-negative")

        self.block_size = int(block_size)
        self.active_ratio = float(active_ratio)
        self.min_blocks = int(min_blocks)
        self.local_blocks = int(local_blocks)

    def _block_descriptors(self, key, key_valid):
        """Compute the element-wise min/max key descriptors for query-dependent block retrieval."""
        batch, heads, token_count, head_dim = key.shape
        block_count = math.ceil(token_count / self.block_size)
        padded_count = block_count * self.block_size
        pad_count = padded_count - token_count

        if pad_count:
            key = torch.cat(
                [key, key.new_zeros(batch, heads, pad_count, head_dim)], dim=2
            )
            key_valid = torch.cat(
                [
                    key_valid,
                    torch.zeros(batch, pad_count, dtype=torch.bool, device=key.device),
                ],
                dim=1,
            )

        key_blocks = key.view(batch, heads, block_count, self.block_size, head_dim)
        valid_blocks = key_valid.view(batch, 1, block_count, self.block_size, 1)
        max_value = torch.finfo(key.dtype).max
        min_value = torch.finfo(key.dtype).min
        block_min = key_blocks.masked_fill(~valid_blocks, max_value).amin(dim=3)
        block_max = key_blocks.masked_fill(~valid_blocks, min_value).amax(dim=3)
        block_valid = valid_blocks.any(dim=3).squeeze(-1).squeeze(1)
        return block_min, block_max, block_valid

    def build_keep_mask(self, query, key, key_padding_mask=None, protected_key_mask=None):
        """Return a boolean ``[B, H, Nq, Nk]`` mask of attended tokens."""
        batch, heads, query_count, _ = query.shape
        key_count = key.shape[2]

        if key_padding_mask is None:
            key_valid = torch.ones(batch, key_count, dtype=torch.bool, device=key.device)
        else:
            key_valid = ~key_padding_mask.bool()

        block_min, block_max, block_valid = self._block_descriptors(key, key_valid)
        block_count = block_min.shape[2]
        selected_count = max(
            self.min_blocks, int(math.ceil(block_count * self.active_ratio))
        )
        selected_count = min(selected_count, block_count)

        # Sign-aware upper-bound block score (Quest/ReSA).
        # The caller supplies the scaled query.
        query_for_score = query.float().unsqueeze(-2)
        block_min = block_min.float().unsqueeze(2)
        block_max = block_max.float().unsqueeze(2)
        block_score = torch.maximum(
            query_for_score * block_max, query_for_score * block_min
        ).sum(dim=-1)
        block_score = block_score.masked_fill(
            ~block_valid[:, None, None, :], torch.finfo(block_score.dtype).min
        )

        topk_blocks = block_score.topk(selected_count, dim=-1, sorted=False).indices
        keep_blocks = torch.zeros(
            batch,
            heads,
            query_count,
            block_count,
            dtype=torch.bool,
            device=query.device,
        )
        keep_blocks.scatter_(-1, topk_blocks, True)

        # Retain the nearest ordinal blocks centred on each query.
        if self.local_blocks:
            query_block = torch.div(
                torch.arange(query_count, device=query.device),
                self.block_size,
                rounding_mode="floor",
            )
            left = (self.local_blocks - 1) // 2
            offsets = torch.arange(
                -left, self.local_blocks - left, device=query.device
            )
            local_blocks = query_block[:, None] + offsets[None, :]
            local_valid = (local_blocks >= 0) & (local_blocks < block_count)
            local_blocks = local_blocks.clamp(0, block_count - 1)
            local_keep = torch.zeros(
                query_count, block_count, dtype=torch.bool, device=query.device
            )
            local_keep.scatter_(1, local_blocks, local_valid)
            keep_blocks |= local_keep[None, None, :, :]

        if protected_key_mask is not None:
            protected_key_mask = protected_key_mask.bool() & key_valid
            pad_count = block_count * self.block_size - key_count
            if pad_count:
                protected_key_mask = torch.cat(
                    [
                        protected_key_mask,
                        torch.zeros(
                            batch,
                            pad_count,
                            dtype=torch.bool,
                            device=query.device,
                        ),
                    ],
                    dim=1,
                )
            protected_blocks = protected_key_mask.view(
                batch, block_count, self.block_size
            ).any(dim=-1)
            keep_blocks |= protected_blocks[:, None, None, :]

        keep_tokens = keep_blocks.repeat_interleave(self.block_size, dim=-1)[
            ..., :key_count
        ]
        keep_tokens &= key_valid[:, None, None, :]
        return keep_tokens

    def forward(self, attention_logits, query, key, key_padding_mask=None, protected_key_mask=None):
        """Mask dense logits according to the selected LRSA key blocks."""
        if self.active_ratio >= 1.0:
            return attention_logits
        keep_mask = self.build_keep_mask(
            query,
            key,
            key_padding_mask=key_padding_mask,
            protected_key_mask=protected_key_mask,
        )
        return attention_logits.masked_fill(~keep_mask, float("-inf"))
