
import torch


class AttentionMask:

    @staticmethod
    def causal_mask(
        seq_len,
        device,
    ):

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            )
        )

        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def padding_mask(
        input_ids,
        pad_token_id,
    ):

        mask = (
            input_ids != pad_token_id
        )

        return mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def combine_masks(
        causal_mask,
        padding_mask,
    ):

        return causal_mask & padding_mask
