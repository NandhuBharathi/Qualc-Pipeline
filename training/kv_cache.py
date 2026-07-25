
import torch


class KVCache:

    def __init__(self):

        self.clear()

    def clear(self):

        self.key_cache = None
        self.value_cache = None

    def update(
        self,
        key,
        value,
    ):

        if self.key_cache is None:

            self.key_cache = key
            self.value_cache = value

        else:

            self.key_cache = torch.cat(
                (
                    self.key_cache,
                    key,
                ),
                dim=2,
            )

            self.value_cache = torch.cat(
                (
                    self.value_cache,
                    value,
                ),
                dim=2,
            )

        return (
            self.key_cache,
            self.value_cache,
        )

    def get(self):

        return (
            self.key_cache,
            self.value_cache,
        )

    def length(self):

        if self.key_cache is None:
            return 0

        return self.key_cache.size(2)
