class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        ''' 
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.inverse_vocab = {v: k for k, v in vocab.items()}

        self.merges_ranks = {pair: idx for idx, pair in enumerate(merges)}

        self.special_tokens_ids = {}
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                special_token_id = self.inverse_vocab[special_token_bytes]
                self.special_tokens_ids[special_token] = special_token_id

        # self.special_token_ids = [self.inverse_vocab[b] for b in self.special_tokens]

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def decode(self, ids: list[int]) -> str:
        # return "".join([self.vocab[id].decode("utf-8") for id in ids])
        all_bytes = b"".join([self.vocab[id] for id in ids])
        return all_bytes.decode("utf-8", errors="replace")

    def bpe_encode(self, text: str) -> list[int]:
        tokens = [self.inverse_vocab[bytes([b])] for b in text.encode("utf-8")]
        while True:
            candidates = []
            for i in range(len(tokens) - 1):
                left_bytes = self.vocab[tokens[i]]
                right_bytes = self.vocab[tokens[i + 1]]
                pair = (left_bytes, right_bytes) 
                if self.merges_ranks.get(pair, None) is not None:
                    candidates.append((self.merges_ranks[pair], i, pair))
            
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            

            rank, pos, pair = candidates[0]
            
            new_bytes = pair[0] + pair[1]
            new_token_id = self.inverse_vocab[new_bytes]
            tokens[pos:pos+2] = [new_token_id]


        return tokens        
    def gpt2_pretokens_bpe(self, text: str) -> list[int]:         
        import regex
        gpt2_pattern = regex.compile(
            # r"""'s|'t|'re|'ve|'m|'ll|'d| # 常见英文缩写
            # \p{L}+|                     # 字母序列 (单词)
            # \p{N}+|                     # 数字序列
            #  ?\p{L}+|                   # 可能前置空格的字母序列
            #  ?\p{N}+|                   # 可能前置空格的数字序列
            # \s+(?!\S)|                  # 后无空格的空白 (行尾空白)
            # \s+                         # 空白序列
            # """,
            # re.VERBOSE | re.IGNORECASE  # 忽略大小写
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        pretokenized_pieces = gpt2_pattern.findall(text)
        final_token_ids = []
        for piece in pretokenized_pieces:
            final_token_ids.extend(self.bpe_encode(piece))

        return final_token_ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.gpt2_pretokens_bpe(text)
        import re

        tokens = [self.inverse_vocab[bytes([b])] for b in text.encode("utf-8")]

        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join([re.escape(token) for token in sorted_tokens])
        parts = re.split(f"({special_pattern})", text)
        result = []

        for part in parts:
            if part in sorted_tokens:
                result.append(self.special_tokens_ids[part])
            else:   
                result.extend(self.gpt2_pretokens_bpe(part))
        
        return result                 


    def encode_iterable(self, iterable):
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id
    

    
    