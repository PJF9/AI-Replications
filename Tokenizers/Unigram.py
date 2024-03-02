class Unigram:
    def __init__(self, corpus=None, ntokens=30_000, cleaning=None, mult=None):
        if (corpus is not None) and (cleaning is not None):
            # Cleaning Corpus
            corpus = [cleaning(text) for text in corpus]

            # Calculating the frequencies of each word (global statistics)
            self._word_freqs = defaultdict(lambda : 0)
            for text in corpus:
                for word in text.split():
                    self._word_freqs[word] += 1

            self._ntokens = ntokens # `self.__basic_vocab()` needs `ntokens`
            mult = mult if mult is not None else 3
            self._token_freqs = self.__basic_vocab(mult)
            self._model = {token: -log(freq / sum(self._token_freqs.values())) for token, freq in self._token_freqs.items()}

        self._cleaning = cleaning if (cleaning is not None) else lambda text: text
        self._ntokens = ntokens
        self.special_t = ["[CLS]", "[UNK]", "[PAD]", "[SEP]"]
        self.vocab_l = []
        self.vocab_d = {}
        self.ivocab_d = {}
        self.vocab_size = 0


    def __basic_vocab(self, mult):
        char_freq = defaultdict(lambda : 0)
        subs_freq = defaultdict(lambda : 0)

        for word, freq in self._word_freqs.items():
            for i in range(len(word)):
                char_freq[word[i]] += freq # updating chracter frequency

                for j in range(i+2, len(word) + 1): # we need to contain `len(word)` because `slice` does not contain `end index`.
                    subs_freq[word[i:j]] += freq    # updating substring frequency
        
        # Sorting Sub-String Tokens
        sorted_subs = sorted(subs_freq.items(), key=lambda x: x[1], reverse=True)

        # Concatenating those 2 token sets
        token_freq = list(char_freq.items()) + sorted_subs[: int(mult * self._ntokens)]

        return dict(token_freq)

    @staticmethod
    def __best_segmentations(word, model):
        # Contains starting character index of the best word segmentations that ends in this particular index
        best_segmentations = [{"start": 0, "score": 0}] + [{"start": None, "score": None} for _ in range(len(word))] # adding one more element for efficiency

        for start_idx in range(len(word)):
            best_score_at_start = best_segmentations[start_idx]["score"] # initially is 0, but then: `{"start": start_idx, "score": score}`

            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx: end_idx]

                if (token in model) and (best_score_at_start is not None):
                    score = model[token] + best_score_at_start

                    # If we have found a better segmentation ending at `end_idx``, we update
                    if (best_segmentations[end_idx]["score"] is None) or (best_segmentations[end_idx]["score"] > score):
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}

        return best_segmentations


    def __segment_word(self, word, model):
        best_segmentations = self.__best_segmentations(word, model) # contains the starting indexes of the best segmentation ending at each character position
        segmentation = best_segmentations[-1]               # contains the index and the score of the best segmentation ending at the last character

        if segmentation["score"] is None: return (["[UNK]"], None) # If the mode cannot segment the word, due to OOV

        score = segmentation["score"] # the score of the best segmentation ending at the last `word` character
        start = segmentation["start"] # the starting index of the best segmentation ending at the last `word` character
        end = len(word)
        tokens = []

        # Calculating the best segmentation
        while start != 0:
            tokens.insert(0, word[start: end])
            start, end = best_segmentations[start]["start"], start # start -> the index of the best segmentation ending at `end` | end -> start, we have segment the word from end to start
        
        tokens.insert(0, word[start: end]) # adding the last segment to the list

        return (tokens, score)


    def __get_loss(self, model):
        l = 0
        for word, freq in self._word_freqs.items():
            _, word_loss = self.__segment_word(word, model)
            l += (freq * word_loss)

        return l
    
    def __get_scores(self, _model):
        scores = {}
        model_loss = self.__get_loss(_model)

        for token in _model.keys():
            if len(token) == 1: # preventing the model for deleting character-level tokens
                scores[token] = float("inf")

            else:
                model_without_token = deepcopy(_model)
                model_without_token.pop(token) # removing the token from the model - Dictionary

                scores[token] = self.__get_loss(model_without_token) - model_loss

        return scores
    

    def fit(self, p_rem=0.1):
    
        while len(self._model) > self._ntokens:
            # Calculating scores and sorting them from lower to higher
            scores = sorted(self.__get_scores(self._model).items(), key=lambda x: x[1])

            # Removing the percentage specified of low-scored tokens
            for i in range(int(len(self._model) * p_rem)):
                self._token_freqs.pop(scores[i][0])

            # Initializing the new model
            self._model = {token: -log(freq / sum(self._token_freqs.values())) for token, freq in self._token_freqs.items()}

        self.vocab_l = self.special_t + sorted(list(self._token_freqs.keys()))
        self.vocab_d = {term: i for i, term in enumerate(self.vocab_l)}
        self.ivocab_d = {i: term for i, term in enumerate(self.vocab_l)}
        self.vocab_size = len(self.vocab_l)

    def save_vocab(self, path):
        # Saving the Vocabulary Dict into a JSON file
        with open(path, "w") as f:
            json.dump(self.vocab_d, f)

    def load_vocab(self, path):
        # Updating the Vocabulary elements from the JSON file
        with open(path, "r") as f:
            self.vocab_d = json.loads(f.read())
        self.vocab_l = list(self.vocab_d.keys())
        self.ivocab_d = {i: token for i, token in enumerate(self.vocab_l)}
        self.vocab_size = len(self.vocab_l)


    def tokenize(self, text, npad=0):
        tokenized_text = []

        for word in self._cleaning(text).split():
            tokenized_text.append(self.__segment_word(word, self._model)[0])
            tokenized_text.append(["[SEP]"])

        for _ in range(npad - len(tokenized_text)):
            tokenized_text.append(["[PAD]"])

        return sum(tokenized_text, []) # starting from [] and adding elements from the `tokenized_text`
    
    def encode(self, text, npad=0):
        return [self.vocab_d[token] for token in self.tokenize(text, npad=npad)]
    
    def __decode_word(self, idx):
        return ''.join([self.ivocab_d[i] for i in idx])

    def decode(self, idx):
        text = ""
        i, j = 0, 0
        while i < len(idx) and idx[i] != self.vocab_d["[PAD]"]:
            if idx[i] == self.vocab_d["[SEP]"]:
                text += self.__decode_word(idx[j: i]) + " "
                j = i + 1
            i += 1

        return text
