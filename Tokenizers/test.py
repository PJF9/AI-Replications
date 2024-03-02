from Wordpiee import WordPiece
from Unigram import Unigram


corpus = []
with open("Romeo and Juliet.txt") as f:
    corpus = f.readlines()

tokenizer = Unigram(corpus=corpus, ntokens=1000) ## Or WordPiece

tokenizer.fit()

print(tokenizer.tokenize("O, what a rash and bloody deed is this!"))

text_encoding = tokenizer.encode("O, what a rash and bloody deed is this!")
print(text_encoding)

text_decoding = tokenizer.decode(text_encoding)
print(text_decoding)
