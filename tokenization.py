from collections import Counter
words=[]
for captions in data.values():
    for caption in captions:
     words_split=caption.split()
     words.extend(words_split)
words =Counter(words)

vocab=[word for word,count in words.items() if count>=5]
vocab = ["<pad>", "<unk>"] + vocab

# Assign index to each word
word2idx={}
for idx,word in enumerate(vocab):
    word2idx[word]=idx



# Reverse mapping
idx2word = {idx: word for word, idx in word2idx.items()}

