import pickle
encoded_captions={}
for img_name,captions in data.items():
    encoded_captions[img_name]=[]
    for caption in captions:
        encoded_caption=[]
        for word in caption.split():
            if word in word2idx:
                encoded_caption.append(word2idx[word])
            else:
                encoded_caption.append(word2idx["<unk>"])
        encoded_captions[img_name].append(encoded_caption)

tokenizer_data = {
    'word2idx': word2idx,
    'idx2word': idx2word,
    'vocab_size': len(vocab),
    'encoded_captions': encoded_captions,
    'captions_dict': data
}

with open('flickr30k_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)

print(f"Vocab size: {len(vocab)}")
print(f"Images: {len(encoded_captions)}")

