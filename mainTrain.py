from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pickle
import random
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self,image_feature_size,hidden_size):
        super().__init__()
        self.fc1=nn.Linear(image_feature_size,hidden_size)
        self.ln1=nn.LayerNorm(hidden_size)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)
    def forward(self,image_features):
        h=self.fc1(image_features)
        h=self.ln1(h)
        h=self.relu(h)
        h=self.dropout(h)

        c=self.fc1(image_features)
        c=self.ln1(c)
        c=self.relu(c)
        c=self.dropout(c)
        h=h.unsqueeze(0)
        c=c.unsqueeze(0)
        return h,c

class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.embed_ln = nn.LayerNorm(embedding_size)     # ← ADD

        self.lstm=nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.dropout=nn.Dropout(0.3)
        self.fc=nn.Linear(hidden_size,vocab_size)
        
    def forward(self,captions,hidden,cell):
        embeddings=self.embedding(captions)
        embeddings=self.embed_ln(embeddings)
        outputs,(hidden,cell)=self.lstm(embeddings,(hidden,cell))
        outputs = self.dropout(outputs)         # ← ADD

        output=self.fc(outputs)
        return output,hidden,cell
class Seq2Seq(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size):
        super().__init__()
        self.encoder=Encoder(2048,hidden_size)
        self.decoder=Decoder(vocab_size,embedding_size,hidden_size)
    
    def forward(self,image_features,captions):
        hidden,cell=self.encoder(image_features)
        outputs,_,_=self.decoder(captions,hidden,cell)
        return outputs

class FlickrDataset(Dataset):
    def __init__(self,features_dict,encoded_captions,img_names):
        self.data=[]
        for img in img_names:
            if img in encoded_captions and img in features_dict:
                for caption in encoded_captions[img]:
                    self.data.append((img,caption))
        self.features_dict=features_dict
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image_name,caption=self.data[idx]
        features=torch.FloatTensor(self.features_dict[image_name])
        caption=torch.LongTensor(caption)
        return features,caption

def collate_fn(batch):
    images,captions=zip(*batch) 
    images=torch.stack(images,0)
    padded_captions=pad_sequence(captions,batch_first=True,padding_value=0)
    lengths=[len(caption) for caption in captions]
    return images,padded_captions,lengths

# Load your saved pickle files
with open('flickr30k_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

with open('flickr30k_tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

word2idx = tokenizer_data['word2idx']
idx2word = tokenizer_data['idx2word']
vocab_size = tokenizer_data['vocab_size']
encoded_captions = tokenizer_data['encoded_captions']

# Split into train/val/test (80/10/10)
all_imgs = list(image_features.keys())
random.seed(42)
random.shuffle(all_imgs)
train_imgs = all_imgs[:int(0.8 * len(all_imgs))]
val_imgs = all_imgs[int(0.8 * len(all_imgs)):int(0.9 * len(all_imgs))]
test_imgs = all_imgs[int(0.9 * len(all_imgs)):]

# Create DataLoaders
train_dataset = FlickrDataset(image_features, encoded_captions, train_imgs)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

val_dataset = FlickrDataset(image_features, encoded_captions, val_imgs)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(vocab_size, embedding_size=512, hidden_size=512).to(device)

# ignore_index=0 means ignore <pad> token in loss
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs=20
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, captions, lengths in train_loader:
        images = images.to(device)
        captions = captions.to(device)

        inputs = captions[:, :-1]    # feed all except last token
        targets = captions[:, 1:]    # predict all except first token

        outputs = model(images, inputs)

        # Flatten for CrossEntropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, captions, lengths in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)

            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    # Before the training loop:

    # Inside the loop, after computing avg_val_loss:
    if avg_val_loss < best_val_loss:
       best_val_loss = avg_val_loss
       torch.save(model.state_dict(), 'best_model.pth')
       print(f"  Saved best model! (Val Loss: {avg_val_loss:.4f})")
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r-o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

def greedy_search(model, image_feature, word2idx, idx2word, max_length=20, device='cuda'):
    """Generate caption by picking the most likely word at each step."""
    model.eval()
    with torch.no_grad():
        # Encode image
        image_feature = image_feature.unsqueeze(0).to(device)
        hidden, cell = model.encoder(image_feature)

        # Start with <start> token
        input_word = torch.LongTensor([[word2idx['<start>']]]).to(device)
        generated = []

        for _ in range(max_length):
            output, hidden, cell = model.decoder(input_word, hidden, cell)
            predicted = output.argmax(dim=2)       # pick highest score word
            word_idx = predicted.item()

            if word_idx == word2idx['<end>']:       # stop if <end> generated
                break

            generated.append(idx2word[word_idx])
            input_word = predicted                  # feed predicted word as next input

    return ' '.join(generated)

import torch.nn.functional as F

def beam_search(model, image_feature, word2idx, idx2word, beam_width=5, max_length=20, device='cuda'):
    """Generate caption by keeping top-k candidates at each step."""
    model.eval()
    with torch.no_grad():
        image_feature = image_feature.unsqueeze(0).to(device)
        hidden, cell = model.encoder(image_feature)

        # Each beam: (word_sequence, score, hidden_state, cell_state)
        beams = [([word2idx['<start>']], 0.0, hidden, cell)]
        completed = []

        for _ in range(max_length):
            new_beams = []

            for seq, score, h, c in beams:
                if seq[-1] == word2idx['<end>']:
                    completed.append((seq, score))
                    continue

                input_word = torch.LongTensor([[seq[-1]]]).to(device)
                output, h_new, c_new = model.decoder(input_word, h, c)
                log_probs = F.log_softmax(output.squeeze(1), dim=1)

                # Get top beam_width candidates
                top_probs, top_indices = log_probs.topk(beam_width, dim=1)

                for i in range(beam_width):
                    new_seq = seq + [top_indices[0, i].item()]
                    new_score = score + top_probs[0, i].item()
                    new_beams.append((new_seq, new_score, h_new, c_new))

            # Keep only top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            if not beams:
                break

        # Add remaining beams to completed
        completed.extend([(seq, score) for seq, score, _, _ in beams])

        # Pick the best sequence
        best_seq = max(completed, key=lambda x: x[1])[0]

        # Convert to words, removing <start> and <end>
        caption = ' '.join([idx2word[idx] for idx in best_seq
                          if idx not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]])

    return caption

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

sample_imgs = random.sample(test_imgs, 5)

for img_name in sample_imgs:
    image_feature = torch.FloatTensor(image_features[img_name])
    greedy_caption = greedy_search(model, image_feature, word2idx, idx2word, device=device)
    beam_caption = beam_search(model, image_feature, word2idx, idx2word, beam_width=5, device=device)

    print(f"Image: {img_name}")
    print(f"Ground Truth: {tokenizer_data['captions_dict'][img_name][0]}")
    print(f"Greedy: {greedy_caption}")
    print(f"Beam:   {beam_caption}")
    print("-" * 50)


from nltk.translate.bleu_score import corpus_bleu
import nltk
nltk.download('wordnet', quiet=True)

# Generate captions for all test images
references = []
hypotheses = []

for img_name in test_imgs:
    if img_name not in image_features:
        continue

    img_feature = torch.FloatTensor(image_features[img_name])
    generated = greedy_search(model, img_feature, word2idx, idx2word, device=device)

    # Ground truth captions (list of lists of words)
    gt_captions = tokenizer_data['captions_dict'][img_name]
    ref = [cap.split() for cap in gt_captions]

    # Predicted caption (list of words)
    hyp = generated.split()

    references.append(ref)
    hypotheses.append(hyp)

# Calculate BLEU scores
bleu_4 = corpus_bleu(references, hypotheses)
bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
print(f"BLEU-1: {bleu_1:.4f}")
print(f"BLEU-4: {bleu_4:.4f}")

from collections import Counter

all_ref_tokens = []
all_hyp_tokens = []

for ref_list, hyp in zip(references, hypotheses):
    all_ref_tokens.extend(ref_list[0])   # use first ground truth caption
    all_hyp_tokens.extend(hyp)

ref_counter = Counter(all_ref_tokens)
hyp_counter = Counter(all_hyp_tokens)

# True positives = tokens that appear in both
tp = sum((hyp_counter & ref_counter).values())

precision = tp / sum(hyp_counter.values()) if sum(hyp_counter.values()) > 0 else 0
recall = tp / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")