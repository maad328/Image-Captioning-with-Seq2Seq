import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
from PIL import Image
import os
import random

# Mixed-precision training support
from torch.cuda.amp import autocast, GradScaler

# ============================================
# OPTIMIZED SEQ2SEQ MODEL - FAST & ACCURATE
# ============================================

class ImprovedEncoder(nn.Module):
    """Encoder: projects 2048-d ResNet features into hidden_size."""
    def __init__(self, image_feature_size=2048, hidden_size=512):
        super(ImprovedEncoder, self).__init__()

        # First projection layer
        self.fc1 = nn.Linear(image_feature_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        # Second projection layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Learned projection for LSTM cell state
        self.fc_cell = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5

    def forward(self, image_features):
        """
        Args:
            image_features: (batch_size, 2048)
        Returns:
            hidden: (batch_size, hidden_size) — initial h0
            cell:   (batch_size, hidden_size) — initial c0
        """
        # First layer
        x = self.fc1(image_features)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second layer — tanh keeps values in [-1,1] for LSTM init
        hidden = self.fc2(x)
        hidden = self.ln2(hidden)
        hidden = torch.tanh(hidden)

        # Learned cell state from the same features
        cell = torch.tanh(self.fc_cell(x))

        return hidden, cell


class ImprovedDecoder(nn.Module):
    """Decoder: LSTM with weight-tied embedding/output."""
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, num_layers=2):
        super(ImprovedDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Layer Norm on embeddings
        self.embed_ln = nn.LayerNorm(embed_size)

        # 2-layer LSTM with inter-layer dropout
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)

        # Layer Norm after LSTM
        self.lstm_ln = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.3)  # Reduced from 0.5

        # Output projection
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Weight tying: share embedding weights with output layer
        # This reduces parameters and improves generalization
        self.fc.weight = self.embedding.weight

    def forward(self, captions, lengths, hidden, cell):
        """
        Args:
            captions: (batch_size, seq_len)
            lengths:  list of actual caption lengths
            hidden:   (batch_size, hidden_size) from encoder
            cell:     (batch_size, hidden_size) from encoder
        Returns:
            outputs: (batch_size, seq_len, vocab_size)
        """
        embeddings = self.embedding(captions)
        embeddings = self.embed_ln(embeddings)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # Replicate for num_layers
        h0 = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

        packed_out, _ = self.lstm(packed, (h0, c0))
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        lstm_out = self.lstm_ln(lstm_out)
        lstm_out = self.dropout(lstm_out)

        outputs = self.fc(lstm_out)
        return outputs


class ImprovedImageCaptioningModel(nn.Module):
    """Complete Seq2Seq model."""
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, num_layers=2):
        super(ImprovedImageCaptioningModel, self).__init__()
        self.encoder = ImprovedEncoder(image_feature_size=2048, hidden_size=hidden_size)
        self.decoder = ImprovedDecoder(vocab_size, embed_size, hidden_size, num_layers)

    def forward(self, image_features, captions, lengths):
        hidden, cell = self.encoder(image_features)
        outputs = self.decoder(captions, lengths, hidden, cell)
        return outputs


# ============================================
# LABEL SMOOTHING LOSS
# ============================================

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0

            mask = (target == self.padding_idx).unsqueeze(1)
            true_dist.masked_fill_(mask, 0)

        loss = -(true_dist * log_probs).sum(dim=1)

        non_pad_mask = (target != self.padding_idx).float()
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss


# ============================================
# DATASET AND DATALOADER
# ============================================

class Flickr30kDataset(Dataset):
    def __init__(self, image_features_dict, encoded_captions_dict, img_names):
        self.image_features_dict = image_features_dict
        self.encoded_captions_dict = encoded_captions_dict
        self.img_names = img_names

        self.data = []
        for img_name in img_names:
            if img_name in encoded_captions_dict and img_name in image_features_dict:
                for caption in encoded_captions_dict[img_name]:
                    self.data.append((img_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image_features = torch.FloatTensor(self.image_features_dict[img_name])
        caption = torch.LongTensor(caption)
        return image_features, caption, img_name


def collate_fn(batch):
    """Collate function with proper length tracking for packed sequences"""
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_names = zip(*batch)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]

    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, padded_captions, lengths, img_names


# ============================================
# LOAD DATA
# ============================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

with open('flickr30k_tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

word2idx = tokenizer_data['word2idx']
idx2word = tokenizer_data['idx2word']
vocab_size = tokenizer_data['vocab_size']
encoded_captions = tokenizer_data['encoded_captions']
captions_dict = tokenizer_data['captions_dict']

with open('flickr30k_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

print(f"✅ Loaded {len(image_features)} image features")
print(f"✅ Vocabulary size: {vocab_size}")

# Split data: 80% train, 10% val, 10% test
all_img_names = list(image_features.keys())
np.random.seed(42)
np.random.shuffle(all_img_names)

train_size = int(0.8 * len(all_img_names))
val_size = int(0.1 * len(all_img_names))

train_imgs = all_img_names[:train_size]
val_imgs = all_img_names[train_size:train_size + val_size]
test_imgs = all_img_names[train_size + val_size:]

print(f"✅ Split - Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Create datasets
train_dataset = Flickr30kDataset(image_features, encoded_captions, train_imgs)
val_dataset = Flickr30kDataset(image_features, encoded_captions, val_imgs)
test_dataset = Flickr30kDataset(image_features, encoded_captions, test_imgs)

# Larger batch size (128 vs 64) — safe with mixed precision
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ============================================
# TRAINING FUNCTIONS
# ============================================

def calculate_accuracy(outputs, targets, pad_idx):
    predictions = outputs.argmax(dim=1)
    mask = targets != pad_idx
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
    return accuracy


def train_epoch(model, dataloader, criterion, optimizer, device, pad_idx,
                epoch, num_epochs, scaler, scheduler):
    """Training with scheduled sampling + mixed precision."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    # Scheduled sampling: 100 % teacher forcing for first 5 epochs,
    # then linearly decay to 60 % by the last epoch.
    warmup_epochs = 5
    if epoch < warmup_epochs:
        teacher_forcing_ratio = 1.0
    else:
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        teacher_forcing_ratio = 1.0 - 0.4 * progress
        teacher_forcing_ratio = max(0.6, teacher_forcing_ratio)

    for images, captions, lengths, _ in tqdm(dataloader, desc=f"Training (TF={teacher_forcing_ratio:.2f})"):
        images = images.to(device)
        captions = captions.to(device)

        cur_batch_size = captions.size(0)
        max_len = captions.size(1) - 1

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

        with autocast():
            # Encode
            hidden, cell = model.encoder(images)

            # Prepare LSTM initial states
            h0 = hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
            c0 = cell.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)

            outputs = torch.zeros(cur_batch_size, max_len, vocab_size, device=device)

            input_token = captions[:, 0].unsqueeze(1)

            for t in range(max_len):
                embedded = model.decoder.embedding(input_token)
                embedded = model.decoder.embed_ln(embedded)

                lstm_out, (h0, c0) = model.decoder.lstm(embedded, (h0, c0))
                lstm_out = model.decoder.lstm_ln(lstm_out)
                lstm_out = model.decoder.dropout(lstm_out)

                output = model.decoder.fc(lstm_out.squeeze(1))
                outputs[:, t] = output

                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing and t + 1 < captions.size(1):
                    input_token = captions[:, t + 1].unsqueeze(1)
                else:
                    input_token = output.argmax(dim=1).unsqueeze(1)

            targets = captions[:, 1:max_len + 1]
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            loss = criterion(outputs_flat, targets_flat)

        accuracy = calculate_accuracy(outputs_flat.float(), targets_flat, pad_idx)

        # Mixed-precision backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # OneCycleLR steps per batch

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


def validate(model, dataloader, criterion, device, pad_idx):
    """Validation with packed sequences and accuracy tracking."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for images, captions, lengths, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            lengths_adjusted = [l - 1 for l in lengths]

            with autocast():
                outputs = model(images, inputs, lengths_adjusted)

            seq_len = outputs.size(1)
            targets_truncated = targets[:, :seq_len]

            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets_truncated.reshape(-1)

            loss = criterion(outputs_flat.float(), targets_flat)
            accuracy = calculate_accuracy(outputs_flat.float(), targets_flat, pad_idx)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


# ============================================
# INITIALIZE AND TRAIN MODEL
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'=' * 60}")
print(f"TRAINING SETUP (OPTIMIZED MODEL)")
print(f"{'=' * 60}")
print(f"🖥️  Device: {device}")

model = ImprovedImageCaptioningModel(
    vocab_size=vocab_size,
    embed_size=512,
    hidden_size=512,
    num_layers=2
).to(device)

criterion = LabelSmoothingLoss(vocab_size, word2idx['<pad>'], smoothing=0.1)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Fewer epochs (20 vs 30) — each epoch is more effective now
num_epochs = 20

# OneCycleLR: warmup then anneal — trains faster than CosineAnnealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.002, epochs=num_epochs,
    steps_per_epoch=len(train_loader), pct_start=0.2
)

# Mixed-precision scaler
scaler = GradScaler()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable parameters: {trainable_params:,}")
print(f"\n✨ OPTIMIZATIONS ENABLED:")
print(f"   ✅ Mixed-Precision Training (AMP)")
print(f"   ✅ Batch Size 128 (↑ from 64)")
print(f"   ✅ OneCycleLR with Warmup")
print(f"   ✅ Encoder → tanh + learned cell state")
print(f"   ✅ Weight Tying (embed ↔ output)")
print(f"   ✅ Dropout 0.3 (↓ from 0.5)")
print(f"   ✅ Scheduled Sampling with 5-epoch warmup")
print(f"   ✅ Early Stopping (patience=5)")
print(f"   ✅ Label Smoothing (0.1)")
print(f"   ✅ 20 Epochs (↓ from 30)")

# Training loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
patience = 5
patience_counter = 0

print(f"\n{'=' * 60}")
print(f"STARTING TRAINING - {num_epochs} EPOCHS")
print(f"{'=' * 60}")

for epoch in range(num_epochs):
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'=' * 60}")

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device,
        word2idx['<pad>'], epoch, num_epochs, scaler, scheduler
    )
    val_loss, val_acc = validate(model, val_loader, criterion, device, word2idx['<pad>'])

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # OneCycleLR steps per batch (already done inside train), but we still
    # need per-epoch logging of the current LR.
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\n📈 Results:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
    print(f"   Learning Rate: {current_lr:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, 'best_model_improved.pth')
        print(f"   ✅ Saved best model! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%)")
    else:
        patience_counter += 1
        print(f"   ⏳ No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)

torch.save(model.state_dict(), 'final_model_improved.pth')
print("✅ Saved final model to final_model_improved.pth")


# ============================================
# PLOT LOSS AND ACCURACY CURVES
# ============================================

print("\n" + "=" * 60)
print("PLOTTING TRAINING CURVES")
print("=" * 60)

actual_epochs = len(train_losses)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(range(1, actual_epochs + 1), train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=8)
axes[0].plot(range(1, actual_epochs + 1), val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=14, fontweight='bold')
axes[0].set_title('Training and Validation Loss (OPTIMIZED)', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, actual_epochs + 1), [acc * 100 for acc in train_accuracies], 'b-o', label='Train Accuracy', linewidth=2, markersize=8)
axes[1].plot(range(1, actual_epochs + 1), [acc * 100 for acc in val_accuracies], 'r-s', label='Val Accuracy', linewidth=2, markersize=8)
axes[1].set_xlabel('Epoch', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
axes[1].set_title('Training and Validation Accuracy (OPTIMIZED)', fontsize=16, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Saved training curves to training_curves_improved.png")

print(f"\n📊 Training Summary:")
print(f"   Best Val Loss: {best_val_loss:.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
print(f"   Best Val Acc: {max(val_accuracies) * 100:.2f}% (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")
print(f"   Final Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1] * 100:.2f}%")
print(f"   Final Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1] * 100:.2f}%")


# ============================================
# INFERENCE FUNCTIONS
# ============================================

def greedy_search(model, image_feature, word2idx, idx2word, max_length=20, device='cuda'):
    """Generate caption using greedy search."""
    model.eval()

    with torch.no_grad():
        image_feature = image_feature.unsqueeze(0).to(device)
        hidden, cell = model.encoder(image_feature)

        input_word = torch.LongTensor([[word2idx['<start>']]]).to(device)
        generated = [word2idx['<start>']]

        h0 = hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        c0 = cell.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)

        for _ in range(max_length):
            embedded = model.decoder.embedding(input_word)
            embedded = model.decoder.embed_ln(embedded)

            lstm_out, (h0, c0) = model.decoder.lstm(embedded, (h0, c0))
            lstm_out = model.decoder.lstm_ln(lstm_out)

            output = model.decoder.fc(lstm_out.squeeze(1))
            predicted = output.argmax(dim=1).item()

            generated.append(predicted)

            if predicted == word2idx['<end>']:
                break

            input_word = torch.LongTensor([[predicted]]).to(device)

        caption = ' '.join([idx2word[idx] for idx in generated
                           if idx not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]])

    return caption


def beam_search(model, image_feature, word2idx, idx2word, beam_width=5,
                max_length=20, device='cuda', length_penalty_alpha=0.7):
    """Generate caption using beam search with length normalization."""
    model.eval()

    with torch.no_grad():
        image_feature = image_feature.unsqueeze(0).to(device)
        hidden, cell = model.encoder(image_feature)

        start_token = word2idx['<start>']
        h0 = hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        c0 = cell.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)

        # Each beam: (sequence, raw_log_prob, h, c)
        beams = [([start_token], 0.0, h0, c0)]
        completed = []

        for step in range(max_length):
            new_beams = []

            for seq, score, h, c in beams:
                if seq[-1] == word2idx['<end>']:
                    completed.append((seq, score))
                    continue

                input_word = torch.LongTensor([[seq[-1]]]).to(device)

                embedded = model.decoder.embedding(input_word)
                embedded = model.decoder.embed_ln(embedded)
                lstm_out, (h_new, c_new) = model.decoder.lstm(embedded, (h, c))
                lstm_out = model.decoder.lstm_ln(lstm_out)
                output = model.decoder.fc(lstm_out.squeeze(1))

                log_probs = F.log_softmax(output, dim=1)
                top_probs, top_indices = log_probs.topk(beam_width, dim=1)

                for i in range(beam_width):
                    new_seq = seq + [top_indices[0, i].item()]
                    new_score = score + top_probs[0, i].item()
                    new_beams.append((new_seq, new_score, h_new, c_new))

            # Length-normalized scoring for ranking
            def _norm_score(beam):
                seq, raw_score, _, _ = beam
                # Penalize by length to avoid favoring short captions
                lp = ((5.0 + len(seq)) / 6.0) ** length_penalty_alpha
                return raw_score / lp

            beams = sorted(new_beams, key=_norm_score, reverse=True)[:beam_width]

            if not beams:
                break

        completed.extend([(seq, score) for seq, score, _, _ in beams])

        if completed:
            # Final ranking with length penalty
            def _final_score(item):
                seq, raw_score = item
                lp = ((5.0 + len(seq)) / 6.0) ** length_penalty_alpha
                return raw_score / lp

            best_seq, _ = max(completed, key=_final_score)
        else:
            best_seq = beams[0][0]

        caption = ' '.join([idx2word[idx] for idx in best_seq
                           if idx not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]])

    return caption


# ============================================
# LOAD BEST MODEL FOR INFERENCE
# ============================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL FOR INFERENCE")
print("=" * 60)

checkpoint = torch.load('best_model_improved.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Loaded best model from epoch {checkpoint['epoch'] + 1}")
print(f"✅ Best validation loss: {checkpoint['val_loss']:.4f}")
print(f"✅ Best validation accuracy: {checkpoint['val_acc'] * 100:.2f}%")


# ============================================
# DELIVERABLE 1: CAPTION EXAMPLES
# ============================================

print("\n" + "=" * 60)
print("DELIVERABLE 1: CAPTION EXAMPLES (5 RANDOM TEST IMAGES)")
print("=" * 60)

def find_image_dir():
    base_input = '/kaggle/input'
    for root, dirs, files in os.walk(base_input):
        if len([f for f in files if f.endswith('.jpg')]) > 1000:
            return root
    return None

IMAGE_DIR = find_image_dir()

if IMAGE_DIR:
    print(f"✅ Found images at: {IMAGE_DIR}")
else:
    print("⚠️  Image directory not found. Skipping visualization.")

np.random.seed(123)
sample_indices = np.random.choice(len(test_imgs), min(5, len(test_imgs)), replace=False)
sample_imgs = [test_imgs[i] for i in sample_indices]

print(f"\n🖼️  Generating captions for {len(sample_imgs)} test images...\n")

if IMAGE_DIR:
    fig, axes = plt.subplots(len(sample_imgs), 1, figsize=(14, 4 * len(sample_imgs)))
    if len(sample_imgs) == 1:
        axes = [axes]
else:
    axes = [None] * len(sample_imgs)

for idx, img_name in enumerate(sample_imgs):
    img_features = torch.FloatTensor(image_features[img_name])

    gt_captions = captions_dict[img_name]

    greedy_caption = greedy_search(model, img_features, word2idx, idx2word, device=device)
    beam_caption = beam_search(model, img_features, word2idx, idx2word, beam_width=5, device=device)

    print(f"{'=' * 60}")
    print(f"Image {idx + 1}: {img_name}")
    print(f"{'=' * 60}")
    print(f"Ground Truth 1: {gt_captions[0]}")
    if len(gt_captions) > 1:
        print(f"Ground Truth 2: {gt_captions[1]}")
    print(f"Greedy Search:  {greedy_caption}")
    print(f"Beam Search:    {beam_caption}")
    print()

    if IMAGE_DIR and axes[idx]:
        try:
            img_path = os.path.join(IMAGE_DIR, img_name)
            img = Image.open(img_path).convert('RGB')

            axes[idx].imshow(img)
            axes[idx].axis('off')

            title = f"Image: {img_name}\n"
            title += f"Ground Truth: {gt_captions[0]}\n"
            title += f"Greedy: {greedy_caption}\n"
            title += f"Beam: {beam_caption}"

            axes[idx].set_title(title, fontsize=11, loc='left', pad=10)
        except Exception as e:
            print(f"⚠️  Could not load image: {e}")

if IMAGE_DIR:
    plt.tight_layout()
    plt.savefig('caption_examples_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved caption examples to caption_examples_improved.png")


# ============================================
# DELIVERABLE 3: QUANTITATIVE EVALUATION
# ============================================

print("\n" + "=" * 60)
print("DELIVERABLE 3: QUANTITATIVE EVALUATION")
print("=" * 60)

import subprocess
subprocess.run(['pip', 'install', 'nltk'], capture_output=True, check=True)

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("\n📊 Generating captions for test set evaluation...")

references = []
hypotheses = []
all_reference_tokens = []
all_hypothesis_tokens = []

eval_size = min(500, len(test_imgs))
print(f"📊 Evaluating on {eval_size} test images...")

for img_name in tqdm(test_imgs[:eval_size], desc="Evaluating"):
    if img_name not in image_features or img_name not in captions_dict:
        continue

    img_features = torch.FloatTensor(image_features[img_name])

    generated_caption = greedy_search(model, img_features, word2idx, idx2word, device=device)

    gt_captions = captions_dict[img_name]

    hypothesis_tokens = generated_caption.split()
    reference_tokens_list = [cap.split() for cap in gt_captions]

    hypotheses.append(hypothesis_tokens)
    references.append(reference_tokens_list)

    all_hypothesis_tokens.extend(hypothesis_tokens)
    all_reference_tokens.extend(reference_tokens_list[0])

print(f"\n✅ Evaluated {len(hypotheses)} captions")

# Calculate BLEU scores
print("\n📊 Calculating BLEU scores...")
bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

# Calculate METEOR
print("📊 Calculating METEOR score...")
meteor_scores = []
for ref_list, hyp in zip(references, hypotheses):
    score = meteor_score(ref_list, hyp)
    meteor_scores.append(score)
meteor_avg = np.mean(meteor_scores)

# Calculate Precision, Recall, F1 (token-level)
print("📊 Calculating Precision, Recall, F1...")
from collections import Counter

def calculate_prf(reference_tokens, hypothesis_tokens):
    ref_counter = Counter(reference_tokens)
    hyp_counter = Counter(hypothesis_tokens)

    tp = sum((hyp_counter & ref_counter).values())

    precision = tp / sum(hyp_counter.values()) if sum(hyp_counter.values()) > 0 else 0
    recall = tp / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

precision, recall, f1 = calculate_prf(all_reference_tokens, all_hypothesis_tokens)

# Print results
print("\n" + "=" * 60)
print("EVALUATION RESULTS (OPTIMIZED MODEL)")
print("=" * 60)

print(f"\n📊 BLEU Scores:")
print(f"   BLEU-1: {bleu_1:.4f}")
print(f"   BLEU-2: {bleu_2:.4f}")
print(f"   BLEU-3: {bleu_3:.4f}")
print(f"   BLEU-4: {bleu_4:.4f}")

print(f"\n📊 METEOR Score: {meteor_avg:.4f}")

print(f"\n📊 Token-level Metrics:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print(f"\n📊 Model Performance:")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Best Val Accuracy: {max(val_accuracies) * 100:.2f}%")
print(f"   Final Train Accuracy: {train_accuracies[-1] * 100:.2f}%")
print(f"   Final Val Accuracy: {val_accuracies[-1] * 100:.2f}%")

# Save results
results_text = f"""EVALUATION RESULTS (OPTIMIZED MODEL)
{'=' * 60}

Model Configuration:
  - Vocabulary Size: {vocab_size}
  - Embedding Size: 512
  - Hidden Size: 512
  - LSTM Layers: 2
  - Total Parameters: {total_params:,}

Optimizations Implemented:
  ✅ Mixed-Precision Training (AMP)
  ✅ Batch Size 128 (↑ from 64)
  ✅ OneCycleLR with Warmup
  ✅ Encoder → tanh + learned cell state
  ✅ Weight Tying (embed ↔ output)
  ✅ Dropout 0.3 (↓ from 0.5)
  ✅ Scheduled Sampling with 5-epoch warmup
  ✅ Early Stopping (patience=5)
  ✅ Label Smoothing (0.1)
  ✅ Length-normalized Beam Search

Training Results:
  - Best Validation Loss: {best_val_loss:.4f}
  - Best Validation Accuracy: {max(val_accuracies) * 100:.2f}%
  - Final Train Accuracy: {train_accuracies[-1] * 100:.2f}%
  - Final Val Accuracy: {val_accuracies[-1] * 100:.2f}%

BLEU Scores:
  BLEU-1: {bleu_1:.4f}
  BLEU-2: {bleu_2:.4f}
  BLEU-3: {bleu_3:.4f}
  BLEU-4: {bleu_4:.4f}

METEOR Score: {meteor_avg:.4f}

Token-level Metrics:
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1-Score:  {f1:.4f}

Training Summary:
  Epochs Trained: {actual_epochs}
  Best Epoch: {val_losses.index(min(val_losses)) + 1}
  Final Train Loss: {train_losses[-1]:.4f}
  Final Val Loss: {val_losses[-1]:.4f}
"""

with open('evaluation_results_improved.txt', 'w') as f:
    f.write(results_text)

print("\n✅ Saved evaluation results to evaluation_results_improved.txt")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("🎉 ALL DELIVERABLES COMPLETE! (OPTIMIZED MODEL)")
print("=" * 60)

print("\n📁 Generated Files:")
print("   1. best_model_improved.pth - Best model checkpoint")
print("   2. final_model_improved.pth - Final model weights")
print("   3. training_curves_improved.png - Loss & Accuracy plots")
print("   4. caption_examples_improved.png - 5 sample image captions")
print("   5. evaluation_results_improved.txt - Quantitative metrics")

print(f"\n📊 Final Model Performance:")
print(f"   Train Accuracy: {train_accuracies[-1] * 100:.2f}%")
print(f"   Val Accuracy: {val_accuracies[-1] * 100:.2f}%")
print(f"   BLEU-4: {bleu_4:.4f}")
print(f"   METEOR: {meteor_avg:.4f}")

print("\n🚀 Training completed successfully with ALL optimizations!")
print("=" * 60)