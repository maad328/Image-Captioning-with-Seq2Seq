<div align="center">

# 📸 Image Captioning with Seq2Seq

**Automatically generate natural language descriptions for images using a ResNet-50 encoder and LSTM decoder**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr30k-blue)](https://www.kaggle.com/datasets/adityajn105/flickr30k)

</div>

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER                                  │
│                                                                 │
│   Image ──► ResNet-50 ──► 2048-d ──► Linear ──► 512-d (h, c)  │
│             (frozen)      vector     + LayerNorm + ReLU         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                         hidden & cell
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        DECODER                                  │
│                                                                 │
│   <start> ──► Embedding ──► LSTM (512) ──► Linear ──► Vocab    │
│               + LayerNorm   1 layer        + Dropout   Logits  │
│                                                                 │
│   Teacher Forcing: feeds ground truth tokens during training    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Image Encoder** | ResNet-50 (pretrained, frozen) | Provides rich 2048-d feature vectors via transfer learning from ImageNet |
| **Projection** | Single Linear layer (2048 → 512) | ResNet features are already high-level; one layer suffices as an adapter |
| **Decoder** | 1-layer LSTM (512 hidden) | Sufficient for short caption sequences; avoids overfitting on Flickr30k |
| **Training** | Teacher forcing | Feeds ground truth tokens as input during training for stable convergence |
| **Inference** | Greedy & Beam Search | Beam search (width=5) explores multiple candidates for better captions |

---

## 📁 Project Structure

```
Image-Captioning-with-Seq2Seq/
│
├── cleancaptions.py        # Step 1: Clean & preprocess raw captions
├── tokenization.py         # Step 2: Build vocabulary (min freq ≥ 5)
├── encodingcaptions.py     # Step 3: Encode captions to integer sequences
├── mainTrain.py            # Step 4: Train Seq2Seq model + evaluation
├── app.py                  # Step 5: Streamlit web app for inference
│
├── requirements.txt        # Python dependencies
├── best_model.pth          # Trained model weights (generated after training)
└── flickr30k_tokenizer.pkl # Tokenizer + vocabulary (generated after preprocessing)
```

---

## 🛠️ Pipeline

### Step 1 — Caption Cleaning (`cleancaptions.py`)
- Reads raw captions from Flickr30k dataset
- Lowercases text, removes special characters
- Wraps each caption with `<start>` and `<end>` tokens

### Step 2 — Tokenization (`tokenization.py`)
- Counts word frequencies across all captions
- Builds vocabulary with words appearing **≥ 5 times**
- Creates `word2idx` and `idx2word` mappings
- Adds `<pad>` and `<unk>` special tokens

### Step 3 — Encoding (`encodingcaptions.py`)
- Converts each caption from words to integer sequences
- Handles unknown words with `<unk>` token
- Saves tokenizer data to `flickr30k_tokenizer.pkl`

### Step 4 — Training (`mainTrain.py`)
- Extracts image features using pretrained ResNet-50 (2048-d vectors)
- Trains Seq2Seq model with teacher forcing
- 80/10/10 train/val/test split
- Saves best model based on validation loss
- Evaluates with BLEU-1, BLEU-4, Precision, Recall, F1

### Step 5 — Inference App (`app.py`)
- Streamlit web app with professional UI
- Upload any image → extracts ResNet-50 features on-the-fly → generates caption
- Supports both Greedy and Beam Search decoding

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, not required)

### Installation

```bash
# Clone the repository
git clone https://github.com/maad328/Image-Captioning-with-Seq2Seq.git
cd Image-Captioning-with-Seq2Seq

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training (Optional)

> Training was performed on [Kaggle](https://www.kaggle.com) using the [Flickr30k dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k). Pre-trained weights (`best_model.pth`) are included.

```bash
python mainTrain.py
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload any image and click **✨ Generate Caption**.

---

## 📊 Evaluation Metrics

The model is evaluated on the test set (10% of Flickr30k) using:

| Metric | Description |
|--------|-------------|
| **BLEU-1** | Unigram overlap with ground truth |
| **BLEU-4** | 4-gram overlap (stricter quality measure) |
| **Precision** | Fraction of predicted words that are relevant |
| **Recall** | Fraction of ground truth words that are captured |
| **F1-Score** | Harmonic mean of precision and recall |

---

## 🖼️ How It Works

1. **You upload an image** to the Streamlit app
2. **ResNet-50** (pretrained on ImageNet) extracts a 2048-dimensional feature vector — encoding what objects, textures, and scenes are in the image
3. **The Encoder** projects this vector to 512 dimensions, creating the initial hidden and cell states for the LSTM
4. **The Decoder LSTM** generates the caption word by word:
   - Starts with `<start>` token
   - At each step, predicts the next most likely word
   - Stops when it predicts `<end>` or reaches max length
5. **The caption** is displayed on screen

---

## 🗂️ Dataset

- **Flickr30k** — 31,783 images with 5 human-written captions each
- Source: [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k)
- Vocabulary built from words appearing ≥ 5 times

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding size | 512 |
| Hidden size | 512 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Epochs | 20 |
| Min word frequency | 5 |

---

## 🔮 Future Improvements

- [ ] **Attention Mechanism** — Let the decoder focus on relevant image regions per word
- [ ] **Train on MS-COCO** — 10x more images for better generalization
- [ ] **Learning rate scheduling** — Decay LR when validation loss plateaus
- [ ] **Scheduled teacher forcing** — Gradually reduce teacher forcing ratio
- [ ] **Fine-tune ResNet** — Unfreeze last conv layers for task-specific features

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using PyTorch & Streamlit**

</div>
