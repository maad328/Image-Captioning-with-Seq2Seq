import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
from torchvision import models, transforms

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioner",
    page_icon="📸",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: #888;
        font-size: 1.05rem;
    }

    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea40, transparent);
        margin: 1rem 0 1.5rem 0;
    }

    .caption-card {
        background: linear-gradient(135deg, #667eea12 0%, #764ba212 100%);
        border: 1px solid #667eea30;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .caption-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #667eea;
        margin-bottom: 0.75rem;
    }
    .caption-card .text {
        font-size: 1.3rem;
        font-weight: 500;
        color: #e0e0e0;
        line-height: 1.7;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .status-ready {
        background: #10b98120;
        color: #10b981;
        border: 1px solid #10b98140;
    }
    .status-loading {
        background: #f59e0b20;
        color: #f59e0b;
        border: 1px solid #f59e0b40;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #aaa;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .placeholder-text {
        color: #555;
        font-size: 1.1rem;
        text-align: center;
        padding: 3rem 1rem;
    }

    .footer {
        text-align: center;
        color: #555;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-bottom: 1rem;
    }

    /* Make the generate button stand out */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.85;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Architecture (must match training) ──────────────────
class Encoder(nn.Module):
    def __init__(self, image_feature_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(image_feature_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, image_features):
        h = self.dropout(self.relu(self.ln1(self.fc1(image_features))))
        c = self.dropout(self.relu(self.ln1(self.fc1(image_features))))
        return h.unsqueeze(0), c.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embed_ln = nn.LayerNorm(embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, hidden, cell):
        embeddings = self.embed_ln(self.embedding(captions))
        outputs, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        outputs = self.dropout(outputs)
        return self.fc(outputs), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(2048, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_size, hidden_size)

    def forward(self, image_features, captions):
        hidden, cell = self.encoder(image_features)
        outputs, _, _ = self.decoder(captions, hidden, cell)
        return outputs


# ─── Cached Loaders ─────────────────────────────────────────────
@st.cache_resource
def load_resnet():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    return feature_extractor


@st.cache_resource
def load_model():
    with open("flickr30k_tokenizer.pkl", "rb") as f:
        tokenizer_data = pickle.load(f)

    word2idx = tokenizer_data["word2idx"]
    idx2word = tokenizer_data["idx2word"]
    vocab_size = tokenizer_data["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(vocab_size, embedding_size=512, hidden_size=512).to(device)
    model.load_state_dict(
        torch.load("best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()
    return model, word2idx, idx2word, device


def extract_features(image, feature_extractor, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.squeeze()


# ─── Caption Generation ────────────────────────────────────────
def greedy_caption(model, features, word2idx, idx2word, device, max_len=25):
    with torch.no_grad():
        hidden, cell = model.encoder(features.unsqueeze(0).to(device))
        input_word = torch.LongTensor([[word2idx["<start>"]]]).to(device)
        words = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_word, hidden, cell)
            predicted = output.argmax(dim=2)
            idx = predicted.item()
            if idx == word2idx["<end>"]:
                break
            words.append(idx2word[idx])
            input_word = predicted
    return " ".join(words)


def beam_caption(model, features, word2idx, idx2word, device,
                 beam_width=5, max_len=25):
    with torch.no_grad():
        hidden, cell = model.encoder(features.unsqueeze(0).to(device))
        beams = [([word2idx["<start>"]], 0.0, hidden, cell)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, h, c in beams:
                if seq[-1] == word2idx["<end>"]:
                    completed.append((seq, score))
                    continue
                inp = torch.LongTensor([[seq[-1]]]).to(device)
                out, h_new, c_new = model.decoder(inp, h, c)
                log_probs = F.log_softmax(out.squeeze(1), dim=1)
                top_probs, top_ids = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    new_beams.append((
                        seq + [top_ids[0, i].item()],
                        score + top_probs[0, i].item(),
                        h_new, c_new,
                    ))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if not beams:
                break

        completed.extend([(s, sc) for s, sc, _, _ in beams])
        best = max(completed, key=lambda x: x[1])[0]
        skip = {word2idx["<start>"], word2idx["<end>"], word2idx["<pad>"]}
        return " ".join(idx2word[i] for i in best if i not in skip)


# ─── Load Everything on Startup ────────────────────────────────
with st.spinner("🔄 Loading ResNet-50 and caption model…"):
    feature_extractor = load_resnet()
    model, word2idx, idx2word, device = load_model()
    feature_extractor = feature_extractor.to(device)

# ─── Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    "<h1>📸 Image Captioner</h1>"
    "<p>Upload an image and let the AI describe it</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="status-badge status-ready">✓ Models loaded &amp; ready</span>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Two-Column Layout ─────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: Upload & Controls ────────────────────────────────────
with left_col:
    st.markdown('<div class="section-title">📤 Upload Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

    st.markdown("")  # spacing

    method = st.radio(
        "Decoding strategy",
        ["Beam Search", "Greedy"],
        index=0,
        horizontal=True,
    )

    if method == "Beam Search":
        beam_width = st.slider("Beam width", 2, 10, 5)

    st.markdown("")  # spacing
    generate_btn = st.button("✨ Generate Caption", disabled=uploaded is None)

# ── RIGHT: Caption Output ──────────────────────────────────────
with right_col:
    st.markdown('<div class="section-title">💬 Generated Caption</div>',
                unsafe_allow_html=True)

    if uploaded and generate_btn:
        with st.spinner("Generating caption…"):
            features = extract_features(image, feature_extractor, device)
            if method == "Beam Search":
                caption = beam_caption(
                    model, features, word2idx, idx2word, device, beam_width
                )
            else:
                caption = greedy_caption(
                    model, features, word2idx, idx2word, device
                )

        st.markdown(
            f'<div class="caption-card">'
            f'<div class="label">Result</div>'
            f'<div class="text">{caption}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    elif uploaded and not generate_btn:
        st.markdown(
            '<div class="placeholder-text">'
            '👈 Click <strong>Generate Caption</strong> to start'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="placeholder-text">'
            "� Upload an image first"
            "</div>",
            unsafe_allow_html=True,
        )

# ─── Footer ─────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Powered by ResNet-50 + Seq2Seq LSTM · '
    "Trained on Flickr30k</div>",
    unsafe_allow_html=True,
)
