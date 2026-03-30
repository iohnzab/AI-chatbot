from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BRAIN = "ai_brain"

# ── Load vocab ONLY from brain file (fixes size mismatch) ─────────────
vocab, word2idx, idx2word, vocab_size = [], {}, {}, 0

def load_vocab_from_brain():
    global vocab, word2idx, idx2word, vocab_size
    vocab_file = f"{BRAIN}/vocab.json"
    if os.path.exists(vocab_file):
        with open(vocab_file) as f:
            data = json.load(f)
        vocab      = data["vocab"]
        word2idx   = {w: i for i, w in enumerate(vocab)}
        idx2word   = {i: w for w, i in word2idx.items()}
        vocab_size = len(vocab)
        return True
    return False

# ── Helpers ───────────────────────────────────────────────────────────
def encode(sentence):
    return [word2idx[w] for w in sentence.lower().split() if w in word2idx]

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

# ── LSTM Cell ─────────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        n = hidden_size
        d = input_size + hidden_size
        self.Wf = np.random.randn(n, d) * 0.01; self.bf = np.zeros((n, 1))
        self.Wi = np.random.randn(n, d) * 0.01; self.bi = np.zeros((n, 1))
        self.Wg = np.random.randn(n, d) * 0.01; self.bg = np.zeros((n, 1))
        self.Wo = np.random.randn(n, d) * 0.01; self.bo = np.zeros((n, 1))

    def expand(self, new_input_size):
        diff = new_input_size - self.input_size
        if diff <= 0:
            return
        def pad(W):
            return np.hstack([W, np.random.randn(W.shape[0], diff) * 0.01])
        self.Wf = pad(self.Wf); self.Wi = pad(self.Wi)
        self.Wg = pad(self.Wg); self.Wo = pad(self.Wo)
        self.input_size = new_input_size

    def forward(self, x, h, c):
        combined = np.vstack([x, h])
        f = sigmoid(self.Wf @ combined + self.bf)
        i = sigmoid(self.Wi @ combined + self.bi)
        g = np.tanh(self.Wg  @ combined + self.bg)
        o = sigmoid(self.Wo @ combined + self.bo)
        c = f * c + i * g
        h = o * np.tanh(c)
        return h, c, (combined, f, i, g, o, c, c)

    def backward(self, dh, dc, cache, lr):
        combined, f, i, g, o, c_prev, c = cache
        do = dh * np.tanh(c)
        dc += dh * o * (1 - np.tanh(c) ** 2)
        df = dc * c_prev; di = dc * g; dg = dc * i; dc_prev = dc * f
        do_raw = do * o * (1 - o)
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dg_raw = dg * (1 - g ** 2)
        self.Wo -= lr * np.clip(do_raw @ combined.T, -5, 5)
        self.Wf -= lr * np.clip(df_raw @ combined.T, -5, 5)
        self.Wi -= lr * np.clip(di_raw @ combined.T, -5, 5)
        self.Wg -= lr * np.clip(dg_raw @ combined.T, -5, 5)
        self.bo -= lr * np.clip(do_raw, -5, 5)
        self.bf -= lr * np.clip(df_raw, -5, 5)
        self.bi -= lr * np.clip(di_raw, -5, 5)
        self.bg -= lr * np.clip(dg_raw, -5, 5)
        d_combined = (self.Wf.T @ df_raw + self.Wi.T @ di_raw +
                      self.Wg.T @ dg_raw + self.Wo.T @ do_raw)
        return d_combined[self.input_size:], dc_prev

# ── LSTM Chatbot ──────────────────────────────────────────────────────
class LSTMChatbot:
    def __init__(self, vocab_size, hidden_size=128, lr=0.005):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.lr          = lr
        self.encoder     = LSTMCell(vocab_size, hidden_size)
        self.decoder     = LSTMCell(vocab_size, hidden_size)
        self.Wy          = np.random.randn(vocab_size, hidden_size) * 0.01
        self.by          = np.zeros((vocab_size, 1))

    def load(self):
        """Load brain and use its ACTUAL sizes — ignores constructor sizes."""
        def load_cell(cell, prefix):
            for name in ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]:
                setattr(cell, name, np.load(f"{BRAIN}/{prefix}_{name}.npy"))
        load_cell(self.encoder, "enc")
        load_cell(self.decoder, "dec")
        self.Wy = np.load(f"{BRAIN}/Wy.npy")
        self.by = np.load(f"{BRAIN}/by.npy")
        # Use brain's actual sizes — this is the key fix
        self.vocab_size          = self.Wy.shape[0]
        self.hidden_size         = self.Wy.shape[1]
        self.encoder.input_size  = self.vocab_size
        self.decoder.input_size  = self.vocab_size
        self.encoder.hidden_size = self.hidden_size
        self.decoder.hidden_size = self.hidden_size

    def save(self):
        os.makedirs(BRAIN, exist_ok=True)
        def save_cell(cell, prefix):
            for name in ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]:
                np.save(f"{BRAIN}/{prefix}_{name}.npy", getattr(cell, name))
        save_cell(self.encoder, "enc")
        save_cell(self.decoder, "dec")
        np.save(f"{BRAIN}/Wy.npy", self.Wy)
        np.save(f"{BRAIN}/by.npy", self.by)
        with open(f"{BRAIN}/vocab.json", "w") as f:
            json.dump({"vocab": vocab, "word2idx": word2idx}, f)

    def expand_vocab(self, new_size):
        if new_size <= self.vocab_size:
            return
        diff = new_size - self.vocab_size
        self.encoder.expand(new_size)
        self.decoder.expand(new_size)
        self.Wy = np.vstack([self.Wy, np.random.randn(diff, self.hidden_size) * 0.01])
        self.by = np.vstack([self.by, np.zeros((diff, 1))])
        self.vocab_size = new_size

    def _encode(self, src_seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for idx in src_seq:
            if idx < self.vocab_size:
                h, c, _ = self.encoder.forward(one_hot(idx, self.vocab_size), h, c)
        return h, c

    def train_step(self, src_seq, tgt_seq):
        SOS = word2idx.get("<SOS>", 1)
        EOS = word2idx.get("<EOS>", 2)
        h, c = self._encode(src_seq)
        x_idx = SOS
        caches, hs, probs_list = [], [h], []
        target_seq = tgt_seq + [EOS]
        for target_idx in target_seq:
            x = one_hot(x_idx, self.vocab_size)
            h, c, cache = self.decoder.forward(x, h, c)
            probs = softmax((self.Wy @ h + self.by).flatten())
            caches.append(cache); hs.append(h)
            probs_list.append((probs, target_idx))
            x_idx = target_idx
        dh  = np.zeros((self.hidden_size, 1))
        dc  = np.zeros((self.hidden_size, 1))
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        for t in reversed(range(len(target_seq))):
            probs, target_idx = probs_list[t]
            dy = probs.reshape(-1, 1); dy[target_idx] -= 1
            dWy += dy @ hs[t + 1].T; dby += dy
            dh, dc = self.decoder.backward(self.Wy.T @ dy + dh, dc, caches[t], self.lr)
        self.Wy -= self.lr * np.clip(dWy, -5, 5)
        self.by -= self.lr * np.clip(dby, -5, 5)

    def respond(self, sentence, max_len=12):
        SOS = word2idx.get("<SOS>", 1)
        EOS = word2idx.get("<EOS>", 2)
        src_seq = encode(sentence)
        if not src_seq:
            return "i did not understand that"
        h, c  = self._encode(src_seq)
        reply = []
        x_idx = SOS
        seen  = []
        for _ in range(max_len):
            x = one_hot(x_idx, self.vocab_size)
            h, c, _ = self.decoder.forward(x, h, c)
            probs = softmax((self.Wy @ h + self.by).flatten())
            top3       = np.argsort(probs)[-3:]
            top3_probs = probs[top3] / probs[top3].sum()
            x_idx      = np.random.choice(top3, p=top3_probs)
            if x_idx == EOS:
                break
            word = idx2word.get(x_idx, "")
            if word in ("<PAD>", "<SOS>", "<EOS>", ""):
                break
            if word in seen[-2:]:
                break
            reply.append(word)
            seen.append(word)
        return " ".join(reply) if reply else "i am still learning"

    def learn(self, user_input, correct_reply, epochs=500):
        global vocab, word2idx, idx2word, vocab_size
        changed = False
        for w in (user_input + " " + correct_reply).lower().split():
            if w not in word2idx:
                word2idx[w] = len(vocab)
                idx2word[len(vocab)] = w
                vocab.append(w)
                changed = True
        if changed:
            vocab_size = len(vocab)
            self.expand_vocab(vocab_size)
        src_seq = encode(user_input)
        tgt_seq = encode(correct_reply)
        if src_seq and tgt_seq:
            for _ in range(epochs):
                self.train_step(src_seq, tgt_seq)
        self.save()
        pairs_file = f"{BRAIN}/learned_pairs.json"
        learned = []
        if os.path.exists(pairs_file):
            with open(pairs_file) as f:
                learned = json.load(f)
        learned.append([user_input, correct_reply])
        with open(pairs_file, "w") as f:
            json.dump(learned, f, indent=2)

# ── Boot ──────────────────────────────────────────────────────────────
brain_ok = load_vocab_from_brain()
np.random.seed(42)
model = LSTMChatbot(vocab_size=max(vocab_size, 3), hidden_size=128, lr=0.005)

if brain_ok and os.path.exists(f"{BRAIN}/enc_Wf.npy"):
    model.load()
    print(f"✅ Brain loaded! vocab={model.vocab_size} hidden={model.hidden_size}")
else:
    print("⚠️  No brain found! Run train_brain.py first.")

# ── API Routes ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class TeachRequest(BaseModel):
    user_input: str
    correct_reply: str

class FeedbackRequest(BaseModel):
    user_input: str
    ai_reply: str
    good: bool

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        reply = model.respond(req.message)
        return {"reply": reply}
    except Exception as e:
        return {"reply": "i am still learning", "error": str(e)}

@app.post("/teach")
def teach(req: TeachRequest):
    try:
        model.learn(req.user_input, req.correct_reply, epochs=500)
        return {"status": "learned"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        if req.good:
            model.learn(req.user_input, req.ai_reply, epochs=200)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/stats")
def stats():
    learned_file = f"{BRAIN}/learned_pairs.json"
    learned_count = 0
    if os.path.exists(learned_file):
        with open(learned_file) as f:
            learned_count = len(json.load(f))
    return {
        "vocab_size"   : model.vocab_size,
        "learned_pairs": learned_count,
        "brain_loaded" : os.path.exists(f"{BRAIN}/enc_Wf.npy")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)