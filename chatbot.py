import numpy as np
import os
import json
import random

# ── 1. TRAINING DATA ──────────────────────────────────────────────────
pairs = [
    ("hello", "hi there"),
    ("hi", "hello how are you"),
    ("hey", "hey there how are you"),
    ("how are you", "i am doing well thanks"),
    ("what is your name", "i am a simple ai chatbot"),
    ("who are you", "i am an ai you created"),
    ("what can you do", "i can chat and learn from examples"),
    ("goodbye", "bye see you later"),
    ("bye", "goodbye take care"),
    ("thanks", "you are welcome"),
    ("thank you", "no problem at all"),
    ("what is ai", "ai stands for artificial intelligence"),
    ("are you smart", "i am learning every time we talk"),
    ("how old are you", "i was just created today"),
    ("do you like humans", "yes i was made to help humans"),
    ("what do you like", "i like learning new things"),
    ("tell me a joke", "why did the chicken cross the road"),
    ("i am sad", "i am sorry to hear that"),
    ("i am happy", "that is great to hear"),
    ("what time is it", "i do not have access to time"),
    ("who made you", "you created me using python"),
    ("can you learn", "yes i learn from the examples you give me"),
    ("you are good", "thank you that means a lot"),
    ("i like you", "i like you too"),
]

# ── 2. BUILD VOCABULARY ───────────────────────────────────────────────
PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"

all_words = set()
for src, tgt in pairs:
    all_words.update(src.lower().split())
    all_words.update(tgt.lower().split())

vocab    = [PAD, SOS, EOS] + sorted(all_words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

BRAIN = "ai_brain"

# ── 3. HELPERS ────────────────────────────────────────────────────────
def encode(sentence):
    tokens = []
    for w in sentence.lower().split():
        if w in word2idx:
            tokens.append(word2idx[w])
        # Unknown words are skipped — handled gracefully
    return tokens if tokens else [0]

def add_to_vocab(sentence):
    """Add new words from sentence into vocabulary."""
    global vocab, word2idx, idx2word, vocab_size
    changed = False
    for w in sentence.lower().split():
        if w not in word2idx:
            word2idx[w] = len(vocab)
            idx2word[len(vocab)] = w
            vocab.append(w)
            changed = True
    if changed:
        vocab_size = len(vocab)
    return changed

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

def save_pairs(new_pair):
    """Save new learned pairs to a file so they persist."""
    pairs_file = f"{BRAIN}/learned_pairs.json"
    learned = []
    if os.path.exists(pairs_file):
        with open(pairs_file, "r") as f:
            learned = json.load(f)
    learned.append(new_pair)
    with open(pairs_file, "w") as f:
        json.dump(learned, f, indent=2)

def load_learned_pairs():
    """Load previously learned pairs from chat."""
    pairs_file = f"{BRAIN}/learned_pairs.json"
    if os.path.exists(pairs_file):
        with open(pairs_file, "r") as f:
            return json.load(f)
    return []

# ── 4. LSTM CELL ──────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        scale = 0.01

        def init(rows, cols):
            return np.random.randn(rows, cols) * scale

        n = hidden_size
        d = input_size + hidden_size

        self.Wf = init(n, d);  self.bf = np.zeros((n, 1))
        self.Wi = init(n, d);  self.bi = np.zeros((n, 1))
        self.Wg = init(n, d);  self.bg = np.zeros((n, 1))
        self.Wo = init(n, d);  self.bo = np.zeros((n, 1))

    def expand(self, new_input_size):
        """Expand weights when new words are added to vocabulary."""
        old = self.input_size
        diff = new_input_size - old
        if diff <= 0:
            return

        def pad(W):
            return np.hstack([W, np.random.randn(W.shape[0], diff) * 0.01])

        self.Wf = pad(self.Wf)
        self.Wi = pad(self.Wi)
        self.Wg = pad(self.Wg)
        self.Wo = pad(self.Wo)
        self.input_size = new_input_size

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([x, h_prev])
        f = sigmoid(self.Wf @ combined + self.bf)
        i = sigmoid(self.Wi @ combined + self.bi)
        g = np.tanh(self.Wg @ combined + self.bg)
        o = sigmoid(self.Wo @ combined + self.bo)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c, (combined, f, i, g, o, c_prev, c)

    def backward(self, dh, dc, cache, lr):
        combined, f, i, g, o, c_prev, c = cache
        do = dh * np.tanh(c)
        dc += dh * o * (1 - np.tanh(c) ** 2)
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f
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
        dh_prev = d_combined[self.input_size:]
        return dh_prev, dc_prev

# ── 5. LSTM CHATBOT ───────────────────────────────────────────────────
class LSTMChatbot:
    def __init__(self, vocab_size, hidden_size=128, lr=0.005):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.lr          = lr
        self.encoder     = LSTMCell(vocab_size, hidden_size)
        self.decoder     = LSTMCell(vocab_size, hidden_size)
        self.Wy          = np.random.randn(vocab_size, hidden_size) * 0.01
        self.by          = np.zeros((vocab_size, 1))

    def expand_vocab(self, new_vocab_size):
        """Grow the network when new words are learned."""
        if new_vocab_size <= self.vocab_size:
            return
        diff = new_vocab_size - self.vocab_size
        self.encoder.expand(new_vocab_size)
        self.decoder.expand(new_vocab_size)
        # Expand output weights too
        new_rows = np.random.randn(diff, self.hidden_size) * 0.01
        self.Wy  = np.vstack([self.Wy, new_rows])
        self.by  = np.vstack([self.by, np.zeros((diff, 1))])
        self.vocab_size = new_vocab_size

    def encode(self, src_seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for idx in src_seq:
            if idx >= self.vocab_size:
                continue
            x = one_hot(idx, self.vocab_size)
            h, c, _ = self.encoder.forward(x, h, c)
        return h, c

    def train_step(self, src_seq, tgt_seq):
        h, c = self.encode(src_seq)
        x_idx      = word2idx[SOS]
        caches     = []
        hs         = [h]
        probs_list = []
        target_seq = tgt_seq + [word2idx[EOS]]

        for target_idx in target_seq:
            x = one_hot(x_idx, self.vocab_size)
            h, c, cache = self.decoder.forward(x, h, c)
            y     = self.Wy @ h + self.by
            probs = softmax(y.flatten())
            caches.append(cache)
            hs.append(h)
            probs_list.append((probs, target_idx))
            x_idx = target_idx

        dh  = np.zeros((self.hidden_size, 1))
        dc  = np.zeros((self.hidden_size, 1))
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        for t in reversed(range(len(target_seq))):
            probs, target_idx = probs_list[t]
            dy = probs.reshape(-1, 1)
            dy[target_idx] -= 1
            dWy    += dy @ hs[t + 1].T
            dby    += dy
            dh_out  = self.Wy.T @ dy + dh
            dh, dc  = self.decoder.backward(dh_out, dc, caches[t], self.lr)

        self.Wy -= self.lr * np.clip(dWy, -5, 5)
        self.by -= self.lr * np.clip(dby, -5, 5)

    def train(self, pairs, epochs=1500, label="Training"):
        print(f"🔄 {label}...\n")
        for epoch in range(1, epochs + 1):
            shuffled = pairs[:]
            random.shuffle(shuffled)
            for src, tgt in shuffled:
                src_seq = encode(src)
                tgt_seq = encode(tgt)
                if src_seq and tgt_seq:
                    self.train_step(src_seq, tgt_seq)
            if epoch % 300 == 0:
                print(f"  Epoch {epoch:4d}/{epochs}")
        print(f"\n✅ {label} complete!\n")

    def learn_from_chat(self, user_input, correct_reply, epochs=800):
        """Learn a new pair from live conversation."""
        # Add new words to vocabulary if needed
        vocab_changed = add_to_vocab(user_input) or add_to_vocab(correct_reply)
        if vocab_changed:
            self.expand_vocab(vocab_size)

        src_seq = encode(user_input)
        tgt_seq = encode(correct_reply)

        if not src_seq or not tgt_seq:
            return

        print(f"  🧠 Learning: '{user_input}' → '{correct_reply}'")
        for _ in range(epochs):
            self.train_step(src_seq, tgt_seq)

        # Save the new pair so it's included in future retraining
        save_pairs([user_input, correct_reply])
        self.save(BRAIN)
        print(f"  💾 Brain updated and saved!\n")

    def respond(self, sentence, max_len=12):
        src_seq = encode(sentence)
        if not src_seq:
            return "i did not understand that"

        h, c  = self.encode(src_seq)
        reply = []
        x_idx = word2idx[SOS]
        seen  = []

        for _ in range(max_len):
            x = one_hot(x_idx, self.vocab_size)
            h, c, _ = self.decoder.forward(x, h, c)
            y     = self.Wy @ h + self.by
            probs = softmax(y.flatten())

            top3       = np.argsort(probs)[-3:]
            top3_probs = probs[top3]
            top3_probs /= top3_probs.sum()
            x_idx      = np.random.choice(top3, p=top3_probs)

            if x_idx == word2idx[EOS]:
                break

            word = idx2word.get(x_idx, "")
            if word in (PAD, SOS, EOS) or not word:
                break
            if word in seen[-2:]:
                break

            reply.append(word)
            seen.append(word)

        return " ".join(reply) if reply else "i am still learning"

    def save(self, folder="ai_brain"):
        os.makedirs(folder, exist_ok=True)

        def save_cell(cell, prefix):
            for name in ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]:
                np.save(f"{folder}/{prefix}_{name}.npy", getattr(cell, name))

        save_cell(self.encoder, "enc")
        save_cell(self.decoder, "dec")
        np.save(f"{folder}/Wy.npy", self.Wy)
        np.save(f"{folder}/by.npy", self.by)

        with open(f"{folder}/vocab.json", "w") as f:
            json.dump({
                "vocab"   : vocab,
                "word2idx": word2idx
            }, f)

    def load(self, folder="ai_brain"):
        def load_cell(cell, prefix):
            for name in ["Wf","Wi","Wg","Wo","bf","bi","bg","bo"]:
                path = f"{folder}/{prefix}_{name}.npy"
                setattr(cell, name, np.load(path))
            cell.input_size = self.vocab_size

        load_cell(self.encoder, "enc")
        load_cell(self.decoder, "dec")
        self.Wy = np.load(f"{folder}/Wy.npy")
        self.by = np.load(f"{folder}/by.npy")
        self.vocab_size = self.Wy.shape[0]
        print("🧠 Brain loaded — no retraining needed!\n")


# ── 6. LOAD VOCAB FROM BRAIN IF EXISTS ───────────────────────────────
if os.path.exists(f"{BRAIN}/vocab.json"):
    with open(f"{BRAIN}/vocab.json", "r") as f:
        data     = json.load(f)
        vocab    = data["vocab"]
        word2idx = {w: i for i, w in enumerate(vocab)}
        idx2word = {i: w for w, i in word2idx.items()}
        vocab_size = len(vocab)

# ── 7. LOAD OR TRAIN ──────────────────────────────────────────────────
np.random.seed(42)
model = LSTMChatbot(vocab_size=vocab_size, hidden_size=128, lr=0.005)

if os.path.exists(BRAIN) and os.path.exists(f"{BRAIN}/enc_Wf.npy"):
    model.load(BRAIN)

    # Also retrain on previously learned chat pairs
    learned = load_learned_pairs()
    if learned:
        print(f"📚 Relearning {len(learned)} previously taught pairs...\n")
        model.train(learned, epochs=500, label="Refreshing memory")
else:
    import shutil
    if os.path.exists(BRAIN):
        shutil.rmtree(BRAIN)

    all_pairs = pairs[:]
    model.train(all_pairs, epochs=1500, label="Initial training")
    model.save(BRAIN)
    print("💾 Brain saved!\n")

# ── 8. CHAT LOOP ──────────────────────────────────────────────────────
print("=" * 45)
print("💬 Chat with your AI!")
print("   Type 'quit' to exit")
print("   Type 'teach' to teach it something new")
print("=" * 45 + "\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("AI: goodbye take care")
        break

    # Teach mode — you correct the AI manually
    if user_input.lower() == "teach":
        question = input("  Teach question : ").strip()
        answer   = input("  Teach answer   : ").strip()
        if question and answer:
            model.learn_from_chat(question, answer)
        continue

    # Normal chat
    reply = model.respond(user_input)
    print(f"AI: {reply}\n")

    # Ask if the reply was good
    feedback = input("  Good reply? (y/n/skip): ").strip().lower()

    if feedback == "n":
        correct = input("  What should I have said? : ").strip()
        if correct:
            model.learn_from_chat(user_input, correct)
    elif feedback == "y":
        # Reinforce the good reply by training on it again
        model.learn_from_chat(user_input, reply, epochs=300)
        print("  ✅ Got it! I'll remember that.\n")
    else:
        print()