import numpy as np
import os
import json
import random

# ══════════════════════════════════════════════════════════════════════
#  MASSIVE TRAINING DATASET — covers everyday conversations
# ══════════════════════════════════════════════════════════════════════
pairs = [
    # ── Greetings ─────────────────────────────────────────────────────
    ("hello", "hi there how are you"),
    ("hi", "hello how can i help you"),
    ("hey", "hey there good to see you"),
    ("good morning", "good morning hope you have a great day"),
    ("good afternoon", "good afternoon how are you doing"),
    ("good evening", "good evening hope your day went well"),
    ("good night", "good night sleep well"),
    ("howdy", "howdy how are you doing today"),
    ("what is up", "not much just here to chat with you"),
    ("how are you", "i am doing great thanks for asking"),
    ("how are you doing", "i am doing well how about you"),
    ("how have you been", "i have been good thanks for asking"),
    ("how is it going", "it is going well thanks"),
    ("are you there", "yes i am here and ready to chat"),
    ("you there", "yes i am here how can i help"),

    # ── Identity ──────────────────────────────────────────────────────
    ("what is your name", "my name is aria your personal ai"),
    ("who are you", "i am aria an ai you built using python"),
    ("what are you", "i am an artificial intelligence chatbot"),
    ("are you a robot", "i am an ai not exactly a robot but close"),
    ("are you human", "no i am an ai but i try my best to be helpful"),
    ("are you real", "i am real in the sense that i exist as an ai"),
    ("are you alive", "i am not alive but i am always here for you"),
    ("who created you", "you created me using python and lstm"),
    ("who made you", "you built me from scratch using python"),
    ("what are you made of", "i am made of code math and a lot of training"),

    # ── Capabilities ──────────────────────────────────────────────────
    ("what can you do", "i can chat answer questions and learn from you"),
    ("what do you know", "i know what you have taught me so far"),
    ("can you help me", "of course i will do my best to help you"),
    ("can you learn", "yes i learn every time you teach me something"),
    ("can you remember", "yes i remember everything saved in my brain"),
    ("do you understand me", "i try my best to understand you"),
    ("can you think", "i process information and generate responses"),
    ("can you talk", "i can chat with you through text"),
    ("can you search the web", "no i cannot access the internet"),
    ("can you do math", "i can do basic math if you teach me"),

    # ── Feelings & Emotions ───────────────────────────────────────────
    ("i am happy", "that is wonderful i am glad to hear that"),
    ("i am sad", "i am sorry to hear that i hope you feel better soon"),
    ("i am angry", "i understand take a deep breath it will be okay"),
    ("i am tired", "you should rest and take care of yourself"),
    ("i am bored", "let us have a fun conversation then"),
    ("i am excited", "that is awesome what are you excited about"),
    ("i am scared", "it is okay to feel scared i am here for you"),
    ("i am stressed", "try to relax and take things one step at a time"),
    ("i am lonely", "i am here to keep you company"),
    ("i am fine", "glad to hear you are doing fine"),
    ("i feel good", "that is great keep that positive energy"),
    ("i feel bad", "i am sorry to hear that what is wrong"),
    ("i love you", "that is very kind of you i appreciate it"),
    ("i hate you", "i am sorry to hear that i will try to do better"),
    ("you are great", "thank you that really means a lot to me"),
    ("you are smart", "thank you i am always trying to learn more"),
    ("you are funny", "i am glad i could make you smile"),
    ("you are helpful", "thank you i always try my best to help"),
    ("you are bad", "i am sorry i will try to improve"),
    ("you are amazing", "thank you you are amazing too"),

    # ── Small Talk ────────────────────────────────────────────────────
    ("what is new", "not much just learning new things every day"),
    ("tell me something", "did you know that honey never expires"),
    ("tell me a fact", "the human brain has about 86 billion neurons"),
    ("tell me a joke", "why do programmers prefer dark mode because light attracts bugs"),
    ("say something funny", "i told my computer i needed a break now it will not stop sending me kit kat ads"),
    ("make me laugh", "why did the scarecrow win an award because he was outstanding in his field"),
    ("what is the meaning of life", "that is a big question many say it is 42"),
    ("do you like music", "i cannot hear music but i think it sounds wonderful"),
    ("do you like movies", "i cannot watch movies but i love hearing about them"),
    ("do you like food", "i cannot eat but i hear pizza is amazing"),
    ("what is your favorite color", "i think i like blue it feels calm and cool"),
    ("what is your favorite food", "i cannot eat but sushi sounds delicious"),
    ("do you have friends", "you are my friend and i value that"),
    ("are you happy", "i feel happy when i am able to help you"),
    ("do you sleep", "no i do not sleep i am always ready to chat"),
    ("do you dream", "i do not dream but i think about our conversations"),
    ("what do you do for fun", "i enjoy learning new things and chatting with you"),

    # ── Basic Knowledge ───────────────────────────────────────────────
    ("what is python", "python is a popular programming language used for many things"),
    ("what is ai", "ai stands for artificial intelligence machines that can learn"),
    ("what is machine learning", "machine learning is when computers learn from data"),
    ("what is a computer", "a computer is a machine that processes information"),
    ("what is the internet", "the internet is a global network connecting computers worldwide"),
    ("what is coding", "coding is writing instructions for computers to follow"),
    ("what is an algorithm", "an algorithm is a set of steps to solve a problem"),
    ("what is data", "data is information collected and used for analysis"),
    ("what is a neural network", "a neural network is a system inspired by the human brain"),
    ("what is deep learning", "deep learning uses many layers of neural networks to learn"),
    ("what is a robot", "a robot is a machine that can perform tasks automatically"),
    ("what is software", "software is programs and code that run on computers"),
    ("what is hardware", "hardware is the physical parts of a computer"),

    # ── General Knowledge ─────────────────────────────────────────────
    ("what is the sun", "the sun is a star at the center of our solar system"),
    ("what is the moon", "the moon is earths natural satellite"),
    ("what is earth", "earth is the planet we live on"),
    ("how many planets are there", "there are eight planets in our solar system"),
    ("what is gravity", "gravity is the force that pulls objects toward each other"),
    ("what is water", "water is a liquid made of hydrogen and oxygen"),
    ("what is fire", "fire is a chemical reaction that produces heat and light"),
    ("what is the sky", "the sky appears blue because of how light scatters in the atmosphere"),
    ("why is the sky blue", "the sky is blue because air scatters blue light more than other colors"),
    ("what is rain", "rain is water droplets that fall from clouds"),
    ("what is snow", "snow is frozen water crystals that fall from the sky"),
    ("what is wind", "wind is moving air caused by differences in air pressure"),
    ("how big is the universe", "the universe is incredibly vast and possibly infinite"),
    ("what is a black hole", "a black hole is a region in space with gravity so strong nothing can escape"),
    ("what is time", "time is the progression of events from past to present to future"),

    # ── Math ──────────────────────────────────────────────────────────
    ("what is one plus one", "one plus one equals two"),
    ("what is two plus two", "two plus two equals four"),
    ("what is ten minus five", "ten minus five equals five"),
    ("what is five times five", "five times five equals twenty five"),
    ("what is ten divided by two", "ten divided by two equals five"),
    ("what is one hundred plus one hundred", "one hundred plus one hundred equals two hundred"),
    ("what is the square root of nine", "the square root of nine is three"),
    ("what is pi", "pi is approximately three point one four one five nine"),

    # ── Polite Phrases ────────────────────────────────────────────────
    ("thanks", "you are very welcome"),
    ("thank you", "no problem at all happy to help"),
    ("thank you so much", "it is my pleasure glad i could help"),
    ("please", "of course i am happy to help"),
    ("sorry", "no worries it is all good"),
    ("i am sorry", "no need to apologize we are all good"),
    ("excuse me", "yes how can i help you"),
    ("you are welcome", "thank you for being so kind"),
    ("no problem", "glad to hear it let me know if you need anything"),
    ("i appreciate it", "it is my pleasure i am always here to help"),

    # ── Farewells ─────────────────────────────────────────────────────
    ("goodbye", "goodbye take care and have a wonderful day"),
    ("bye", "bye see you next time"),
    ("see you later", "see you later take care"),
    ("see you", "see you soon take care"),
    ("talk to you later", "sounds good talk to you later"),
    ("i have to go", "okay take care come back anytime"),
    ("i am leaving", "okay goodbye hope to chat again soon"),
    ("catch you later", "catch you later take care"),
    ("take care", "you too take care and stay safe"),
    ("farewell", "farewell until we meet again"),

    # ── Questions about time ───────────────────────────────────────────
    ("what time is it", "i do not have access to a clock sorry"),
    ("what day is it", "i do not have access to a calendar"),
    ("what year is it", "i do not have real time access to dates"),
    ("what is today", "i am not connected to a clock or calendar"),

    # ── Help & Support ────────────────────────────────────────────────
    ("help", "of course what do you need help with"),
    ("i need help", "i am here to help what is going on"),
    ("i have a problem", "tell me about it and i will try to help"),
    ("i am confused", "that is okay let me try to help clarify"),
    ("i do not understand", "let me try to explain it differently"),
    ("can you explain", "of course i will do my best to explain"),
    ("what does that mean", "let me try to explain what that means"),
    ("i am stuck", "let us figure it out together"),
]

print(f"📚 Total training pairs: {len(pairs)}\n")

# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════
PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"

all_words = set()
for src, tgt in pairs:
    all_words.update(src.lower().split())
    all_words.update(tgt.lower().split())

vocab    = [PAD, SOS, EOS] + sorted(all_words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"✅ Vocabulary size: {vocab_size} words\n")

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def encode(sentence):
    return [word2idx.get(w, 0) for w in sentence.lower().split()]

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

# ══════════════════════════════════════════════════════════════════════
#  LSTM CELL
# ══════════════════════════════════════════════════════════════════════
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        scale = 0.01
        n = hidden_size
        d = input_size + hidden_size

        self.Wf = np.random.randn(n, d) * scale; self.bf = np.zeros((n, 1))
        self.Wi = np.random.randn(n, d) * scale; self.bi = np.zeros((n, 1))
        self.Wg = np.random.randn(n, d) * scale; self.bg = np.zeros((n, 1))
        self.Wo = np.random.randn(n, d) * scale; self.bo = np.zeros((n, 1))

    def expand(self, new_input_size):
        diff = new_input_size - self.input_size
        if diff <= 0:
            return
        def pad(W):
            return np.hstack([W, np.random.randn(W.shape[0], diff) * 0.01])
        self.Wf = pad(self.Wf); self.Wi = pad(self.Wi)
        self.Wg = pad(self.Wg); self.Wo = pad(self.Wo)
        self.input_size = new_input_size

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([x, h_prev])
        f = sigmoid(self.Wf @ combined + self.bf)
        i = sigmoid(self.Wi @ combined + self.bi)
        g = np.tanh(self.Wg  @ combined + self.bg)
        o = sigmoid(self.Wo @ combined + self.bo)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c, (combined, f, i, g, o, c_prev, c)

    def backward(self, dh, dc, cache, lr):
        combined, f, i, g, o, c_prev, c = cache
        do = dh * np.tanh(c)
        dc += dh * o * (1 - np.tanh(c) ** 2)
        df = dc * c_prev; di = dc * g
        dg = dc * i;      dc_prev = dc * f
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

# ══════════════════════════════════════════════════════════════════════
#  LSTM CHATBOT MODEL
# ══════════════════════════════════════════════════════════════════════
class LSTMChatbot:
    def __init__(self, vocab_size, hidden_size=256, lr=0.005):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.lr          = lr
        self.encoder     = LSTMCell(vocab_size, hidden_size)
        self.decoder     = LSTMCell(vocab_size, hidden_size)
        self.Wy          = np.random.randn(vocab_size, hidden_size) * 0.01
        self.by          = np.zeros((vocab_size, 1))

    def expand_vocab(self, new_vocab_size):
        if new_vocab_size <= self.vocab_size:
            return
        diff = new_vocab_size - self.vocab_size
        self.encoder.expand(new_vocab_size)
        self.decoder.expand(new_vocab_size)
        self.Wy = np.vstack([self.Wy, np.random.randn(diff, self.hidden_size) * 0.01])
        self.by = np.vstack([self.by, np.zeros((diff, 1))])
        self.vocab_size = new_vocab_size

    def _encode(self, src_seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for idx in src_seq:
            if idx < self.vocab_size:
                h, c, _ = self.encoder.forward(one_hot(idx, self.vocab_size), h, c)
        return h, c

    def train_step(self, src_seq, tgt_seq):
        h, c       = self._encode(src_seq)
        x_idx      = word2idx[SOS]
        caches, hs, probs_list = [], [h], []
        target_seq = tgt_seq + [word2idx[EOS]]

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

    def train(self, pairs, epochs, label="Training"):
        total = epochs * len(pairs)
        done  = 0
        print(f"🔄 {label}")
        print(f"   {len(pairs)} pairs × {epochs} epochs = {total:,} total steps\n")

        for epoch in range(1, epochs + 1):
            shuffled = pairs[:]
            random.shuffle(shuffled)
            for src, tgt in shuffled:
                src_seq = encode(src)
                tgt_seq = encode(tgt)
                if src_seq and tgt_seq:
                    self.train_step(src_seq, tgt_seq)
                done += 1

            # Progress bar — updates every 10 epochs
            if epoch % 10 == 0:
                pct = epoch / epochs * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"  [{bar}] {pct:.1f}%  Epoch {epoch}/{epochs}", end="\r")

        print(f"\n✅ {label} complete!\n")

    def respond(self, sentence, max_len=12):
        src_seq = encode(sentence)
        if not src_seq:
            return "i did not understand that"
        h, c  = self._encode(src_seq)
        reply = []
        x_idx = word2idx[SOS]
        seen  = []
        for _ in range(max_len):
            x = one_hot(x_idx, self.vocab_size)
            h, c, _ = self.decoder.forward(x, h, c)
            probs = softmax((self.Wy @ h + self.by).flatten())
            top3       = np.argsort(probs)[-3:]
            top3_probs = probs[top3] / probs[top3].sum()
            x_idx      = np.random.choice(top3, p=top3_probs)
            if x_idx == word2idx[EOS]:
                break
            word = idx2word.get(x_idx, "")
            if word in (PAD, SOS, EOS, ""):
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
            json.dump({"vocab": vocab, "word2idx": word2idx}, f)
        print(f"💾 Brain saved to '{folder}/'\n")

# ══════════════════════════════════════════════════════════════════════
#  MAIN — DELETE OLD BRAIN AND RETRAIN
# ══════════════════════════════════════════════════════════════════════
import shutil

BRAIN = "ai_brain"

print("=" * 50)
print("  🧠 AI BRAIN TRAINER")
print("  This will delete your old brain and")
print("  train a smarter one from scratch.")
print("=" * 50 + "\n")

confirm = input("Continue? (yes/no): ").strip().lower()
if confirm != "yes":
    print("Cancelled.")
    exit()

# Backup learned pairs before deleting
learned_pairs = []
if os.path.exists(f"{BRAIN}/learned_pairs.json"):
    with open(f"{BRAIN}/learned_pairs.json") as f:
        learned_pairs = json.load(f)
    print(f"📝 Found {len(learned_pairs)} previously learned pairs — keeping them!\n")

# Delete old brain
if os.path.exists(BRAIN):
    shutil.rmtree(BRAIN)
    print("🗑️  Old brain deleted\n")

# Combine base pairs + previously learned pairs
all_pairs = pairs[:]
for item in learned_pairs:
    if isinstance(item, list) and len(item) == 2:
        all_pairs.append((item[0], item[1]))

print(f"📚 Total pairs to train on: {len(all_pairs)}\n")

# Train
np.random.seed(42)
model = LSTMChatbot(vocab_size=vocab_size, hidden_size=64, lr=0.01)
model.train(all_pairs, epochs=300, label="Training smart brain")

# Save
model.save(BRAIN)

# Restore learned pairs file
if learned_pairs:
    with open(f"{BRAIN}/learned_pairs.json", "w") as f:
        json.dump(learned_pairs, f, indent=2)
    print(f"✅ Restored {len(learned_pairs)} previously learned pairs\n")

# Quick test
print("🧪 Quick test:\n")
tests = ["hello", "how are you", "what is python", "tell me a joke", "goodbye"]
for t in tests:
    print(f"  You: {t}")
    print(f"  AI : {model.respond(t)}\n")

print("🎉 Done! Now run chatbot.py to chat with your smarter AI!")
