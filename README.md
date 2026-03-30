# 🧠 Aria — AI Chatbot Built from Scratch

> A real neural network chatbot built from scratch using Python and LSTM — no frameworks, no APIs, just pure math and code.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy)
![AI](https://img.shields.io/badge/AI-LSTM%20Neural%20Network-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 What is this?

**Aria** is a conversational AI chatbot powered by a sequence-to-sequence LSTM (Long Short-Term Memory) neural network — built entirely from scratch without any AI frameworks like PyTorch or TensorFlow.

This means every part of the AI — the neurons, the gates, the learning algorithm, the memory — is written in pure Python and NumPy. No black boxes.

---

## ✨ Features

- 🧠 **Real neural network** — LSTM encoder-decoder architecture from scratch
- 📚 **130+ training pairs** covering everyday conversation topics
- 💾 **Persistent brain** — trains once, saves weights to disk, loads instantly next time
- 🎓 **Live learning** — teach Aria new things during chat, it remembers permanently
- 📈 **Gets smarter over time** — every correction is saved and reused in future training
- 🔤 **Dynamic vocabulary** — learns new words you teach it on the fly

---

## 🏗️ How it works

```
Your message
     ↓
Encoder LSTM  →  reads your words one by one  →  builds a "thought vector"
     ↓
Decoder LSTM  →  uses thought vector  →  generates reply word by word
     ↓
Aria's reply
```

The AI learns by:
1. Making a prediction
2. Checking how wrong it was (loss)
3. Adjusting its weights via backpropagation
4. Repeating thousands of times until it gets good

---

## 📁 Project Structure

```
AI-chatbot/
│
├── chatbot.py          # Main chat interface with live learning
├── train_brain.py      # Full brain trainer with large dataset
│
└── ai_brain/           # Saved neural network weights (auto-generated)
    ├── enc_Wf.npy      # Encoder forget gate
    ├── enc_Wi.npy      # Encoder input gate
    ├── enc_Wg.npy      # Encoder cell gate
    ├── enc_Wo.npy      # Encoder output gate
    ├── dec_Wf.npy      # Decoder forget gate
    ├── dec_Wi.npy      # Decoder input gate
    ├── dec_Wg.npy      # Decoder cell gate
    ├── dec_Wo.npy      # Decoder output gate
    ├── Wy.npy          # Output weights
    ├── by.npy          # Output bias
    ├── vocab.json      # Vocabulary of all known words
    └── learned_pairs.json  # Everything Aria learned from you
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/AI-chatbot.git
cd AI-chatbot
```

### 2. Install dependencies
```bash
pip3 install numpy
```

### 3. Train Aria's brain
```bash
python3 train_brain.py
```
> ⏳ This takes 10-20 minutes on first run. After that it loads instantly every time.

### 4. Chat with Aria
```bash
python3 chatbot.py
```

---

## 💬 How to use

```
💬 Chat with your AI (type 'quit' to exit)

You: hello
AI: hi there how are you

  Good reply? (y/n/skip): y
  ✅ Got it! I'll remember that.

You: what is quantum physics
AI: i am still learning

  Good reply? (y/n/skip): n
  What should I have said?: quantum physics studies matter at the smallest scales
  🧠 Learning: 'what is quantum physics' → 'quantum physics studies matter at the smallest scales'
  💾 Brain updated and saved!
```

### Teaching mode
Type `teach` anytime to add something new directly:
```
You: teach
  Teach question : what is your favorite movie
  Teach answer   : i cannot watch movies but i hear inception is great
  🧠 Learning... 💾 Saved!
```

---

## 🧠 What Aria knows out of the box

| Category | Examples |
|---|---|
| Greetings | hello, hi, good morning, good night |
| Identity | who are you, are you human, who made you |
| Emotions | i am sad, i am happy, i am stressed |
| Small talk | tell me a joke, tell me a fact |
| Basic knowledge | what is python, what is ai, what is gravity |
| Math | what is two plus two, what is pi |
| Polite phrases | thanks, sorry, i appreciate it |
| Farewells | goodbye, bye, see you later |

---

## ⚙️ Model Architecture

| Component | Details |
|---|---|
| Architecture | Sequence-to-Sequence LSTM |
| Encoder | 1-layer LSTM with 4 gates |
| Decoder | 1-layer LSTM with 4 gates |
| Hidden size | 128 neurons |
| Vocabulary | 433+ words (grows as you teach it) |
| Activation | Sigmoid + Tanh |
| Optimizer | SGD with gradient clipping |
| Loss | Cross-entropy |

---

## 📊 Training

| Setting | Value |
|---|---|
| Training pairs | 140 base pairs |
| Epochs | 500 |
| Learning rate | 0.005 |
| Gradient clip | ±5 |

---

## 🛣️ Roadmap

- [x] Basic LSTM chatbot
- [x] Persistent brain saving/loading
- [x] Live learning from chat
- [x] Dynamic vocabulary expansion
- [ ] Typo tolerance
- [ ] Personal memory (remembers your name)
- [ ] Upgrade to Transformer architecture
- [ ] Web interface
- [ ] Voice input/output

---

## 🤝 Why I built this

I wanted to understand how AI actually works under the hood — not just call an API, but build the real math from scratch. Every weight, every gate, every backpropagation step is written by hand in this project.

This is the same core technology behind real AI systems like early Siri, Google Translate, and pre-GPT chatbots — just at a smaller scale.

---

## 📚 What I learned

- How LSTM gates (forget, input, cell, output) work mathematically
- How backpropagation flows through time (BPTT)
- How sequence-to-sequence models encode meaning into vectors
- How neural networks learn from mistakes and adjust weights
- How to save and load neural network state

---

## 🛠️ Built with

- **Python 3.9+**
- **NumPy** — all matrix math and neural network operations
- No other dependencies!

---

## 📄 License

MIT License — feel free to use, modify, and learn from this code.

---

<p align="center">Built from scratch with 🧠 math and ☕ coffee</p>
