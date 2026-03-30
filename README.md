# 🧠 Aria — AI Chatbot Built from Scratch

> A real neural network chatbot with a web interface — built from scratch using Python, LSTM, and FastAPI. No AI frameworks. Just pure math, code, and a clean browser UI.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![AI](https://img.shields.io/badge/AI-LSTM%20Neural%20Network-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 What is this?

**Aria** is a conversational AI chatbot powered by a sequence-to-sequence LSTM (Long Short-Term Memory) neural network — built entirely from scratch without PyTorch or TensorFlow.

Every part of the AI — the neurons, the gates, the backpropagation, the memory — is written in pure Python and NumPy. No black boxes. Served through a FastAPI backend with a clean, minimal web interface.

---

## ✨ Features

- 🧠 **Real neural network** — LSTM encoder-decoder architecture from scratch
- 🌐 **Web interface** — chat in your browser, no terminal needed
- 📚 **130+ training pairs** covering everyday conversation topics
- 💾 **Persistent brain** — trains once, saves to disk, loads instantly next time
- 🎓 **Live learning** — teach Aria new things during chat, she remembers permanently
- 👍 **Feedback system** — mark replies as good or correct them on the fly
- 📈 **Gets smarter over time** — every correction saved and reused in future training
- 🔤 **Dynamic vocabulary** — learns new words on the fly

---

## 🏗️ How it works

```
Your message
     ↓
Encoder LSTM  →  reads words one by one  →  builds a "thought vector"
     ↓
Decoder LSTM  →  uses thought vector  →  generates reply word by word
     ↓
Aria's reply
```

The AI learns by:
1. Making a prediction
2. Checking how wrong it was (loss)
3. Adjusting weights via backpropagation
4. Repeating thousands of times until it gets good

---

## 📁 Project Structure

```
AI-chatbot/
│
├── chatbot.py            # Terminal chat interface
├── train_brain.py        # Full brain trainer with large dataset
├── server.py             # FastAPI backend server
├── index.html            # Web chat interface
│
└── ai_brain/             # Saved neural network weights (auto-generated)
    ├── enc_Wf.npy        # Encoder forget gate
    ├── enc_Wi.npy        # Encoder input gate
    ├── enc_Wg.npy        # Encoder cell gate
    ├── enc_Wo.npy        # Encoder output gate
    ├── dec_Wf.npy        # Decoder forget gate
    ├── dec_Wi.npy        # Decoder input gate
    ├── dec_Wg.npy        # Decoder cell gate
    ├── dec_Wo.npy        # Decoder output gate
    ├── Wy.npy            # Output weights
    ├── by.npy            # Output bias
    ├── vocab.json        # All words Aria knows
    └── learned_pairs.json  # Everything you taught Aria
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
pip3 install numpy fastapi uvicorn
```

### 3. Train Aria's brain
```bash
python3 train_brain.py
```
> ⏳ Takes 10–20 minutes on first run. After that it loads instantly every time.

### 4. Start the server
```bash
python3 server.py
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 💬 Using the web interface

```
Type a message → press Enter → Aria replies

After each reply:
  👍 "Good reply"  → Aria reinforces what she said
  ✏️  "Correct it" → You type the right answer → Aria learns permanently
  ⏭️  "Skip"       → Move on without feedback
```

### Teaching Aria directly
Click **+ Teach** anytime:
```
Question : what is python
Answer   : python is a popular programming language
→ 🧠 Aria learns it and saves it permanently
```

---

## 🖥️ Terminal mode (optional)

Prefer the terminal? Run the original chatbot directly:
```bash
python3 chatbot.py
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
| Vocabulary | 430+ words (grows as you teach it) |
| Activation | Sigmoid + Tanh |
| Optimizer | SGD with gradient clipping |
| Loss function | Cross-entropy |
| Framework | None — pure NumPy only |

---

## 📊 Training

| Setting | Value |
|---|---|
| Base training pairs | 140 |
| Epochs | 500 (configurable) |
| Learning rate | 0.005 |
| Gradient clipping | ±5 |
| Training time | ~10–20 min on CPU |

---

## 🛣️ Roadmap

- [x] LSTM chatbot from scratch (terminal)
- [x] Persistent brain saving and loading
- [x] Live learning from chat corrections
- [x] Dynamic vocabulary expansion
- [x] FastAPI web server
- [x] Clean web chat interface
- [x] Feedback system (good reply / correct it)
- [ ] Typo tolerance
- [ ] Remember user name and preferences
- [ ] Upgrade to Transformer architecture
- [ ] Deploy online (Railway / Render)
- [ ] Voice input and output

---

## 🤝 Why I built this

I wanted to understand how AI actually works under the hood — not just call an API, but build the real math from scratch. Every weight, every gate, every backpropagation step is written by hand in this project.

This is the same core technology behind early Siri, Google Translate, and pre-GPT chatbots — just at a much smaller scale.

---

## 📚 What I learned

- How LSTM gates (forget, input, cell, output) work mathematically
- How backpropagation through time (BPTT) works
- How sequence-to-sequence models encode meaning into vectors
- How neural networks learn from mistakes and adjust weights
- How to save and load neural network weights
- How to serve an AI model via REST API using FastAPI
- How to build a real-time chat UI with HTML, CSS, and JavaScript

---

## 🛠️ Built with

- **Python 3.9+**
- **NumPy** — all matrix math and neural network operations
- **FastAPI** — REST API backend
- **Uvicorn** — ASGI server
- **HTML / CSS / JS** — frontend chat interface
- No ML frameworks — no PyTorch, no TensorFlow, no scikit-learn

---

## 📄 License

MIT License — feel free to use, modify, and learn from this code.

---

<p align="center">Built from scratch with 🧠 math and ☕ coffee</p>
