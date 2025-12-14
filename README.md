# 🏛️ Politixpert – AI Political Analyst

**Politixpert** is a **Retrieval-Augmented Generation (RAG)** application designed to analyze and summarize the positions of **Moroccan political parties** using a large corpus of articles, press releases, and parliamentary questions.

The system relies on **semantic search** to retrieve relevant documents and a **generative LLM** to produce **accurate, neutral summaries in Arabic**, complete with **source citations**.

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-violet)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## ✨ Features

* **RAG Architecture**: Combines vector-based retrieval with generative AI for improved factual accuracy.
* **Semantic Search**: Uses `intfloat/multilingual-e5-base` with **FAISS** to retrieve relevant context even when exact keywords do not match.
* **Arabic Generative Summaries**: Uses `Qwen2.5-1.5B-Instruct` to generate fluent, neutral summaries in Arabic.
* **Source Citations**: Every generated answer includes references to the original documents (Title, Date, Link).
* **CPU Optimized**: Runs efficiently on standard hardware without a GPU using optimized PyTorch and quantization.
* **Offline Mode**: All models and embeddings are stored locally.

---

## 🛠️ Tech Stack

### Backend

* Python 3.10
* Flask

### Machine Learning

* PyTorch (CPU)
* Transformers
* Sentence-Transformers
* FAISS (CPU)

### Models

* **LLM**: `Qwen/Qwen2.5-1.5B-Instruct`
* **Embeddings**: `intfloat/multilingual-e5-base`

### Frontend

* HTML5
* TailwindCSS (via CDN)

---

## 📂 Project Structure

```bash
politixpert/
│
├── app.py                     # Flask application entry point
├── rag_engine.py              # Core RAG logic (loading, search, generation)
├── download_models.py         # Script to download Hugging Face models locally
├── requirements.txt           # Python dependencies
├── politixpert_data_cleaned.csv  # Cleaned dataset
├── embeddings.npy             # Precomputed embeddings (generated on GPU)
├── metadata.pkl               # Metadata linked to embeddings
│
├── models/                    # Local model storage
│   ├── qwen/
│   └── e5/
│
└── templates/
    └── index.html             # Web interface
```

---

## 🚀 Installation & Setup

### Prerequisites

* Python **3.10 or 3.11** (required for PyTorch compatibility)
* At least **8 GB RAM**

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/politixpert.git
cd politixpert
```

---

### 2️⃣ Create a Virtual Environment

**Windows**

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies (CPU Only)

⚠️ It is **strongly recommended** to install the CPU version of PyTorch to avoid unnecessary GPU dependencies.

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

---

### 4️⃣ Download Models Locally

⚠️ **This step is mandatory before running the application.**

Run the following script **once** to download the required Hugging Face models into the `models/` directory:

````bash
python download_models.py
````

---

### 5️⃣ Set Up the Data (Embeddings)

Due to GitHub file size limitations, large embedding files are **not stored in this repository**.

📥 **Download the precomputed embeddings and metadata from Google Drive:**

👉 [https://drive.google.com/drive/folders/1386iRLH_bL0MYJfkCaCQoBm60y70KcEG?usp=sharing](https://drive.google.com/drive/folders/1386iRLH_bL0MYJfkCaCQoBm60y70KcEG?usp=sharing)

After downloading, place the following files in the project root directory:

* `embeddings.npy`
* `metadata.pkl`
* `politixpert_data_cleaned.csv`
* (optional) any additional required data files

> These files were generated in a GPU environment and are required for the RAG retrieval step.

---

## 🏃‍♂️ Usage

### Start the Flask Server

```bash
python app.py
```

⏳ The first launch may take **30–60 seconds** to load models into memory.

---

### Access the Application

Open your browser and go to:

```
http://localhost:5000
```

---

### Ask a Question

Enter your question **in Arabic**, for example:

> ما موقف الأحزاب السياسية من إصلاح قطاع الصحة؟

The system will retrieve relevant documents and generate a neutral, cited summary.

---

## 🧠 How It Works

1. **User Query**: The user submits a question through the web interface.
2. **Embedding**: The query is converted into a vector using the E5 embedding model.
3. **Retrieval**: FAISS searches the local vector index to retrieve the top 15 relevant document chunks.
4. **Context Construction**: Results are grouped by political party and compiled into a structured prompt.
5. **Generation**: The Qwen 1.5B model generates concise, neutral summaries in Arabic.
6. **Display**: The frontend displays the generated answers along with their citations.

---

## 🤝 Contributing

Contributions are welcome! 🎉

Feel free to:

* Open an issue for bug reports or feature requests
* Submit a pull request with improvements or fixes

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Politixpert** aims to promote transparent, accessible, and neutral political analysis through AI.

