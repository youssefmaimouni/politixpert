import pandas as pd
import numpy as np
import faiss
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import defaultdict

class PolitixpertRAG:
    def __init__(self, csv_path):
        print("🔄 Initialisation du moteur RAG (Optimisé)...")
        self.device = "cpu"
        
        # Fichiers pré-calculés
        self.emb_file = "embeddings.npy"
        self.meta_file = "metadata.pkl"

        # 1. Vérification si les fichiers Kaggle existent
        if os.path.exists(self.emb_file) and os.path.exists(self.meta_file):
            print("🚀 Fichiers pré-calculés trouvés ! Chargement rapide...")
            self._load_precomputed()
        else:
            print("⚠️ Fichiers pré-calculés introuvables. Calcul local (LENT)...")
            # Chargement CSV classique
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["content"].notna()]
            self.df = self.df[self.df["content"].str.len() > 100].reset_index(drop=True)
            self.chunks = []
            self.metadata = []
            self._prepare_chunks()
            
            # Modèle d'embedding pour le calcul
            self.embed_model = SentenceTransformer("./models/e5", device="cpu")
            self._compute_index_locally()

        # On a toujours besoin du modèle d'embedding pour encoder la QUESTION de l'utilisateur
        if not hasattr(self, 'embed_model'):
             self.embed_model = SentenceTransformer("./models/e5", device="cpu")

        # 5. Chargement du LLM (Qwen) sur CPU
        print("🤖 Chargement du LLM (Qwen)...")
        model_path = "./models/qwen" 

        self.generator = pipeline(
            "text-generation",
            model=model_path,  # Le chemin local au lieu de l'ID HuggingFace
            device=-1,
            trust_remote_code=True,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print("✅ Système prêt !")

    def _load_precomputed(self):
        """Charge les données générées sur Kaggle"""
        # 1. Charger les vecteurs
        embeddings = np.load(self.emb_file).astype("float32")
        
        # 2. Charger les métadonnées (Texte, Titres...)
        with open(self.meta_file, "rb") as f:
            self.metadata = pickle.load(f)
            
        # On extrait les chunks de texte des métadonnées pour l'affichage
        self.chunks = [m["text"] for m in self.metadata]
        
        # 3. Créer l'index FAISS (Ça prend 1 seconde car les vecteurs sont déjà là)
        print(f"🗂️ Indexation de {len(embeddings)} vecteurs...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def _chunk_text(self, text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _prepare_chunks(self):
        for idx, row in self.df.iterrows():
            text_chunks = self._chunk_text(row["content"])
            for i, chunk in enumerate(text_chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    "title": row.get("title", "Sans titre"),
                    "description": row.get("description", ""),
                    "date": str(row.get("date", "")),
                    "source": row["source"],
                    "link": row.get("link", "#"),
                    "text": chunk
                })

    def _compute_index_locally(self):
        """Méthode de secours si pas de fichiers Kaggle"""
        embeddings = self.embed_model.encode(
            self.chunks, 
            batch_size=16, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def semantic_search(self, query, top_k=15):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                # Le texte est déjà dans meta si chargé via pickle, sinon dans self.chunks
                text_content = meta.get("text", self.chunks[idx])
                
                results.append({
                    "score": float(scores[0][i]),
                    "text": text_content,
                    "title": meta["title"],
                    "description": meta["description"],
                    "date": meta["date"],
                    "source": meta["source"],
                    "link": meta["link"]
                })
        return results

    def _build_context(self, docs, max_docs=3):
        context_parts = []
        sources = []
        
        for d in docs[:max_docs]:
            doc_entry = f"""
معلومات الوثيقة:
- العنوان: {d['title']}
- وصف السياق: {d['description']}
- المحتوى النصي: {d['text']}
"""
            context_parts.append(doc_entry)
            sources.append({"title": d['title'], "date": d['date'], "link": d['link']})
            
        return "\n___________________\n".join(context_parts), sources

    def generate_answer(self, question):
        results = self.semantic_search(question, top_k=15)
        grouped = defaultdict(list)
        for r in results:
            grouped[r["source"]].append(r)

        final_response = []

        for party, docs in grouped.items():
            context_str, sources = self._build_context(docs)
            
            messages = [
                {"role": "system", "content": "أنت محلل سياسي خبير ومحايد."},
                {"role": "user", "content": f"""
استناداً للوثائق التالية، لخص موقف حزب "{party}".

الوثائق:
{context_str}

السؤال: {question}

التعليمات:
- ادمج المعلومات من العناوين والنصوص.
- اكتب ملخصاً مركزاً (فقرة أو فقرتين).
- تحدث بصيغة الغائب.
"""}
            ]

            try:
                out = self.generator(
                    messages, 
                    max_new_tokens=250, 
                    do_sample=False, 
                    return_full_text=False
                )
                summary = out[0]["generated_text"]
            except Exception as e:
                print(f"Erreur génération pour {party}: {e}")
                summary = "تعذر توليد الملخص."

            final_response.append({
                "party": party,
                "summary": summary,
                "sources": sources
            })
            
        return final_response