import pandas as pd
import numpy as np
import faiss
import torch
import pickle
import os
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import defaultdict

class PolitixpertRAG:
    def __init__(self, csv_path):
        print("🔄 Initialisation du moteur RAG (Optimisé CPU + Local)...")
        self.device = "cpu"
        
        # Chemins des fichiers
        self.emb_file = "embeddings.npy"
        self.meta_file = "metadata.pkl"
        
        # Chemins des modèles locaux (téléchargés via download_models.py)
        # Si les dossiers n'existent pas, changez pour les IDs HuggingFace
        self.model_path_e5 = "./models/e5" if os.path.exists("./models/e5") else "intfloat/multilingual-e5-base"
        self.model_path_qwen = "./models/qwen" if os.path.exists("./models/qwen") else "Qwen/Qwen2.5-1.5B-Instruct"

        # 1. Vérification si les fichiers Kaggle existent
        if os.path.exists(self.emb_file) and os.path.exists(self.meta_file):
            print("🚀 Fichiers pré-calculés trouvés ! Chargement rapide...")
            self._load_precomputed()
        else:
            print("⚠️ Fichiers pré-calculés introuvables. Calcul local (LENT)...")
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["content"].notna()]
            self.df = self.df[self.df["content"].str.len() > 100].reset_index(drop=True)
            self.chunks = []
            self.metadata = []
            self._prepare_chunks()
            
            self.embed_model = SentenceTransformer(self.model_path_e5, device="cpu")
            self._compute_index_locally()

        # On charge le modèle d'embedding s'il n'est pas déjà chargé
        if not hasattr(self, 'embed_model'):
             print(f"🧠 Chargement du modèle d'embedding depuis {self.model_path_e5}...")
             self.embed_model = SentenceTransformer(self.model_path_e5, device="cpu")

        # 5. Chargement du LLM (Qwen) sur CPU
        print(f"🤖 Chargement du LLM (Qwen) depuis {self.model_path_qwen}...")
        self.generator = pipeline(
            "text-generation",
            model=self.model_path_qwen,
            device=-1, # CPU
            trust_remote_code=True,
            model_kwargs={"low_cpu_mem_usage": True} # Optimisation RAM
        )
        print("✅ Système prêt !")

    def _load_precomputed(self):
        embeddings = np.load(self.emb_file).astype("float32")
        with open(self.meta_file, "rb") as f:
            self.metadata = pickle.load(f)
        self.chunks = [m["text"] for m in self.metadata]
        
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

    def _clean_output(self, text):
        """Supprime les caractères chinois et nettoie le texte"""
        text = re.sub(r'[\u4e00-\u9fff]+', '', text) # Supprime Hanzi
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def semantic_search(self, query, top_k=20, min_score=0.82):
        """Recherche avec filtrage strict (Score & Longueur)"""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        
        # On cherche 2x plus de candidats pour pouvoir filtrer
        search_k = top_k * 2
        scores, indices = self.index.search(q_emb, search_k)
        
        results = []
        for i in range(search_k):
            score = float(scores[0][i])
            idx = indices[0][i]
            
            # FILTRE 1 : Score de pertinence
            if score < min_score:
                continue

            if idx < len(self.metadata):
                meta = self.metadata[idx]
                text_content = meta.get("text", self.chunks[idx])
                
                # FILTRE 2 : Longueur du texte (éviter le bruit)
                if len(text_content) < 50:
                    continue
                
                results.append({
                    "score": score,
                    "text": text_content,
                    "title": meta["title"],
                    "description": meta["description"],
                    "date": meta["date"],
                    "source": meta["source"],
                    "link": meta["link"]
                })
                
                if len(results) >= top_k:
                    break
        return results

    def _build_context(self, docs, max_docs=4):
        context_parts = []
        sources = []
        
        for d in docs[:max_docs]:
            # Contexte enrichi avec la DATE et la DESCRIPTION
            doc_entry = f"""
معلومات الوثيقة:
- التاريخ: {d['date']}
- العنوان: {d['title']}
- السياق: {d['description']}
- المحتوى النصي: {d['text']}
"""
            context_parts.append(doc_entry)
            sources.append({"title": d['title'], "date": d['date'], "link": d['link']})
            
        return "\n___________________\n".join(context_parts), sources

    def generate_answer(self, question):
        # On demande 20 docs, et on filtre avec min_score=0.82
        results = self.semantic_search(question, top_k=20, min_score=0.82)
        
        grouped = defaultdict(list)
        for r in results:
            grouped[r["source"]].append(r)

        final_response = []

        for party, docs in grouped.items():
            context_str, sources = self._build_context(docs)
            
            # Prompt ROBURSTE (Dates + Espace + Arabe uniquement)
            messages = [
                {"role": "system", "content": "أنت محلل سياسي خبير. اكتب باللغة العربية الفصحى فقط."},
                {"role": "user", "content": f"""
استناداً للوثائق التالية، لخص موقف حزب "{party}" بخصوص السؤال: "{question}".

الوثائق المتاحة:
{context_str}

⚠️ تعليمات صارمة (Strict Instructions):
1. **اللغة**: اكتب باللغة العربية فقط (Arabic Only). لا تستخدم أحرف صينية أو رموز غريبة.
2. **التواريخ**: انتبه لتواريخ الوثائق لفهم السياق (مثلاً نصوص نهاية 2023 تتحدث عن مالية 2024).
3. **الفضاء**: إذا كان السؤال عن "غزو الفضاء" أو "الكواكب" والنصوص سياسية، اكتب "لا توجد معلومات".
4. **التلخيص**: لخص الموقف في فقرة أو فقرتين بصيغة الغائب (يرى الحزب، يؤكد الحزب).
"""}
            ]

            try:
                out = self.generator(
                    messages, 
                    max_new_tokens=300, 
                    do_sample=False, 
                    return_full_text=False
                )
                raw_summary = out[0]["generated_text"]
                # Nettoyage final des caractères chinois
                summary = self._clean_output(raw_summary)
            except Exception as e:
                print(f"Erreur génération pour {party}: {e}")
                summary = "تعذر توليد الملخص."

            final_response.append({
                "party": party,
                "summary": summary,
                "sources": sources
            })
            
        return final_response