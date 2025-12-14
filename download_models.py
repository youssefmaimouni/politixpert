from huggingface_hub import snapshot_download
import os

# Création du dossier pour stocker les modèles
os.makedirs("models", exist_ok=True)

print("⬇️ Téléchargement de Qwen 1.5B vers './models/qwen'...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir="./models/qwen",
    local_dir_use_symlinks=False  # Important pour Windows
)

print("⬇️ Téléchargement de E5 Base vers './models/e5'...")
snapshot_download(
    repo_id="intfloat/multilingual-e5-base",
    local_dir="./models/e5",
    local_dir_use_symlinks=False
)

print("✅ Téléchargement terminé ! Vous pouvez maintenant utiliser les modèles hors ligne.")