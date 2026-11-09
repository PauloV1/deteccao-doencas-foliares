"""Empacota o projeto ao lado da pasta.

  python scripts/empacotar.py            -> deteccao_folhas.zip: código, notebook, testes e documentação (para o Colab)
  python scripts/empacotar.py --entrega  -> deteccao_folhas_entrega.zip: o mesmo + resultados/ + modelo em models/

Ficam sempre de fora: dados, caches, saídas de treino e arquivos temporários."""
import argparse
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXCLUIR_PASTAS = {"__pycache__", ".pytest_cache", ".venv", "trabalho", "saidas", ".git", ".ipynb_checkpoints"}
EXCLUIR_SUFIXOS = {".pyc", ".pt", ".zip", ".npy", ".npz"}


def incluir(p: Path, entrega: bool) -> bool:
    rel = p.relative_to(RAIZ)
    if not p.is_file() or set(rel.parts) & EXCLUIR_PASTAS:
        return False
    if rel.parts[0] == "models" and not entrega:
        return p.name == ".gitkeep"                              # o Colab não precisa do modelo; ele é gerado no treino
    if rel.parts[0] == "resultados":
        return entrega and p.suffix != ".pt"                  # resultados/ só na entrega (sem pesos dos runs)
    if p.suffix in EXCLUIR_SUFIXOS:
        return entrega and rel.parts[0] == "models" and p.suffix == ".pt"
    return True


ap = argparse.ArgumentParser()
ap.add_argument("--entrega", action="store_true", help="inclui resultados/ e o modelo em models/")
args = ap.parse_args()
saida = RAIZ.parent / (f"{RAIZ.name}_entrega.zip" if args.entrega else f"{RAIZ.name}.zip")
arquivos = sorted(p for p in RAIZ.rglob("*") if incluir(p, args.entrega))
with zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as z:
    for p in arquivos:
        z.write(p, Path(RAIZ.name) / p.relative_to(RAIZ))
print(f"{len(arquivos)} arquivos -> {saida} ({saida.stat().st_size / 1e6:.1f} MB)")
