"""Treina o modelo final com todos os jogos de data/raw e salva models/params.json.
Também copia os arquivos que o Space do Hugging Face precisa pra pasta app/.

    python scripts/treinar.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brasileirao.predictor import Previsor  # noqa: E402

prev = Previsor.treinar()
prev.salvar(ROOT / "models" / "params.json")
print(f"modelo treinado com jogos até {prev.p['dados_ate']} -> models/params.json")

shutil.copy(ROOT / "models" / "params.json", ROOT / "app" / "params.json")
shutil.copy(ROOT / "src" / "brasileirao" / "predictor.py", ROOT / "app" / "predictor.py")
print("app/ atualizado (params.json + predictor.py)")