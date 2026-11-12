"""Treina o modelo final com todos os jogos de data/raw e salva models/params.json.
Se tiver jogos a disputar na temporada atual (data/raw/jogos_<ano>.csv), também salva
a tabela, as previsões dos próximos jogos e a simulação do fim do campeonato.
Por último, copia o params.json pra pasta static/ (Space do Hugging Face) e pra app/ (versão Gradio local).

    python scripts/treinar.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brasileirao.data import load_all, load_fixtures  # noqa: E402
from brasileirao.predictor import Previsor  # noqa: E402
from brasileirao.simulacao import simular, tabela  # noqa: E402

raw = load_all()
prev = Previsor.treinar(raw)
print(f"modelo treinado com jogos até {prev.p['dados_ate']}")

temporada = int(raw["temporada"].max())
faltam = load_fixtures(temporada)
if len(faltam):
    jogados = raw[raw["temporada"] == temporada]
    tab = tabela(jogados).reset_index()
    sim = simular(prev, jogados, faltam, n=20000).reset_index(names="time")
    proximos = []
    for r in faltam.itertuples(index=False):
        p = prev.prever(r.mandante, r.visitante)
        proximos.append({"rodada": int(r.rodada), "data": r.data.strftime("%d/%m/%Y"),
                         "mandante": r.mandante, "visitante": r.visitante,
                         "p_mandante": p["mandante"], "p_empate": p["empate"], "p_visitante": p["visitante"],
                         "gols_esperados": p["gols_esperados"]})
    prev.p["temporada_atual"] = {
        "temporada": temporada,
        "rodadas_disputadas": int(jogados["rodada"].max()),
        "tabela": tab.to_dict(orient="records"),
        "simulacao": sim.to_dict(orient="records"),
        "n_simulacoes": 20000,
        "proximos_jogos": proximos,
    }
    print(f"temporada {temporada}: {len(jogados)} jogos disputados, {len(faltam)} a disputar, simulação ok")

prev.salvar(ROOT / "models" / "params.json")
print("-> models/params.json")

(ROOT / "app").mkdir(exist_ok=True)
shutil.copy(ROOT / "models" / "params.json", ROOT / "app" / "params.json")
shutil.copy(ROOT / "src" / "brasileirao" / "predictor.py", ROOT / "app" / "predictor.py")
print("app/ atualizado (params.json + predictor.py)")

# Space estático (o que vai pro Hugging Face): só precisa do params.json
(ROOT / "static").mkdir(exist_ok=True)
shutil.copy(ROOT / "models" / "params.json", ROOT / "static" / "params.json")
print("static/ atualizado (params.json)")