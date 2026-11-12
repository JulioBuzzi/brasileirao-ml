"""Converte o CSV exportado da planilha (JogosBrasil - Brasileirao) pro formato do projeto.

    python scripts/importar_planilha.py "JogosBrasil - Brasileirao.csv" --temporada 2026

Gera data/raw/jogos_<temporada>.csv com os jogos disputados (com placar) e os que
ainda faltam (gols vazios). Antes de gravar, confere a planilha e para se achar erro.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brasileirao.data import norm_team  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("arquivo")
parser.add_argument("--temporada", type=int, default=2026)
args = parser.parse_args()

df = pd.read_csv(args.arquivo)
df = df[(df["Temporada"] == args.temporada) & df["time_casa"].notna()].copy()
df["mandante"] = df["time_casa"].map(norm_team)
df["visitante"] = df["time_fora"].map(norm_team)
jogados = df[df["gols_casa"].notna()]

# ------------------------------------------------------------------ conferências
erros = []

dup = df[df.duplicated(["mandante", "visitante"], keep=False)]
for (m, v), g in dup.groupby(["mandante", "visitante"]):
    erros.append(f"{m} x {v} aparece {len(g)} vezes com o mesmo mando ({', '.join(g['id_jogo'])}). "
                 f"Um deles provavelmente está com os times invertidos.")

esperado = jogados["gols_casa"].gt(jogados["gols_fora"]).map({True: "Vitória", False: None})
esperado = esperado.fillna(jogados["gols_casa"].lt(jogados["gols_fora"]).map({True: "Derrota", False: "Empate"}))
for _, r in jogados[esperado != jogados["resultado_casa"]].iterrows():
    erros.append(f"{r['id_jogo']}: placar {r['gols_casa']:.0f}x{r['gols_fora']:.0f} mas resultado_casa = {r['resultado_casa']}")

times = sorted(set(df["mandante"]) | set(df["visitante"]))
if len(times) != 20:
    erros.append(f"{len(times)} times na temporada (esperado 20): {times}")

if erros:
    print("Corrija a planilha antes de importar:")
    for e in erros:
        print(" -", e)
    sys.exit(1)

# avisos que não impedem a importação
saldo_errado = jogados[(jogados["gols_casa"] - jogados["gols_fora"]) != jogados["saldo_casa"]]
for _, r in saldo_errado.iterrows():
    print(f"aviso: {r['id_jogo']} com saldo_casa errado (não afeta o modelo)")

out = pd.DataFrame({
    "rodada": df["rodada"].astype(int),
    "data": df["data"],
    "mandante": df["mandante"],
    "visitante": df["visitante"],
    "gols_mandante": df["gols_casa"].astype("Int64"),
    "gols_visitante": df["gols_fora"].astype("Int64"),
}).sort_values(["rodada", "data"])

destino = ROOT / "data" / "raw" / f"jogos_{args.temporada}.csv"
destino.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(destino, index=False)
print(f"{len(jogados)} jogos disputados + {len(out) - len(jogados)} a disputar -> {destino.relative_to(ROOT)}")
print("times:", ", ".join(times))