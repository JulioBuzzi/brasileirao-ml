"""Backtest rodada a rodada (walk-forward).

Pra cada rodada das temporadas de teste:
  1. treina com todos os jogos disputados ANTES do primeiro jogo da rodada
  2. prevê os jogos da rodada
Assim o modelo nunca vê o resultado que está tentando prever.
"""

import numpy as np
import pandas as pd

from .models import CLASSES, all_models


def rps(probs, y_idx):
    """Ranked Probability Score (quanto menor, melhor). Leva em conta a ordem
    mandante > empate > visitante: errar 'vitória' por 'empate' é menos grave
    que errar por 'derrota'."""
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    obs = np.zeros_like(probs); obs[np.arange(len(y_idx)), y_idx] = 1
    cum_o = np.cumsum(obs, axis=1)[:, :2]
    return np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2)


def metrics(probs, resultados):
    y = np.array([CLASSES.index(r) for r in resultados])
    p = np.clip(probs, 1e-12, 1)
    obs = np.zeros_like(p); obs[np.arange(len(y)), y] = 1
    return {
        "log_loss": -np.mean(np.log(p[np.arange(len(y)), y])),
        "rps": rps(p, y),
        "brier": np.mean(np.sum((p - obs) ** 2, axis=1)),
        "acuracia": np.mean(p.argmax(axis=1) == y),
    }


def walk_forward(df, temporadas=(2023, 2024, 2025), models=None, verbose=True):
    """Devolve um DataFrame com uma linha por (jogo, modelo) e as probabilidades."""
    models = models or all_models()
    rows = []
    for temp in temporadas:
        season = df[df["temporada"] == temp]
        for rodada in sorted(season["rodada"].unique()):
            jogos = season[season["rodada"] == rodada]
            inicio = jogos["data"].min()
            treino = df[df["data"] < inicio]
            for m in models:
                p = m.fit(treino).predict_proba(jogos)
                for (idx, j), pj in zip(jogos.iterrows(), p):
                    rows.append({"jogo": idx, "temporada": temp, "rodada": rodada,
                                 "mandante": j["mandante"], "visitante": j["visitante"],
                                 "resultado": j["resultado"], "modelo": m.nome,
                                 "p_H": pj[0], "p_D": pj[1], "p_A": pj[2]})
        if verbose:
            print(f"temporada {temp} ok")
    return pd.DataFrame(rows)


def summarize(preds, by=("modelo",)):
    out = []
    for keys, g in preds.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append({**dict(zip(by, keys)), "jogos": len(g),
                    **metrics(g[["p_H", "p_D", "p_A"]].to_numpy(), g["resultado"])})
    return pd.DataFrame(out)