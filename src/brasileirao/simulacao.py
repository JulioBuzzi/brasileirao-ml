"""Tabela atual e simulação de Monte Carlo do resto do campeonato.

Cada jogo que falta é sorteado a partir da matriz de placares do Dixon-Coles,
com as probabilidades de vitória/empate/derrota ajustadas pra bater com o modelo
final (média Dixon-Coles + Elo). Repetindo isso milhares de vezes, sai a chance de
cada time terminar campeão, no G4, no G6 ou no Z4.

Critérios de desempate (CBF): pontos, vitórias, saldo de gols, gols marcados.
"""

import numpy as np
import pandas as pd


def tabela(jogos):
    """Classificação a partir dos jogos disputados (colunas mandante, visitante, gols_m, gols_v)."""
    linhas = []
    for r in jogos.itertuples(index=False):
        for time, gp, gc in ((r.mandante, r.gols_m, r.gols_v), (r.visitante, r.gols_v, r.gols_m)):
            linhas.append({"time": time, "J": 1, "V": int(gp > gc), "E": int(gp == gc), "D": int(gp < gc),
                           "GP": gp, "GC": gc})
    t = pd.DataFrame(linhas).groupby("time").sum()
    t["SG"] = t["GP"] - t["GC"]
    t["P"] = 3 * t["V"] + t["E"]
    t = t.sort_values(["P", "V", "SG", "GP"], ascending=False)
    t.insert(0, "pos", range(1, len(t) + 1))
    return t[["pos", "P", "J", "V", "E", "D", "GP", "GC", "SG"]]


def _ajusta_matriz(m, alvo):
    """Reescala as 3 regiões da matriz de placares (mandante vence / empate / visitante vence)
    pra que somem exatamente as probabilidades do modelo final."""
    m = m.copy()
    mask_h, mask_d, mask_a = np.tril(np.ones_like(m), -1) > 0, np.eye(len(m)) > 0, np.triu(np.ones_like(m), 1) > 0
    for mask, p in zip((mask_h, mask_d, mask_a), alvo):
        s = m[mask].sum()
        if s > 0:
            m[mask] *= p / s
    return m / m.sum()


def simular(previsor, jogados, faltam, n=10000, seed=0):
    """Devolve um DataFrame por time com pontos esperados e probabilidades de posição."""
    rng = np.random.default_rng(seed)
    base = tabela(jogados)
    times = list(base.index)
    idx = {t: i for i, t in enumerate(times)}
    k = len(times)

    P0 = base["P"].reindex(times).to_numpy()
    V0 = base["V"].reindex(times).to_numpy()
    SG0 = base["SG"].reindex(times).to_numpy()
    GP0 = base["GP"].reindex(times).to_numpy()

    P = np.tile(P0, (n, 1)).astype(float)
    V = np.tile(V0, (n, 1)).astype(float)
    SG = np.tile(SG0, (n, 1)).astype(float)
    GP = np.tile(GP0, (n, 1)).astype(float)

    for r in faltam.itertuples(index=False):
        m, _, _ = previsor.placares(r.mandante, r.visitante, max_gols=8)
        pr = previsor.prever(r.mandante, r.visitante)
        m = _ajusta_matriz(m, (pr["mandante"], pr["empate"], pr["visitante"]))
        flat = m.ravel()
        sorteio = rng.choice(len(flat), size=n, p=flat)
        gm, gv = np.divmod(sorteio, m.shape[1])
        i, j = idx[r.mandante], idx[r.visitante]
        P[:, i] += np.where(gm > gv, 3, np.where(gm == gv, 1, 0))
        P[:, j] += np.where(gv > gm, 3, np.where(gm == gv, 1, 0))
        V[:, i] += gm > gv
        V[:, j] += gv > gm
        SG[:, i] += gm - gv
        SG[:, j] += gv - gm
        GP[:, i] += gm
        GP[:, j] += gv

    # ordena cada simulação pelos critérios de desempate (+ ruído minúsculo pra empates totais)
    chave = P * 1e9 + V * 1e6 + (SG + 500) * 1e3 + GP + rng.random((n, k)) * 1e-3
    ordem = np.argsort(-chave, axis=1)
    pos = np.empty_like(ordem)
    pos[np.arange(n)[:, None], ordem] = np.arange(1, k + 1)

    out = pd.DataFrame({
        "pos_atual": base["pos"].reindex(times).to_numpy(),
        "pontos_atuais": P0,
        "pontos_esperados": P.mean(0),
        "campeao": (pos == 1).mean(0),
        "G4": (pos <= 4).mean(0),
        "G6": (pos <= 6).mean(0),
        "Z4": (pos > k - 4).mean(0),
        "pos_media": pos.mean(0),
    }, index=times)
    return out.sort_values("pontos_esperados", ascending=False)