"""Features calculadas de forma "online": cada jogo só enxerga o que aconteceu antes dele.
Assim nada do futuro vaza pro treino.

- Elo: rating de força de cada time, atualizado depois de cada jogo
- Forma: pontos e gols nos últimos N jogos
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd


class Elo:
    """Elo no estilo World Football Elo.

    k        : quanto um jogo mexe no rating
    hfa      : vantagem de jogar em casa (em pontos de Elo)
    regress  : na virada de temporada, puxa cada rating essa fração em direção à média
    new_team : rating de quem nunca jogou a Série A (ou volta depois de sumir)
    """

    def __init__(self, k=20, hfa=60, regress=0.25, new_team=1430, mean=1500):
        self.k, self.hfa, self.regress = k, hfa, regress
        self.new_team, self.mean = new_team, mean

    def run(self, df):
        """Devolve o df com elo_m e elo_v ANTES de cada jogo."""
        ratings = {}
        last_season = {}
        elo_m, elo_v = [], []
        current = None

        for row in df.itertuples(index=False):
            if row.temporada != current:
                current = row.temporada
                # virada de temporada: regressão à média
                for t in ratings:
                    ratings[t] = ratings[t] + self.regress * (self.mean - ratings[t])

            for t in (row.mandante, row.visitante):
                # time novo, ou que ficou temporadas fora (rebaixado e voltou)
                if t not in ratings or current - last_season.get(t, current) > 1:
                    ratings[t] = min(ratings.get(t, self.new_team), self.new_team)
                last_season[t] = current

            rm, rv = ratings[row.mandante], ratings[row.visitante]
            elo_m.append(rm)
            elo_v.append(rv)

            expected = 1 / (1 + 10 ** (-(rm + self.hfa - rv) / 400))
            score = {"H": 1.0, "D": 0.5, "A": 0.0}[row.resultado]
            margin = abs(row.gols_m - row.gols_v)
            mult = 1 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
            delta = self.k * mult * (score - expected)
            ratings[row.mandante] = rm + delta
            ratings[row.visitante] = rv - delta

        out = df.copy()
        out["elo_m"] = elo_m
        out["elo_v"] = elo_v
        out["elo_diff"] = out["elo_m"] + self.hfa - out["elo_v"]
        self.ratings_ = ratings
        return out


def add_form(df, windows=(5, 10)):
    """Média de pontos, gols pró e gols contra nos últimos N jogos de cada time
    (qualquer mando), e média de pontos do mandante em casa / visitante fora."""
    hist = defaultdict(lambda: deque(maxlen=max(windows)))       # (pts, gp, gc)
    hist_home = defaultdict(lambda: deque(maxlen=max(windows)))  # pts em casa
    hist_away = defaultdict(lambda: deque(maxlen=max(windows)))  # pts fora
    last_date = {}
    feats = []

    for row in df.itertuples(index=False):
        f = {}
        for side, team in (("m", row.mandante), ("v", row.visitante)):
            h = list(hist[team])
            for w in windows:
                last = h[-w:]
                f[f"pts{w}_{side}"] = np.mean([x[0] for x in last]) if last else np.nan
                f[f"gp{w}_{side}"] = np.mean([x[1] for x in last]) if last else np.nan
                f[f"gc{w}_{side}"] = np.mean([x[2] for x in last]) if last else np.nan
            f[f"descanso_{side}"] = (row.data - last_date[team]).days if team in last_date else np.nan
        hh, ha = list(hist_home[row.mandante]), list(hist_away[row.visitante])
        f["pts_casa_m"] = np.mean(hh) if hh else np.nan
        f["pts_fora_v"] = np.mean(ha) if ha else np.nan
        feats.append(f)

        pts_m = {"H": 3, "D": 1, "A": 0}[row.resultado]
        pts_v = {"H": 0, "D": 1, "A": 3}[row.resultado]
        hist[row.mandante].append((pts_m, row.gols_m, row.gols_v))
        hist[row.visitante].append((pts_v, row.gols_v, row.gols_m))
        hist_home[row.mandante].append(pts_m)
        hist_away[row.visitante].append(pts_v)
        last_date[row.mandante] = last_date[row.visitante] = row.data

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    out["descanso_m"] = out["descanso_m"].clip(upper=30)
    out["descanso_v"] = out["descanso_v"].clip(upper=30)
    return out


def build_features(df, elo_params=None):
    elo = Elo(**(elo_params or {}))
    return add_form(elo.run(df)), elo