"""Modelo final (média Dixon-Coles + Elo) pronto pra uso: treina com todos os jogos,
salva/carrega os parâmetros em JSON e prevê qualquer confronto.

    from brasileirao.predictor import Previsor
    prev = Previsor.treinar()            # usa todos os jogos de data/raw
    prev.salvar("models/params.json")
    prev = Previsor.carregar("models/params.json")
    prev.prever("Flamengo", "Palmeiras")
"""

import json
from datetime import date

import numpy as np
from scipy.stats import poisson



class Previsor:
    def __init__(self, params):
        self.p = params

    # -------------------------------------------------------------- treino
    @classmethod
    def treinar(cls, df=None):
        # imports aqui dentro: pra só PREVER (ex.: no Space) basta este arquivo + numpy/scipy
        from .data import load_all
        from .features import build_features
        from .models import DixonColes, EloLogit

        df_feat, elo = build_features(df if df is not None else load_all())
        dc = DixonColes().fit(df_feat)
        lr = EloLogit().fit(df_feat).lr_
        ultima = int(df_feat["temporada"].max())
        params = {
            "versao": 1,
            "treinado_em": date.today().isoformat(),
            "dados_ate": df_feat["data"].max().date().isoformat(),
            "times_ultima_temporada": sorted(set(df_feat[df_feat["temporada"] == ultima]["mandante"])),
            "ultima_temporada": ultima,
            "elo": {"ratings": {t: float(r) for t, r in elo.ratings_.items()}, "hfa": elo.hfa,
                    "novo": elo.new_team,
                    # coeficientes da logística (classes H, D, A), entrada = elo_diff / 100
                    "coef": lr.coef_[:, 0].tolist(), "intercept": lr.intercept_.tolist()},
            "dixon_coles": {"times": list(dc.times_), "ataque": dc.att_.tolist(), "defesa": dc.def_.tolist(),
                            "c": float(dc.c_), "casa": float(dc.home_), "rho": float(dc.rho_),
                            "ataque_novo": float(dc.att_novo_), "defesa_novo": float(dc.def_novo_)},
        }
        return cls(params)

    def salvar(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.p, f, ensure_ascii=False, indent=1)

    @classmethod
    def carregar(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------ previsão
    @property
    def times(self):
        return self.p["times_ultima_temporada"]

    def elo(self, time):
        e = self.p["elo"]
        return e["ratings"].get(time, e["novo"])

    def _prob_elo(self, mandante, visitante):
        e = self.p["elo"]
        diff = (self.elo(mandante) + e["hfa"] - self.elo(visitante)) / 100
        z = np.array(e["coef"]) * diff + np.array(e["intercept"])
        z = np.exp(z - z.max())
        return z / z.sum()

    def _dc(self, time):
        d = self.p["dixon_coles"]
        if time in d["times"]:
            i = d["times"].index(time)
            return d["ataque"][i], d["defesa"][i]
        return d["ataque_novo"], d["defesa_novo"]

    def placares(self, mandante, visitante, max_gols=10):
        d = self.p["dixon_coles"]
        am, dm = self._dc(mandante)
        av, dv = self._dc(visitante)
        lam = np.exp(d["c"] + d["casa"] + am - dv)
        mu = np.exp(d["c"] + av - dm)
        g = np.arange(max_gols + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        r = d["rho"]
        m[0, 0] *= 1 - lam * mu * r; m[0, 1] *= 1 + lam * r
        m[1, 0] *= 1 + mu * r;       m[1, 1] *= 1 - r
        return m / m.sum(), lam, mu

    def prever(self, mandante, visitante):
        m, lam, mu = self.placares(mandante, visitante)
        p_dc = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
        p = (p_dc + self._prob_elo(mandante, visitante)) / 2
        top = sorted(((m[i, j], f"{i}x{j}") for i in range(6) for j in range(6)), reverse=True)[:5]
        return {
            "mandante": float(p[0]), "empate": float(p[1]), "visitante": float(p[2]),
            "gols_esperados": (float(lam), float(mu)),
            "placares": [(s, float(pr)) for pr, s in top],
            "elo": (self.elo(mandante), self.elo(visitante)),
        }