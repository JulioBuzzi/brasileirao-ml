"""Modelos de previsão. Todos seguem a mesma interface:

    model.fit(treino_df)               # só jogos que já aconteceram
    model.predict_proba(jogos_df)      # array (n, 3): P(mandante), P(empate), P(visitante)
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

CLASSES = ["H", "D", "A"]


def _to_idx(resultados):
    return np.array([CLASSES.index(r) for r in resultados])


# ------------------------------------------------------------------------ baseline
class Frequencia:
    """Chuta sempre a mesma coisa: a frequência histórica de vitória do mandante,
    empate e vitória do visitante nas últimas temporadas."""

    nome = "Baseline (frequência)"

    def __init__(self, temporadas=5):
        self.temporadas = temporadas

    def fit(self, df):
        ult = df[df["temporada"] > df["temporada"].max() - self.temporadas]
        self.p_ = ult["resultado"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
        return self

    def predict_proba(self, df):
        return np.tile(self.p_, (len(df), 1))


# ---------------------------------------------------------------------------- Elo
class EloLogit:
    """Usa só a diferença de Elo (já com o mando) e transforma em 3 probabilidades
    com uma regressão logística multinomial."""

    nome = "Elo + regressão logística"

    def __init__(self, desde=2006):
        self.desde = desde

    def fit(self, df):
        d = df[df["temporada"] >= self.desde]
        self.lr_ = LogisticRegression(C=1.0, max_iter=1000)
        self.lr_.fit(d[["elo_diff"]] / 100, _to_idx(d["resultado"]))
        return self

    def predict_proba(self, df):
        return self.lr_.predict_proba(df[["elo_diff"]] / 100)


# -------------------------------------------------------------------- Dixon-Coles
class DixonColes:
    """Modelo de Poisson para os gols (Dixon & Coles, 1997).

    gols do mandante  ~ Poisson(exp(c + casa + ataque_m - defesa_v))
    gols do visitante ~ Poisson(exp(c + ataque_v - defesa_m))

    + correção rho pros placares baixos (0x0, 1x0, 0x1, 1x1)
    + jogos antigos pesam menos (meia-vida em dias)
    """

    nome = "Dixon-Coles (Poisson)"

    def __init__(self, meia_vida=365, temporadas=4, l2=0.01, max_gols=10, min_jogos=10):
        self.meia_vida, self.temporadas, self.l2, self.max_gols = meia_vida, temporadas, l2, max_gols
        self.min_jogos = min_jogos

    def fit(self, df):
        d = df[df["temporada"] > df["temporada"].max() - self.temporadas]
        times = sorted(set(d["mandante"]) | set(d["visitante"]))
        self.times_ = {t: i for i, t in enumerate(times)}
        n = len(times)
        hi = d["mandante"].map(self.times_).to_numpy()
        ai = d["visitante"].map(self.times_).to_numpy()
        x, y = d["gols_m"].to_numpy(), d["gols_v"].to_numpy()
        dias = (d["data"].max() - d["data"]).dt.days.to_numpy()
        w = 0.5 ** (dias / self.meia_vida)

        def nll(p):
            att, dfn, c, home = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            lam = np.exp(c + home + att[hi] - dfn[ai])
            mu = np.exp(c + att[ai] - dfn[hi])
            loss = np.sum(w * (lam - x * np.log(lam) + mu - y * np.log(mu)))
            loss += self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            gl, gm = w * (lam - x), w * (mu - y)  # derivada em relação a log(lambda), log(mu)
            g_att = np.zeros(n); g_def = np.zeros(n)
            np.add.at(g_att, hi, gl); np.add.at(g_att, ai, gm)
            np.add.at(g_def, ai, -gl); np.add.at(g_def, hi, -gm)
            grad = np.concatenate([g_att + 2 * self.l2 * att, g_def + 2 * self.l2 * dfn,
                                   [gl.sum() + gm.sum(), gl.sum()]])
            return loss, grad

        p0 = np.zeros(2 * n + 2); p0[2 * n] = np.log(max(np.mean(np.r_[x, y]), 0.1))
        res = minimize(nll, p0, jac=True, method="L-BFGS-B")
        p = res.x
        self.att_, self.def_, self.c_, self.home_ = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        # time sem histórico (subiu agora): usa a média dos 4 mais fracos
        # (entre os que têm jogos suficientes pra ter parâmetros confiáveis)
        n_jogos = np.bincount(np.r_[hi, ai], minlength=n)
        confiaveis = np.where(n_jogos >= self.min_jogos)[0]
        forca = self.att_[confiaveis] + self.def_[confiaveis]
        fracos = confiaveis[np.argsort(forca)[:4]]
        self.att_novo_, self.def_novo_ = self.att_[fracos].mean(), self.def_[fracos].mean()

        # quem tem poucos jogos na janela (ex.: subiu e jogou 1 rodada) tem parâmetros
        # instáveis: mistura com o perfil de time novo, proporcional ao nº de jogos
        peso = np.minimum(n_jogos / self.min_jogos, 1.0)
        self.att_ = peso * self.att_ + (1 - peso) * self.att_novo_
        self.def_ = peso * self.def_ + (1 - peso) * self.def_novo_

        # rho: ajusta depois, com as médias de gols fixas
        lam = np.exp(self.c_ + self.home_ + self.att_[hi] - self.def_[ai])
        mu = np.exp(self.c_ + self.att_[ai] - self.def_[hi])
        self.rho_ = minimize_scalar(lambda r: -np.sum(w * np.log(np.clip(self._tau(x, y, lam, mu, r), 1e-10, None))),
                                    bounds=(-0.2, 0.2), method="bounded").x
        return self

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        t = np.ones_like(lam, dtype=float)
        t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
        t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
        t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
        t = np.where((x == 1) & (y == 1), 1 - rho, t)
        return t

    def _params(self, team):
        i = self.times_.get(team)
        if i is None:
            return self.att_novo_, self.def_novo_
        return self.att_[i], self.def_[i]

    def expected_goals(self, mandante, visitante):
        am, dm = self._params(mandante)
        av, dv = self._params(visitante)
        return np.exp(self.c_ + self.home_ + am - dv), np.exp(self.c_ + av - dm)

    def score_matrix(self, mandante, visitante):
        """Matriz P[i, j] = probabilidade do placar i x j."""
        lam, mu = self.expected_goals(mandante, visitante)
        g = np.arange(self.max_gols + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * self.rho_
        m[0, 1] *= 1 + lam * self.rho_
        m[1, 0] *= 1 + mu * self.rho_
        m[1, 1] *= 1 - self.rho_
        return m / m.sum()

    def predict_proba(self, df):
        out = []
        for m, v in zip(df["mandante"], df["visitante"]):
            s = self.score_matrix(m, v)
            out.append([np.tril(s, -1).sum(), np.trace(s), np.triu(s, 1).sum()])
        return np.array(out)


# --------------------------------------------------------------- gradient boosting
FEATURES_GBM = [
    "elo_diff", "elo_m", "elo_v",
    "pts5_m", "pts5_v", "pts10_m", "pts10_v",
    "gp5_m", "gc5_m", "gp5_v", "gc5_v", "gp10_m", "gc10_m", "gp10_v", "gc10_v",
    "pts_casa_m", "pts_fora_v", "descanso_m", "descanso_v", "rodada",
]


class GradientBoosting:
    """HistGradientBoosting com Elo + forma recente + descanso + rodada."""

    nome = "Gradient Boosting (Elo + forma)"

    def __init__(self, desde=2006, seed=0):
        self.desde, self.seed = desde, seed

    def fit(self, df):
        d = df[df["temporada"] >= self.desde]
        self.clf_ = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=80,
            l2_regularization=1.0, random_state=self.seed)
        self.clf_.fit(d[FEATURES_GBM], _to_idx(d["resultado"]))
        return self

    def predict_proba(self, df):
        return self.clf_.predict_proba(df[FEATURES_GBM])


class Media:
    """Média simples das probabilidades de outros modelos."""

    def __init__(self, models, nome=None):
        self.models = models
        self.nome = nome or "Média (" + " + ".join(m.nome.split(" (")[0] for m in models) + ")"

    def fit(self, df):
        for m in self.models:
            m.fit(df)
        return self

    def predict_proba(self, df):
        return np.mean([m.predict_proba(df) for m in self.models], axis=0)


def all_models():
    return [Frequencia(), EloLogit(), DixonColes(), GradientBoosting(),
            Media([DixonColes(), EloLogit()], nome="Média (Dixon-Coles + Elo)")]