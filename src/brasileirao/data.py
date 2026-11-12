"""Carrega e unifica os jogos do Brasileirão Série A.

Fontes:
- 2003–2024: campeonato-brasileiro-full.csv (github.com/adaoduque/Brasileirao_Dataset)
- 2025:      brasileirao-2025.json (mesmo repositório)
- 2026+:     CSV próprio (data/raw/jogos_<ano>.csv), com colunas
             rodada,data,mandante,visitante,gols_mandante,gols_visitante

Saída: um DataFrame com uma linha por jogo e colunas padronizadas:
temporada, rodada, data, mandante, visitante, gols_m, gols_v, resultado (H/D/A)
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# commit fixo do dataset público -> resultados reproduzíveis
FONTE = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/15f027c3be337bb130d6ae94436b5cdca737377c/"
ARQUIVOS_FONTE = ["campeonato-brasileiro-full.csv", "data/brasileirao-2025.json"]


def download():
    """Baixa os arquivos do Brasileirao_Dataset pra data/raw (só na primeira vez)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for arq in ARQUIVOS_FONTE:
        destino = RAW / Path(arq).name
        if not destino.exists():
            print(f"baixando {arq}...")
            urllib.request.urlretrieve(FONTE + arq, destino)

# nomes diferentes pro mesmo clube nas fontes -> um nome só
TEAM_NAMES = {
    "Atlético-MG": "Atletico-MG",
    "Botafogo": "Botafogo-RJ",
    "Ceará SC": "Ceara",
    "EC Vitória": "Vitoria",
    "Grêmio": "Gremio",
    "Sport Recife": "Sport",
    "São Paulo": "Sao Paulo",
    "Vasco da Gama": "Vasco",
    "Red Bull Bragantino": "Bragantino",
    "RB Bragantino": "Bragantino",
    "Athletico Paranaense": "Athletico-PR",
    "Atlético-GO": "Atletico-GO",
    "Cuiabá": "Cuiaba",
    "Goiás": "Goias",
    "Avaí": "Avai",
    "Criciúma": "Criciuma",
    "Paraná": "Parana",
    "América-MG": "America-MG",
    "Vitória": "Vitoria",
    "Ceará": "Ceara",
}


def norm_team(name):
    name = str(name).strip()
    return TEAM_NAMES.get(name, name)


def _result(gm, gv):
    return "H" if gm > gv else ("A" if gm < gv else "D")


def load_csv_2003_2024(path=RAW / "campeonato-brasileiro-full.csv"):
    df = pd.read_csv(path).sort_values("ID").reset_index(drop=True)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)

    # o CSV não tem coluna de temporada: começa uma nova quando a rodada "volta" pra 1
    # depois de rodadas altas (isso trata certo a temporada 2020, que terminou em fev/2021)
    nova = (df["rodata"] == 1) & (df["rodata"].shift(fill_value=99) >= 30)
    df["temporada"] = 2002 + nova.cumsum()

    out = pd.DataFrame({
        "temporada": df["temporada"],
        "rodada": df["rodata"],
        "data": df["data"],
        "mandante": df["mandante"].map(norm_team),
        "visitante": df["visitante"].map(norm_team),
        "gols_m": df["mandante_Placar"].astype(int),
        "gols_v": df["visitante_Placar"].astype(int),
    })
    return out


def load_json_season(path, season):
    """Lê o JSON de uma temporada (formato do adaoduque). Algumas datas vêm sem ano
    ('07/12'), então o ano é completado a partir da temporada."""
    raw = json.load(open(path, encoding="utf-8"))
    rows = []
    for rodada, jogos in raw.items():
        for j in jogos:
            d = j["date"]
            dia, mes = d[:5].split("/")
            data = pd.Timestamp(year=season, month=int(mes), day=int(dia))
            rows.append({
                "temporada": season,
                "rodada": int(rodada),
                "data": data,
                "mandante": norm_team(j["clubs"]["home"]),
                "visitante": norm_team(j["clubs"]["away"]),
                "gols_m": int(j["goals"]["home"]),
                "gols_v": int(j["goals"]["away"]),
            })
    return pd.DataFrame(rows)


def load_own_csv(path, season):
    """CSV próprio (ex.: exportado do Google Sheets) com uma temporada."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["gols_mandante", "gols_visitante"])  # jogos ainda não disputados ficam de fora
    return pd.DataFrame({
        "temporada": season,
        "rodada": df["rodada"].astype(int),
        "data": pd.to_datetime(df["data"], dayfirst=True),
        "mandante": df["mandante"].map(norm_team),
        "visitante": df["visitante"].map(norm_team),
        "gols_m": df["gols_mandante"].astype(int),
        "gols_v": df["gols_visitante"].astype(int),
    })


def load_all():
    """Todos os jogos disponíveis, em ordem cronológica."""
    download()
    parts = [load_csv_2003_2024()]
    for season in range(2025, 2031):
        js = RAW / f"brasileirao-{season}.json"
        own = RAW / f"jogos_{season}.csv"
        if js.exists():
            parts.append(load_json_season(js, season))
        elif own.exists():
            parts.append(load_own_csv(own, season))
    df = pd.concat(parts, ignore_index=True)
    df["resultado"] = [_result(a, b) for a, b in zip(df["gols_m"], df["gols_v"])]
    df = df.sort_values(["data", "temporada", "rodada"], kind="stable").reset_index(drop=True)
    return df