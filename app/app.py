"""Space do Hugging Face: previsões do Brasileirão.
Usa só predictor.py + params.json (gerados por scripts/treinar.py)."""

from pathlib import Path

import gradio as gr
import pandas as pd

try:
    from predictor import Previsor  # dentro do Space
except ImportError:  # rodando de dentro do repo
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from brasileirao.predictor import Previsor

prev = Previsor.carregar(Path(__file__).parent / "params.json")
TIMES = prev.times
ATUAL = prev.p.get("temporada_atual")


def pct(x):
    if x >= 0.9995:
        return ">99,9%"
    if 0 < x < 0.0005:
        return "<0,1%"
    return f"{x * 100:.1f}%".replace(".", ",")


def prever(mandante, visitante):
    if mandante == visitante:
        raise gr.Error("Escolha dois times diferentes.")
    r = prev.prever(mandante, visitante)
    probs = {f"{mandante} vence": r["mandante"], "Empate": r["empate"], f"{visitante} vence": r["visitante"]}
    lam, mu = r["gols_esperados"]
    em, ev = r["elo"]
    resumo = (f"**Gols esperados:** {mandante} {lam:.2f} x {mu:.2f} {visitante}  \n"
              f"**Elo:** {mandante} {em:.0f} · {visitante} {ev:.0f}  \n"
              f"**Odd justa** (1 / probabilidade): {1/r['mandante']:.2f} · {1/r['empate']:.2f} · {1/r['visitante']:.2f}")
    placares = pd.DataFrame(r["placares"], columns=["Placar", "Probabilidade"])
    placares["Probabilidade"] = placares["Probabilidade"].map(pct)
    return probs, resumo, placares


def tabela_proximos():
    df = pd.DataFrame(ATUAL["proximos_jogos"])
    return pd.DataFrame({
        "Rodada": df["rodada"], "Data": df["data"], "Mandante": df["mandante"], "Visitante": df["visitante"],
        "Mandante vence": df["p_mandante"].map(pct), "Empate": df["p_empate"].map(pct),
        "Visitante vence": df["p_visitante"].map(pct),
    })


def tabela_simulacao():
    sim = pd.DataFrame(ATUAL["simulacao"])
    tab = pd.DataFrame(ATUAL["tabela"]).set_index("time")
    sim = sim.sort_values(["pos_atual"])
    return pd.DataFrame({
        "Pos": sim["pos_atual"], "Time": sim["time"], "Pts": sim["pontos_atuais"],
        "J": sim["time"].map(tab["J"]), "SG": sim["time"].map(tab["SG"]),
        "Pts esperados": sim["pontos_esperados"].round(1),
        "Campeão": sim["campeao"].map(pct), "G4": sim["G4"].map(pct), "G6": sim["G6"].map(pct),
        "Rebaixamento": sim["Z4"].map(pct),
    })


with gr.Blocks(title="Previsão Brasileirão") as demo:
    gr.Markdown(
        f"""# ⚽ Previsão Brasileirão Série A
Média de dois modelos: **Dixon-Coles** (Poisson para os gols) + **Elo**, treinados com jogos desde 2003, até **{prev.p['dados_ate']}**.
No backtest rodada a rodada de 2023–2025 (1.140 jogos), o RPS ficou em 0,208, contra 0,221 do baseline. Em 2026, fora da amostra, ficou em 0,201 contra 0,218.
Código e metodologia: [github.com/JulioBuzzi/brasileirao-ml](https://github.com/JulioBuzzi/brasileirao-ml)"""
    )
    with gr.Tab("Confronto"):
        with gr.Row():
            mand = gr.Dropdown(TIMES, value="Flamengo" if "Flamengo" in TIMES else TIMES[0], label="Mandante")
            vis = gr.Dropdown(TIMES, value="Palmeiras" if "Palmeiras" in TIMES else TIMES[1], label="Visitante")
        btn = gr.Button("Prever", variant="primary")
        with gr.Row():
            probs = gr.Label(label="Probabilidades", num_top_classes=3)
            with gr.Column():
                resumo = gr.Markdown()
                placares = gr.Dataframe(label="Placares mais prováveis", interactive=False)
        btn.click(prever, [mand, vis], [probs, resumo, placares])
        demo.load(prever, [mand, vis], [probs, resumo, placares])

    if ATUAL:
        with gr.Tab("Próximos jogos"):
            gr.Markdown(f"Jogos que faltam no Brasileirão {ATUAL['temporada']}.")
            gr.Dataframe(tabela_proximos(), interactive=False, wrap=True)
        with gr.Tab("Simulação do campeonato"):
            gr.Markdown(
                f"Tabela depois da rodada {ATUAL['rodadas_disputadas']} e chance de cada time terminar campeão, "
                f"no G4, no G6 ou rebaixado, em {ATUAL['n_simulacoes']:,} simulações do resto do campeonato."
                .replace(",", "."))
            gr.Dataframe(tabela_simulacao(), interactive=False, wrap=True)

    gr.Markdown("*Projeto de portfólio. Não é recomendação de aposta.*")

if __name__ == "__main__":
    demo.launch()