---
language: pt
license: mit
tags:
  - football
  - soccer
  - sports-analytics
  - dixon-coles
  - elo
  - brasileirao
metrics:
  - rps
  - log_loss
---

# Previsão Brasileirão Série A (Dixon-Coles + Elo)

Modelo que dá a probabilidade de **vitória do mandante, empate e vitória do visitante** em jogos do Campeonato Brasileiro Série A, além dos gols esperados e da probabilidade de cada placar.

- **Código, notebooks e backtest:** [github.com/JulioBuzzi/brasileirao-ml](https://github.com/JulioBuzzi/brasileirao-ml)
- **Demo:** [Space no Hugging Face](https://huggingface.co/spaces/JulioBuzzi/previsao-brasileirao)

## Como funciona

Média simples das probabilidades de dois modelos:

1. **Dixon-Coles**: os gols de cada time seguem uma Poisson com parâmetros de ataque, defesa e vantagem de casa, mais a correção ρ para placares baixos. É ajustado com as últimas 4 temporadas, e os jogos perdem peso com o tempo (meia-vida de 365 dias).
2. **Elo**: rating atualizado a cada jogo (k=20, mando=60, multiplicador por saldo de gols, regressão de 25% à média entre temporadas). Uma regressão logística multinomial transforma a diferença de Elo nas 3 probabilidades.

O arquivo `params.json` tem todos os parâmetros treinados: ratings de Elo, ataque e defesa de cada time, coeficientes da logística e ρ.

## Avaliação

Backtest rodada a rodada nas temporadas 2023, 2024 e 2025 (1.140 jogos). Antes de cada rodada, o modelo é retreinado só com os jogos já disputados. Os hiperparâmetros e a escolha do modelo foram feitos antes, numa validação com 2020–2022.

| Modelo | RPS ↓ | Log-loss ↓ | Brier ↓ | Acurácia |
|---|---|---|---|---|
| Baseline (frequência histórica) | 0,2206 | 1,0531 | 0,6346 | 48,2% |
| Elo + logística | 0,2099 | 1,0211 | 0,6122 | 49,3% |
| Dixon-Coles | 0,2088 | 1,0165 | 0,6088 | 49,2% |
| Gradient Boosting (Elo + forma) | 0,2119 | 1,0306 | 0,6171 | 49,7% |
| **Média Dixon-Coles + Elo (este modelo)** | **0,2084** | **1,0160** | **0,6084** | **49,9%** |

O ganho de RPS sobre o baseline é de 0,0122, com intervalo de confiança de 95% (bootstrap por rodada) entre 0,0079 e 0,0166.

**Teste fora da amostra em 2026** (339 jogos, rodadas 1 a 34, sem nenhuma decisão tomada com esses dados): RPS de 0,2014 contra 0,2177 do baseline. O ganho é de 0,0163, com IC 95% entre 0,0067 e 0,0259.

## Dados

Jogos da Série A de 2003 a 2025, do [Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset) (adaoduque), e de 2026 a partir de uma planilha própria, atualizada rodada a rodada. O `params.json` também traz a tabela atual, as previsões dos próximos jogos e uma simulação de Monte Carlo do fim do campeonato.

## Limitações

- Usa só resultados e placares: não sabe de lesões, escalação, troca de técnico ou calendário (Libertadores, Copa do Brasil).
- Nunca coloca o empate como resultado mais provável. Isso é comum em modelos de futebol, mas faz a acurácia ficar perto de 50%.
- Times recém-promovidos começam com um perfil de "time fraco" até acumular 10 jogos.
- **Não é recomendação de aposta.**