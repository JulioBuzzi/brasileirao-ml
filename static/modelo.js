// Mesmas contas do src/brasileirao/predictor.py, em JavaScript.
// Recebe o params.json gerado por scripts/treinar.py.

function poissonPmf(k, lam) {
  let p = Math.exp(-lam);
  for (let i = 1; i <= k; i++) p *= lam / i;
  return p;
}

function criarModelo(params) {
  const elo = params.elo;
  const dc = params.dixon_coles;

  const ratingElo = (time) => (time in elo.ratings ? elo.ratings[time] : elo.novo);

  function probElo(mandante, visitante) {
    const diff = (ratingElo(mandante) + elo.hfa - ratingElo(visitante)) / 100;
    const z = elo.coef.map((c, i) => c * diff + elo.intercept[i]);
    const max = Math.max(...z);
    const e = z.map((v) => Math.exp(v - max));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s); // [mandante, empate, visitante]
  }

  function paramsDc(time) {
    const i = dc.times.indexOf(time);
    return i >= 0 ? [dc.ataque[i], dc.defesa[i]] : [dc.ataque_novo, dc.defesa_novo];
  }

  function placares(mandante, visitante, maxGols = 10) {
    const [am, dm] = paramsDc(mandante);
    const [av, dv] = paramsDc(visitante);
    const lam = Math.exp(dc.c + dc.casa + am - dv);
    const mu = Math.exp(dc.c + av - dm);
    const m = [];
    let total = 0;
    for (let i = 0; i <= maxGols; i++) {
      m.push([]);
      for (let j = 0; j <= maxGols; j++) {
        let p = poissonPmf(i, lam) * poissonPmf(j, mu);
        if (i === 0 && j === 0) p *= 1 - lam * mu * dc.rho;
        if (i === 0 && j === 1) p *= 1 + lam * dc.rho;
        if (i === 1 && j === 0) p *= 1 + mu * dc.rho;
        if (i === 1 && j === 1) p *= 1 - dc.rho;
        m[i].push(p);
        total += p;
      }
    }
    for (const linha of m) for (let j = 0; j < linha.length; j++) linha[j] /= total;
    return { matriz: m, lam, mu };
  }

  function prever(mandante, visitante) {
    const { matriz, lam, mu } = placares(mandante, visitante);
    let h = 0, d = 0, a = 0;
    const lista = [];
    matriz.forEach((linha, i) => linha.forEach((p, j) => {
      if (i > j) h += p; else if (i === j) d += p; else a += p;
      if (i < 6 && j < 6) lista.push([p, `${i}x${j}`]);
    }));
    const pe = probElo(mandante, visitante);
    lista.sort((x, y) => y[0] - x[0]);
    return {
      mandante: (h + pe[0]) / 2,
      empate: (d + pe[1]) / 2,
      visitante: (a + pe[2]) / 2,
      golsEsperados: [lam, mu],
      placares: lista.slice(0, 5).map(([p, s]) => [s, p]),
      elo: [ratingElo(mandante), ratingElo(visitante)],
    };
  }

  return { prever, times: params.times_ultima_temporada };
}

if (typeof module !== "undefined") module.exports = { criarModelo };
