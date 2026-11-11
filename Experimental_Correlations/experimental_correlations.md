Vou olhar os dois repositórios para apontar exatamente onde a covariância experimental é construída e o que precisa mudar.

A análise é viável e barata (roda em laptop com o notebook da summer school), mas um GP ajustado aos dados de um único experimento só identifica parte da correlação. Reescrevi os slides 8 a 15 de `lasnpa_correlations.tex` em torno dessa ideia e deixei um módulo de partida, `exp_cov.py`.

## O que o GP nos dados consegue inferir

Modelo por observável, em função da centralidade:

y_i = f(c_i) + ε_stat + ε_unc + ε_cor, com Σ_sys = σ_i σ_j [(1−α) δ_ij + α exp(−Δc²/ℓ²)]

- **Identificável:** a fração correlacionada α. Se os pontos espalham em torno de uma curva suave muito menos do que as barras sistemáticas permitem, o erro não pode ser independente bin a bin.
- **Não identificável:** o comprimento ℓ de longo alcance. Um deslocamento coerente de todos os bins é indistinguível de outra tendência física f. Precisa de prior ou de informação extra.
- **Informação extra para fixar ℓ e ρ₀:**
  - incertezas publicadas de razões (K/π, v₃/v₂, entre energias), que dão ρ diretamente;
  - o mesmo observável medido por ALICE, ATLAS e CMS;
  - sistemáticas separadas por fonte no HEPData.

No teste sintético do `exp_cov.py` (14 bins, sistemático de 5%, ℓ fixo em 0.5), o ajuste devolve α ≈ 0.06 quando o verdadeiro é 0 e α ≈ 0.88 quando é 0.9.

## Etapas

1. **Dados:** baixar do HEPData mantendo stat e sys separados. No `bayes_exp.py` (linhas 47–51) eles são somados em quadratura; é preciso guardar os dois.
2. **GP nos dados:** `fit_observable` para cada observável, varrendo ℓ em vez de ajustá-lo.
3. **Montar Σ_exp:** `build_cov` segue os mesmos grupos de `expt_obs_corr_group` do `bayes_mcmc.py`.
4. **Inferência:** quatro runs, mudando só Σ_exp:
   - A: diagonal
   - B: ansatz do paper (ℓ = 0.5, fator 0.8), para reproduzir a Fig. 23
   - C: GP no melhor ajuste
   - D: GP marginalizado (misturar cadeias com (α, ℓ) sorteados do posterior do GP)
5. **Closure:** pseudo-dados com ruído correlacionado conhecido; checar se o GP recupera α e se a cobertura de 90% volta.

## Onde mexer no código

- **Notebook da summer school:** nas células 84 e 86, trocar `exp_var=np.diag(y_exp_variance)` pela matriz de `build_cov`. Por padrão ele usa pseudo-dados de validação; a linha para os dados reais está comentada na célula 50. As bandas de η/s e ζ/s saem da célula 102.
- **js-sims-bayes:** a Fig. 23 vem de `compute_cov` em `bayes_mcmc.py`, ligada por `assume_corr_exp_error` e `cent_corr_length` em `configurations.py` (linhas 262–263).
- **Dois detalhes do original:**
  - o "c" é `linspace(0,1,nbins)`, ou seja, índice do bin e não centralidade real;
  - a correlação é aplicada ao erro total (stat + sys), não só ao sistemático.

  Para reproduzir a Fig. 23, siga isso; para a sua versão, vale corrigir os dois e dizer isso no talk.

## Plots e onde entram

| Slide | Arquivo em `Figures/` | Conteúdo |
|---|---|---|
| 10 | `gp_fit_data.pdf` | pontos do HEPData com GP |
| 11 | `pull_distribution.pdf` | pulls vs. N(0,1) |
| 13 | `corr_matrix_ansatz.pdf`, `corr_matrix_gp.pdf` | matrizes 110×110 |
| 14 | `eta_s_posterior_runs.pdf`, `zeta_s_posterior_runs.pdf` | estilo Fig. 23, runs A a D |
| 15 | `parameter_widths_runs.pdf` | intervalo de 90% dos 17 parâmetros |
| 16 | `closure_coverage.pdf` | closure |

O plot de pulls é o mais rápido de fazer e o mais convincente: mostra, só com dados públicos, que a hipótese diagonal é inconsistente.

Os números do slide de setup (17 parâmetros, 110 pontos, 10 PCs, Pb–Pb 2.76 TeV) vêm do notebook; confirme contra a análise final. Se quiser, o próximo passo é eu adaptar o notebook para rodar os runs A e B e gerar a primeira versão da figura das viscosidades.
