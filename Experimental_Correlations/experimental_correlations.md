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

----

Sugiro inverter a hierarquia: usar a propagação a partir dos dados brutos (o que a nota de vocês já faz) como método principal, e deixar o GP e as redes neurais como diagnóstico e direção futura. Nenhum método, GP ou rede neural, cria informação sobre correlações que não está nos dados publicados; o que muda é de onde você tira essa informação.

## Alternativas, em ordem de prioridade

1. **Propagação bottom-up (estender a nota).** Yields, ⟨p_T⟩ e U(x_T) são todos derivados do mesmo espectro. Propagando as incertezas do espectro bruto por Monte Carlo, você obtém a covariância entre bins e também entre observáveis, sem ansatz. A nota já tem a maquinaria; falta aplicá-la aos observáveis integrados da análise do JETSCAPE.

2. **Modelo de fatores (low-rank + diagonal).** Σ = D + Σₖ uₖuₖᵀ, com poucos modos físicos: normalização, tilt em p_T, inclinação em centralidade. É equivalente a amostrar as amplitudes como parâmetros de nuisance no MCMC, como se faz em ajustes de PDFs. Os casos (g, h) da nota já são isso, calibrados em δN e δ⟨p_T⟩.

3. **Erro sobre o erro.** Tratar os próprios σ_sys como incertos (modelo de variância Gama do Cowan, EPJC 2019, citado de memória). Fica robusto a erros inflados ou subestimados e implementa-se com um parâmetro extra por fonte.

4. **Envelope de cenários.** Rodar uma família de covariâncias plausíveis e reportar a envoltória, como na tabela da nota. É o menos sofisticado e o mais difícil de contestar.

5. **Termo de discrepância do modelo (Kennedy–O'Hagan).** Um GP para o erro da teoria, separado do experimental. Importa porque erro experimental correlacionado e deficiência do modelo têm a mesma cara; o χ²/ponto de 2.3–3.5 da nota sugere que parte do efeito é do modelo.

## Onde redes neurais ajudam de fato

- **Simulation-based inference** (neural posterior estimation com normalizing flows): treina-se em pares (parâmetros, pseudo-dados), com o ruído gerado por um simulador em que α, ℓ e os modos são sorteados de um prior. O posterior já sai marginalizado sobre as correlações e não exige likelihood gaussiana.
- **Cobertura amortizada:** com a rede treinada, closure em milhares de pseudo-datasets custa segundos, contra um MCMC por dataset.
- **O que não funciona:** uma rede que "aprende Σ" direto dos dados. Com uma única realização de cada medida não há conjunto de treino; ela cairia na mesma degenerescência do GP, com menos controle.

## Como usar a nota

Ela é o melhor material que você tem, por três razões:

- **Contraexemplo à Fig. 23.** No JETSCAPE as correlações alargam o posterior; no espectro escalado, a tensão em w fica igual ou mais forte (−2Δln L de 8.1 para até 23.1). A mensagem do talk passa a ser que o efeito não tem sinal definido e depende de como o observável é construído.
- **Diagnóstico pronto.** χ²/287 = 0.09 com os erros originais é o argumento dos pulls com números reais: erros inflados e contados duas vezes.
- **Método validado.** A propagação linear concorda com o Monte Carlo não linear em 1–3%.

No talk, eu colocaria um slide "Case study: scaled spectra" com a `fig_w_short.pdf` e três linhas da tabela (original, (c), (h)), logo depois da Fig. 23. A direção futura seria aplicar a mesma propagação aos observáveis integrados e marginalizar com SBI.

Ao apresentar, mostre só w: a própria nota diz que ζ/s e σ_k ainda derivam em algumas cadeias e que os melhores ajustes dependem de extrapolação do emulador.

Quer que eu adicione esse slide e reescreva o "This work" em torno da propagação, com o GP rebaixado a diagnóstico?
