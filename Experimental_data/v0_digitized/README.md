# Dados digitalizados de v0 e v0(pT): STAR e CMS

Fontes (resultados preliminares; cite sempre os originais):

- **STAR**, arXiv:2608.00190, Figs. 1-3 (Au+Au e O+O a 200 GeV; Au+Au BES 9.2-200 GeV)
- **CMS**, CMS-PAS-HIN-25-016, Figs. 1-6 (pp 13 TeV, pPb 8.16 TeV, OO e NeNe 5.36 TeV)

## Método

As figuras dos dois PDFs são vetoriais. Os pontos não foram lidos de pixels: as posições dos
marcadores, barras de erro e caixas sistemáticas vêm dos comandos de desenho do PDF
(`code/vecdig.py`), e cada eixo é calibrado pelos ticks e seus rótulos impressos. Cada série é
identificada pelo marcador (cor, forma, cheio/vazio) e ligada ao texto da legenda.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `data/*_points.csv` | um ponto por linha (tabela plana) |
| `data/*_curves.csv` | curvas e bandas de modelos (3DGlauber+MUSIC, TRENTo+MUSIC(+SMASH), HIJING, PYTHIA) e a banda O+O 10-20% do STAR |
| `data/*.json` | tudo em forma hierárquica (painel -> séries -> pontos), com estilo do marcador e limites dos eixos |
| `data/derived_v0_meanpT.json` | v0 e <pT> por classe, **derivados** da razão entre figuras (ver abaixo) |
| `data/calibration_log.txt`, `data/crosschecks.txt` | verificações |
| `plots/` | todas as figuras redesenhadas a partir dos dados extraídos |
| `code/` | `extract_all.py` (extração), `plot_all.py` (figuras), `crosscheck.py`, `load_v0_data.py` |

Colunas de `*_points.csv`: `figure, system, energy, selection, series, x_name, y_name, x, y,
stat_up, stat_dn, syst_up, syst_dn, box_x_lo, box_x_hi, hidden_stat_max, clipped`.

- `stat_up/stat_dn`: barra estatística. **Vazio** quando a barra é menor que o marcador e por isso
  não foi desenhada; nesse caso `hidden_stat_max` (meia altura do marcador, em unidades de y) é um
  limite superior para o erro estatístico.
- `syst_up/syst_dn`: meia altura da caixa sistemática.
- `box_x_lo/box_x_hi`: bordas horizontais da caixa. No CMS as caixas são contíguas, ou seja, são as
  bordas dos bins de pT (ou de pT/<pT>). No STAR a largura é fixa e apenas estética.
- `clipped = 1`: a barra ou caixa toca a borda do painel, então o erro é um limite inferior.

Uso rápido:

```python
from load_v0_data import load, series
df = load("data")
x, y, stat, syst = series(df, "CMS", "fig1_OO", "Ntrk 250-400")
```

Para refazer tudo: `python3 code/extract_all.py STAR.pdf CMS.pdf data && python3 code/crosscheck.py data && python3 code/plot_all.py data plots`.

## Verificações

- Eixos: resíduo do ajuste aos ticks rotulados <= 1e-3 da unidade do eixo em todos os 33 painéis; a
  linha tracejada de zero, que não entra na calibração, cai em y = 0 (|y| < 0.002).
- STAR: v0 = [Fig. 1b]/[Fig. 1c] reproduz os oito valores de v0 da Fig. 1a com diferença < 0.1%.
- CMS: [Fig. 1]/[Fig. 4] dá <pT> e v0 constantes ao longo de pT em cada classe; a Fig. 6 é
  reproduzida por v0(pT) x v0 dentro de 1%; a Fig. 5 coincide com a Fig. 4 dentro de 0.1%.

## Limitações

- São valores lidos de figuras preliminares, não tabelas oficiais (não há HEPData para nenhuma das
  duas). A precisão é a do desenho no PDF, cerca de 1e-4 da faixa do eixo.
- Erros estatísticos só existem onde a barra é visível (65 de 978 pontos no STAR, 135 de 2508 no CMS).
- CMS Fig. 4, pp, Ntrk 10-20: só 21 dos 29 pontos aparecem; os demais estão acima do limite do eixo.
- CMS pp, Ntrk 100-120: as Figs. 1, 4 e 6 diferem entre si em até ~0.7% (as outras classes fecham em 0.1%);
  a diferença está nas figuras originais.
- `derived_v0_meanpT.json` não é resultado publicado pelas colaborações: é a razão entre figuras.
- STAR Fig. 2(a): as bandas foram rotuladas como 3DGlauber+MUSIC com base na legenda do painel (b).
