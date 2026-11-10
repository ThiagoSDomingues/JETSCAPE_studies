A partir do notebook, aqui estão todos os arquivos e pastas necessários:

---

**Arquivos de entrada (dados experimentais)**
```
experimental_data/differential_radial_flow/
    HEPData-ins2907010-v1-Figure_2a.csv        # ATLAS v0 global
    HEPData-ins2907010-v1-Figure_4a_cent7.csv  # ATLAS v0(pT) 50-60%
main.root
SystUncs.root
```

**Resultados de PCA**
```
pca_results/
    pca_results_global_v0_all_corrections.pkl
    pca_results_v0pt_all_corrections.pkl
```

**Emuladores treinados**
```
emulators_global_v0/
    gp_emulators_global_v0_Grad.pkl

emulators_v0pt/
    gp_emulators_v0pt_Grad.pkl
```

**Predições pré-computadas (apenas para a célula 1)**
```
predictions_global_v0/
    all_predictions_global_v0.pkl
```

**Amostras do posterior**
```
new_LHC_posterior_samples.csv

posterior/
    mcmc_chain_Grad.csv
```

**Design points**
```
design_pts_Pb_Pb_2760_production/
    design_points_main_PbPb-2760.dat   # usado na célula 1
    design_ranges_main_PbPb-2760.dat
    design_pts_main_PbPb-2760.dat      # usado nas células 2 e 3
```

---

**Pastas de output criadas automaticamente pelo código (não precisam existir antes):**
```
predictions_global_v0_vs_exp/
predictions_v0pt_vs_exp/
```
