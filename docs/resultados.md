# Resultados

Treino no Google Colab (Tesla T4; 2,5 h de treino no total) com o protocolo completo e os valores padrão de `Config`; uma semente nas etapas de campo e três (0, 1, 2) nas etapas só com PlantVillage. Os arquivos estão em `resultados/`; o modelo exportado (`brasil`) é o `models/model.pt`. Metodologia em [`protocolo.md`](protocolo.md).

## Modelos (teste, %)

| Run | PlantVillage | sem vazamento | Recall 10 raras | PlantDoc (sem vazamento) | PlantWild, 38 cl. (sem vazamento) | ECE PlantDoc / PlantWild | Classes novas |
|---|---|---|---|---|---|---|---|
| `baseline` | 99,42 | 99,36 | 99,09 | 42,4 (41,1) | 31,3 (30,9) | 20,3 / 17,9 | — |
| `balanceado` | 99,30 | 99,23 | 99,47 | 41,9 (42,1) | 29,6 (29,1) | 18,2 / 19,6 | — |
| `sem_troca` | 99,64 | 99,61 | 99,57 | 40,3 (40,2) | 26,4 (25,7) | 19,7 / 19,7 | — |
| `campo_pd` | 99,26 | 99,20 | 98,81 | 72,0 (74,2) | 53,0 (51,8) | 8,0 / 11,9 | — |
| `campo_todos` | 99,29 | 99,21 | 98,98 | 74,6 (76,1) | 74,9 (74,8) | 8,2 / 4,0 | — |
| `campo_cont` | 99,34 | 99,25 | 98,90 | 75,8 (78,0) | 76,6 (76,4) | 7,2 / 3,1 | — |
| **`brasil`** (modelo exportado) | 99,27 | 99,19 | 99,39 | 74,2 (76,6) | 75,5 (75,6) | 6,1 / 3,2 | 80,6 |

Testes: PlantVillage 10.849 imagens (9.404 sem vazamento), PlantDoc 236 (209), PlantWild 1.460 das 38 classes comuns (1.362). Recall das 10 raras = classes com menos imagens de treino. ECE = erro de calibração esperado.

## Ruído entre treinos (média ± desvio, %)

| Run (3 sementes) | PlantVillage | PV macro-F1 | Recall 10 raras | PlantDoc | PlantWild (38 cl.) |
|---|---|---|---|---|---|
| `baseline` | 99,47 ± 0,06 | 99,37 ± 0,08 | 99,10 ± 0,02 | 40,68 ± 3,31 | 32,03 ± 0,65 |
| `balanceado` | 99,33 ± 0,03 | 99,27 ± 0,02 | 99,53 ± 0,07 | 42,23 ± 0,88 | 30,71 ± 1,00 |
| `sem_troca` | 99,64 ± 0,01 | 99,62 ± 0,01 | 99,66 ± 0,08 | 39,83 ± 1,53 | 27,33 ± 0,87 |

## Cada modificação contra o seu controle

Diferença pareada em pontos percentuais, IC95% por bootstrap sobre as imagens de teste; "melhora/piora" só quando o IC exclui zero. Tabela completa em `resultados/comparacao_modificacoes.csv`.

| Modificação | Tratamento × controle | Teste | n | Controle → Tratamento | Δ pp [IC95%] | Veredito |
|---|---|---|---|---|---|---|
| 1, amostragem balanceada | `balanceado` × `baseline` | PlantVillage | 10849 | 99,4 → 99,3 | -0,12 [-0,27; +0,02] | sem diferença clara |
| 1, amostragem balanceada | `balanceado` × `baseline` | PlantDoc sem vazamento | 209 | 41,1 → 42,1 | +0,96 [-3,35; +5,74] | sem diferença clara |
| 1, amostragem balanceada | `balanceado` × `baseline` | PlantWild sem vazamento | 1362 | 30,9 → 29,1 | -1,84 [-3,67; -0,22] | piora |
| 2, troca de fundo | `balanceado` × `sem_troca` | PlantVillage | 10849 | 99,6 → 99,3 | -0,34 [-0,48; -0,21] | piora |
| 2, troca de fundo | `balanceado` × `sem_troca` | PlantDoc sem vazamento | 209 | 40,2 → 42,1 | +1,91 [-3,35; +7,18] | sem diferença clara |
| 2, troca de fundo | `balanceado` × `sem_troca` | PlantWild sem vazamento | 1362 | 25,7 → 29,1 | +3,38 [+1,54; +5,21] | melhora |
| 3, + PlantDoc | `campo_pd` × `baseline` | PlantVillage | 10849 | 99,4 → 99,3 | -0,16 [-0,27; -0,05] | piora |
| 3, + PlantDoc | `campo_pd` × `baseline` | PlantDoc sem vazamento | 209 | 41,1 → 74,2 | +33,01 [+26,78; +39,71] | melhora |
| 3, + PlantDoc | `campo_pd` × `baseline` | PlantWild sem vazamento | 1362 | 30,9 → 51,8 | +20,85 [+18,14; +23,72] | melhora |
| 4, + PlantWild (mesmo orçamento) | `campo_todos` × `campo_pd` | PlantVillage | 10849 | 99,3 → 99,3 | +0,03 [-0,11; +0,16] | sem diferença clara |
| 4, + PlantWild (mesmo orçamento) | `campo_todos` × `campo_pd` | PlantDoc sem vazamento | 209 | 74,2 → 76,1 | +1,91 [-2,87; +6,22] | sem diferença clara |
| 4, + PlantWild (mesmo orçamento) | `campo_todos` × `campo_pd` | PlantWild sem vazamento | 1362 | 51,8 → 74,8 | +23,05 [+20,34; +25,70] | melhora |
| 5a, culturas do Brasil vs, mesmo treino sem classes novas | `brasil` × `campo_cont` | PlantVillage | 10849 | 99,3 → 99,3 | -0,06 [-0,18; +0,05] | sem diferença clara |
| 5a, culturas do Brasil vs, mesmo treino sem classes novas | `brasil` × `campo_cont` | PlantDoc sem vazamento | 209 | 78,0 → 76,6 | -1,44 [-4,78; +1,91] | sem diferença clara |
| 5a, culturas do Brasil vs, mesmo treino sem classes novas | `brasil` × `campo_cont` | PlantWild sem vazamento | 1362 | 76,4 → 75,6 | -0,88 [-2,13; +0,44] | sem diferença clara |
| 5b, culturas do Brasil vs, ponto de partida | `brasil` × `campo_todos` | PlantVillage | 10849 | 99,3 → 99,3 | -0,02 [-0,16; +0,10] | sem diferença clara |
| 5b, culturas do Brasil vs, ponto de partida | `brasil` × `campo_todos` | PlantDoc sem vazamento | 209 | 76,1 → 76,6 | +0,48 [-3,35; +4,31] | sem diferença clara |
| 5b, culturas do Brasil vs, ponto de partida | `brasil` × `campo_todos` | PlantWild sem vazamento | 1362 | 74,8 → 75,6 | +0,73 [-0,73; +2,20] | sem diferença clara |
| efeito total da adaptação ao campo | `campo_todos` × `baseline` | PlantVillage | 10849 | 99,4 → 99,3 | -0,13 [-0,28; +0,02] | sem diferença clara |
| efeito total da adaptação ao campo | `campo_todos` × `baseline` | PlantDoc sem vazamento | 209 | 41,1 → 76,1 | +34,93 [+27,75; +42,11] | melhora |
| efeito total da adaptação ao campo | `campo_todos` × `baseline` | PlantWild sem vazamento | 1362 | 30,9 → 74,8 | +43,91 [+41,04; +46,70] | melhora |

## Culturas do Brasil (modelo `brasil`)

Acerto nas classes novas: **80,6%** (603 imagens do teste do PlantWild); macro-F1 com as 53 classes no teste de campo: 74,5; folhas de laboratório classificadas como cultura nova: **0,0%**.

| Classe nova | Recall |
|---|---|
| Bean___Halo_blight | 28,6% |
| Tobacco___Mosaic_virus | 41,2% |
| Bean___healthy | 55,3% |
| Bean___Mosaic_virus | 64,0% |
| Bean___Rust | 69,7% |
| Tobacco___healthy | 71,4% |
| Rice___healthy | 74,0% |
| Rice___Blast | 75,9% |
| Coffee___healthy | 85,2% |
| Citrus___Canker | 85,7% |
| Rice___Sheath_blight | 92,0% |
| Banana___healthy | 93,8% |
| Banana___Panama_disease | 97,6% |
| Corn_(maize)___Smut | 98,3% |
| Coffee___Leaf_rust | 100,0% |

## O que os números sustentam

1. **Adaptar com fotos de campo é o efeito dominante.** `campo_todos` × `baseline`: PlantDoc 41,2% → 76,1% (+34,9 pp; IC +27,8 a +42,1) e PlantWild 30,9% → 74,8% (+43,9 pp), com variação de −0,15 pp no PlantVillage (não significativa). Só as fotos do PlantDoc (`campo_pd`) já dão +33,0 pp no PlantDoc, ao custo pequeno de −0,16 pp no PlantVillage (IC exclui zero).
2. **O PlantWild ajuda no seu próprio teste, não demonstradamente fora dele.** Com o mesmo orçamento, `campo_todos` × `campo_pd`: PlantWild +23,1 pp (mesma fonte no treino e no teste), PlantDoc +1,9 pp (IC −2,9 a +6,2).
3. **Incluir as culturas do Brasil não mostrou custo detectável nas 38 classes antigas.** `brasil` × `campo_cont` (mesmo treino extra): PlantVillage −0,06 pp, PlantDoc −1,4 pp, PlantWild −0,9 pp (IC −2,1 a +0,4), todos sem diferença clara. As classes novas são desiguais: café-ferrugem 100%, mal-do-Panamá 97,6%, carvão do milho 98,3%, mancha-da-bainha do arroz 92,0%; feijão (28,6% a 69,7%) e fumo (41,2% e 71,4%) são fracos, com poucas imagens por classe.
4. **Amostragem balanceada (3 sementes) desloca recall para as classes raras, com custo pequeno.** Recall das 10 raras +0,43 ± 0,04 pp e acurácia no PlantVillage −0,14 ± 0,04 pp; sem ganho em campo (PlantDoc +1,6 ± 2,0; PlantWild −1,3 ± 0,7).
5. **A troca de fundo é uma troca de lugar (3 sementes de cada lado).** PlantVillage −0,31 ± 0,02 pp contra PlantWild +3,4 ± 0,8 e PlantDoc +2,4 ± 1,0 (erro-padrão; t de Welch indicativo com n = 3: −20, +4,4 e +2,4). Foi mantida no modelo final porque o ganho em campo supera a perda no laboratório; o efeito sobre os modelos finais, ajustados com fotos de campo, não foi medido.
6. **Calibração.** O ECE cai de ~20% (PlantDoc) e ~18% (PlantWild) no `baseline` para 6,1% e 3,3% no `brasil`. O limiar de 60% da API não foi calibrado.
7. **Erros em campo.** No PlantDoc inteiro (236 imagens, 74,2%) há 61 erros: 40 acertam a cultura e erram a doença (milho: mancha-de-turcicum × cercosporiose; maçã: ferrugem × sarna; tomate: manchas e mofo) e 14 acertam o nome da doença e erram a cultura: 9 são folhas saudáveis de uma espécie tomadas pela saudável de outra (maçã e pêssego → cereja; soja → feijão, framboesa e mirtilo) e 5 são a mesma doença em outra cultura (requeima e pinta-preta entre batata e tomate, mancha bacteriana do tomate → pimentão, oídio da abóbora → cereja). No PlantWild as classes mais fracas são crestamento de halo do feijão (28,6%), oídio da cereja (31,8%), septoriose (35,7%) e mosaico (36,1%) do tomate.
8. **O PlantVillage está saturado:** ≥ 99,26% em todos os modelos e F1 por classe ≥ 0,945 (menor: pinta-preta do tomate). Tirar as imagens vazadas muda a acurácia em ≤ 0,1 pp.

## Limitações

- As etapas de campo têm uma semente. A validação de campo do `brasil` é praticamente plana entre as épocas 2 a 4 (macro-F1 74,9 a 75,1), então a escolha da época é quase arbitrária; diferenças menores que ~3 pp no PlantDoc não são interpretáveis.
- O PlantDoc tem 209 imagens sem vazamento (IC largo). PlantWild e as classes novas são fotos da web com teste da mesma fonte: otimista para fotos de lavoura.
- Os IC não incluem a variação entre treinos nem corrigem as 42 comparações (esperam-se ~2 "melhora/piora" por acaso); os mapeamentos de classes foram feitos à mão e podem ter ruído.
