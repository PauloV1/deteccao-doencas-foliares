# Protocolo

## Modelo e treino
ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, timm), 224 px, rede inteira treinável, AdamW + cosseno com aquecimento, label smoothing 0,1, precisão mista. Aumento de dados em lote na GPU: recorte, rotação ±20°, flips e cor. Em metade das fotos do PlantVillage o fundo é trocado: a folha é segmentada (fundo = região de cor de fundo ligada à borda) e colada sobre recortes de fotos de campo de treino ou manchas sintéticas.

## Conjuntos

| Conjunto | Papel |
|---|---|
| PlantVillage treino | 96% treina; 4% (estratificado) é a validação que escolhe a época nas etapas só com PlantVillage |
| PlantVillage teste | só avalia (split do repositório) |
| PlantDoc treino / teste | o treino entra nas etapas de campo; o teste só avalia |
| PlantWild (classes mapeadas) | split oficial: o treino treina, a validação escolhe a época das etapas de campo, o teste só avalia |

Nada do teste é usado para treinar, escolher época, ponto de partida ou hiperparâmetro.

## Etapas (runs)

| Run | O que é |
|---|---|
| `baseline` | PlantVillage, amostragem natural |
| `balanceado` | PlantVillage, classes equiprováveis |
| `sem_troca` | `balanceado` sem troca de fundo |
| `campo_pd` | a partir do `baseline`: PlantDoc (50%) + PlantVillage |
| `campo_todos` | a partir do `baseline`: PlantVillage 40% + PlantDoc 15% + PlantWild 45% |
| `campo_cont` | controle do `brasil`: continua o `campo_todos` com o mesmo treino extra, só nas 38 classes |
| `brasil` | continua o `campo_todos` com a cabeça ampliada (38 + 15 classes novas); **é o modelo exportado** |
| `<run>_s1`, `_s2` | outras sementes de `baseline`, `balanceado` e `sem_troca` |

O ponto de partida das etapas de campo (`campo_pai`) e o modelo final são fixados antes de ver resultados. `campo_pd` e `campo_todos` têm o mesmo orçamento de treino; `campo_cont` e `brasil` também.

## Vazamento treino → teste
O PlantVillage é dividido por imagem (não por folha) e PlantDoc e PlantWild vêm de buscas na web, então há imagens de teste quase idênticas a imagens de treino. Cópias exatas (hash) são removidas e, para o resto, cada imagem de teste ganha uma marca de vazamento: similaridade de cosseno ≥ 0,95 em features de uma ConvNeXt-Tiny só com pesos ImageNet (128 px) com alguma imagem de treino e, no PlantVillage, duplicata por pixel invariante a flip e rotação de 90°. Todas as métricas saem também "sem vazamento".

## Métricas e estatística
- Acurácia nos três testes (as métricas do PlantWild usam as 38 classes comuns a todos os modelos), recall das 10 classes mais raras, macro-F1 (só sobre as classes presentes no gabarito), ECE (calibração) e, no `brasil`, acerto nas classes novas e fração de folhas de laboratório classificadas como cultura nova.
- Cada modificação é comparada com o seu controle por diferença pareada, com IC95% por bootstrap sobre as imagens de teste. Veredito "melhora/piora" só quando o IC exclui zero.
- O IC trata as imagens como independentes (imagens quase idênticas não são), não cobre a variação entre treinos e não corrige múltiplas comparações; o desvio entre sementes (`sementes.csv`) dá a escala do ruído de treino.

## Saídas (`MyDrive/deteccao_folhas_saidas/`)

| Arquivo | Conteúdo |
|---|---|
| `todos_os_modelos.csv` | uma linha por run, com as métricas abaixo |
| `comparacao_modificacoes.csv` | `modificação, tratamento, controle, teste, n, controle %, tratamento %, Δ pp, IC95 inf, IC95 sup, veredito` |
| `sementes.csv`, `ablacao_troca_fundo.csv`, `tempos.csv` | média e desvio entre sementes; troca de fundo; minutos de treino por run |
| `ambiente.json`, `config.json`, `fontes.json` | versões das bibliotecas e GPU; configuração usada; URL e SHA-256 de cada base baixada |
| `mascaras_vazamento.npz`, `aumento.png`, `troca_de_fundo.png`, `comparacao_modelos.png` | marcas de vazamento dos três testes; figuras |
| `<run>/` | `best.pt`, `done.json` (histórico por época), `metrics.json`, `ok.npz` (acertos por amostra), `matriz_confusao.png` |
| `model.pt`, `model_config.json` | modelo exportado (`brasil`) e sua lista de classes |

Colunas de métrica (frações de 0 a 1; o prefixo indica o teste: `pv` PlantVillage, `pd` PlantDoc, `pw` PlantWild):

| Coluna | Significado |
|---|---|
| `pv_acc`, `pd_top1`, `pw_top1`, `*_top5` | acurácia top-1 e top-5 (`pw_*` usa as 38 classes comuns a todos os modelos) |
| `*_acc_clean` | acurácia sem as imagens de teste vazadas |
| `pv_macro_f1`, `pd_macro_f1`, `pw_macro_f1` | macro-F1 sobre as classes presentes no gabarito |
| `pv_minority10_recall` | recall médio das 10 classes com menos imagens de treino |
| `*_ece` | erro de calibração esperado (menor é melhor) |
| `*_healthy_vs_sick` | acerto em saudável × doente |
| `pw_all_top1`, `new_top1`, `new_n`, `field_macro_f1` | só no `brasil`: acurácia do PlantWild com as 53 classes; acerto e número de imagens nas classes novas; macro-F1 com as 53 classes |
| `pv_leak_to_new` | só no `brasil`: fração das folhas do teste do PlantVillage classificadas como cultura nova |
