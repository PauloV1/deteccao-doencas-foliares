# Detecção de doenças em folhas

Classifica doenças foliares por imagem com uma ConvNeXt-Tiny pré-treinada, ajustada em etapas (PlantVillage → fotos de campo → culturas do Brasil). Inclui uma API de inferência com Grad-CAM e uma página web de upload. Projeto da disciplina *Tendências em Inteligência Computacional*.

> Ferramenta de apoio; não substitui a avaliação de um engenheiro agrônomo.

## Estrutura

```
folhas/      pacote: dados, aumento, treino, avaliação, vazamento, comparação, pipeline
notebooks/   treino.ipynb — só chama funções de folhas/
api/         FastAPI + página web
tests/       testes automatizados
docs/        protocolo e dados
scripts/     empacotamento
models/      model.pt e model_config.json (gerados pelo treino)
```

## Treinar (Google Colab, GPU T4)

1. Envie `deteccao_folhas.zip` para a raiz do Google Drive.
2. Abra `notebooks/treino.ipynb` no Colab, selecione GPU T4 e execute tudo (≈ 2,5 h). Se a sessão cair, reexecute: o treino retoma dos checkpoints.
3. Os resultados ficam em `MyDrive/deteccao_folhas_saidas/` (≈ 1,5 GB, pelos pesos de cerca de 13 runs): tabelas e figuras, métricas por modelo, `ambiente.json`, `fontes.json`, `model.pt` e `model_config.json`. A lista de arquivos e a legenda das colunas estão em [`docs/protocolo.md`](docs/protocolo.md).

## Usar o modelo

```bash
pip install -r requirements.txt
cp <saidas>/model.pt <saidas>/model_config.json models/
uvicorn api.app:app --port 8000        # http://localhost:8000 (documentação interativa em /docs)
```

`POST /predict` recebe a imagem (`file`; query `gradcam=true|false`, `top_k`) e devolve cultura, condição, top-k, `confiavel` (confiança ≥ 60%) e o Grad-CAM em base64. Limites: 10 MB e 40 megapixels; a orientação EXIF é respeitada. A API não tem autenticação nem limite de requisições: não a exponha na internet.

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Resultados

Modelo em `models/` (53 classes: as 38 do PlantVillage + 15 culturas do Brasil, 224 px), treinado no Colab em 2,5 h. Acurácia no teste (%):

| Run | PlantVillage | PlantDoc (sem vazamento) | PlantWild, 38 cl. (sem vazamento) | Classes novas |
|---|---|---|---|---|
| `baseline` | 99,42 | 41,1 | 30,9 | — |
| `campo_pd` | 99,26 | 74,2 | 51,8 | — |
| `campo_todos` | 99,29 | 76,1 | 74,8 | — |
| `campo_cont` | 99,34 | 78,0 | 76,4 | — |
| **`brasil`** (modelo exportado) | 99,27 | 76,6 | 75,6 | 80,6 |

- Adaptar com fotos de campo é o que importa: PlantDoc 41,2% → 76,1% e PlantWild 30,9% → 74,8%, com perda de ~0,15 pp no PlantVillage (≥ 99,26% em todos os modelos).
- Incluir as culturas do Brasil não mostrou custo detectável nas 38 classes antigas; as culturas novas acertam 80,6%, mas feijão e fumo são fracos.
- Amostragem balanceada: +0,43 pp de recall nas classes raras por −0,14 pp de acurácia. Troca de fundo: −0,31 pp no PlantVillage, +3,4 pp no PlantWild.
- Com uma semente nas etapas de campo, diferenças menores que ~3 pp no PlantDoc não são interpretáveis.

Tabelas completas, comparações e ressalvas: [`docs/resultados.md`](docs/resultados.md). Arquivos brutos em `resultados/`.

## Mais detalhes

- [`docs/protocolo.md`](docs/protocolo.md): modelo, conjuntos, etapas, controles, vazamento, estatística e legenda das saídas.
- [`docs/resultados.md`](docs/resultados.md): resultados e análise.
- [`docs/dados.md`](docs/dados.md): bases, mapeamento de classes e lacunas.

## Limitações

- Os testes de campo são fotos da web (PlantDoc, PlantWild); não há fotos de lavoura tiradas por produtores.
- Soja só tem a classe saudável; não há dados para cana, trigo e algodão.
- A API sempre devolve alguma doença, mesmo para imagens que não são folhas; o limiar de 60% não é calibrado.
- Os intervalos de confiança cobrem o sorteio das imagens de teste, não a variação entre treinos (medida pelas sementes).

## Licenças

O código é distribuído sob a licença MIT ([`LICENSE`](LICENSE)). As bases e os pesos iniciais têm licenças próprias:

| Recurso | Licença |
|---|---|
| [PlantVillage](https://github.com/attaullah/downsampled-plant-disease-dataset) | CC0 1.0 (segundo o repositório usado) |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | CC BY 4.0 |
| [PlantWild](https://huggingface.co/datasets/uqtwei2/PlantWild) | CC BY-NC-ND 4.0 (uso não comercial, sem derivações) |
| `convnext_tiny.fb_in22k_ft_in1k` (timm) | Apache-2.0 |

O PlantWild limita o uso a fins acadêmicos e torna incerta a publicação dos pesos treinados com ele.
