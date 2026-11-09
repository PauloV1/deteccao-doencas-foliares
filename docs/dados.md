# Dados

| Base | Uso | Conteúdo |
|---|---|---|
| PlantVillage | treino e teste principal | 54.305 fotos de folhas em fundo uniforme, 38 classes (14 culturas); split 80/20 por imagem |
| PlantDoc | campo: treino e teste | 2.342 fotos de treino e 236 de teste, 27 classes mapeadas para o PlantVillage |
| PlantWild (release 1) | campo: treino, validação e teste | 18.542 fotos da web em 89 classes, com split oficial; só as classes mapeadas entram |

As bases são baixadas de referências fixas (tag do PlantVillage, commit do PlantDoc, revisão do PlantWild); `fontes.json` registra a URL e o SHA-256 de cada arquivo.

## Classes
As 38 classes do PlantVillage ocupam os índices 0–37. PlantDoc e PlantWild são mapeados às classes do PlantVillage à mão (`folhas/catalogo.py`); o mapeamento pode ter ruído.
O modelo `brasil` acrescenta 15 classes do PlantWild sem equivalente no PlantVillage: arroz (saudável, brusone, mancha-da-bainha), café (saudável, ferrugem), feijão (saudável, crestamento de halo, mosaico, ferrugem), banana (saudável, mal-do-Panamá), citros (cancro), fumo (saudável, mosaico) e carvão do milho. Hortaliças (alface, alho, pepino, berinjela) podem ser ligadas com `br_hortalicas=True`.

## Lacunas
- Soja só tem a classe saudável, e não há dados para cana-de-açúcar, trigo e algodão.
- As culturas novas só têm fotos da web, e o teste vem da mesma fonte: é otimista para fotos de lavoura.
- A base Digipathos (Embrapa) foi avaliada e não é usada: boa parte das fotos é de laboratório, sobre placa de calibração ou fundo de scanner, com poucas imagens por classe e recortes sem como agrupar por origem; isso induziria o atalho "fundo = classe".
