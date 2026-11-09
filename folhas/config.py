"""Configuração do experimento. Os valores padrão são o protocolo completo."""
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

NO_COLAB = Path("/content").exists()


@dataclass
class Config:
    # imagem: resolução de treino e do cache de imagens
    tamanho: int = 224
    # onde ficam os dados/cache (local, rápido) e os resultados (no Colab: Drive, para sobreviver à sessão)
    trabalho: str = "/content/trabalho" if NO_COLAB else "trabalho"
    saidas: str = "/content/drive/MyDrive/deteccao_folhas_saidas" if NO_COLAB else "saidas"
    # pastas dos dados já baixados; None = baixar em `trabalho`
    pv_dir: str | None = None
    pd_dir: str | None = None
    pw_dir: str | None = None

    # etapas com PlantVillage (baseline, balanced)
    epocas: int = 8
    amostras: int = 20000          # amostras sorteadas por época
    lote: int = 64
    lr: float = 1e-4
    congelar: int = 0              # estágios iniciais congelados (0 = rede inteira treinável)
    fundo_p: float = 0.5           # prob. de trocar o fundo de cada foto do PlantVillage no treino

    # adaptação ao campo
    campo_pai: str = "baseline"    # ponto de partida, fixado a priori: 'baseline' | 'balanced'
    campo_lr: float = 5e-5
    campo_frac: float = 0.5        # campo_pd: fração de cada lote vinda do PlantDoc
    campo_mesmo_orcamento: bool = True   # campo_pd treina o mesmo que campo_todos
    campo_epocas: int = 3          # usados só se campo_mesmo_orcamento=False
    campo_amostras: int = 10000
    usar_plantwild: bool = True
    todos_epocas: int = 4
    todos_amostras: int = 20000
    todos_mix: tuple = (0.40, 0.15, 0.45)   # campo_todos: PlantVillage / PlantDoc / PlantWild

    # culturas do Brasil (classes novas, vindas do PlantWild)
    usar_br: bool = True
    br_hortalicas: bool = False
    br_epocas: int = 4
    br_amostras: int = 20000
    br_mix: tuple = (0.30, 0.70)   # PlantVillage (replay) / campo
    br_pot: float = 0.5            # amostragem de campo ∝ n_classe^-br_pot
    br_controle: bool = True       # campo_cont: mesmo treino extra do `brasil`, só nas 38 classes

    # análises
    rodar_sementes: bool = True
    sementes: tuple = (1, 2)
    ablacao_fundo: bool = True     # sem_troca: mede o efeito da troca de fundo
    limiar_vazamento: float = 0.95  # similaridade a partir da qual uma imagem de teste conta como vazada
    exigir_gpu: bool = True
    limite_avaliacao: int | None = None   # só para testes rápidos

    def __post_init__(self):
        # FOLHAS_CONFIG (JSON) sobrescreve campos; usado em testes automatizados
        for k, v in json.loads(os.environ.get("FOLHAS_CONFIG", "{}")).items():
            setattr(self, k, tuple(v) if isinstance(getattr(self, k), tuple) else v)
        if self.usar_br and not self.usar_plantwild:
            raise ValueError("usar_br exige usar_plantwild=True (as culturas novas vêm do PlantWild)")
        if self.campo_pai not in ("baseline", "balanced"):
            raise ValueError("campo_pai deve ser 'baseline' ou 'balanced'")

    @property
    def cache_dir(self) -> Path:
        return Path(self.trabalho) / "cache"

    @property
    def saidas_dir(self) -> Path:
        return Path(self.saidas)

    def como_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
