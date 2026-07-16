# Processos Estocásticos

Repositório com análises de sistemas LIT discretos excitados por processos
estocásticos: caracterização no tempo e na frequência, identificação de modelos,
validação e detecção de anomalias.

## Estrutura

```
stochastic_processes/
├── README.md
├── requirements.txt
├── trabalho_1/                 # Trabalho 1 (concluído)
│   ├── enunciado.pdf
│   ├── dados/                  # CSVs de treino, validação e teste
│   ├── notebooks/              # análise em Jupyter
│   └── relatorio/              # fonte LaTeX e PDF
└── trabalho_2/                 # Trabalho 2 (em preparação)
    ├── dados/
    ├── notebooks/
    └── relatorio/
```

## Trabalho 1

Caracterização, identificação e validação de um sistema LIT discreto com entrada
branca e colorida, além de detecção de anomalias via resíduos.

| Artefato | Caminho |
|----------|---------|
| Enunciado | `trabalho_1/enunciado.pdf` |
| Dados | `trabalho_1/dados/` |
| Caderno principal | `trabalho_1/notebooks/resolucao_trabalho.ipynb` |
| Relatório PDF | `trabalho_1/relatorio/relatorio.pdf` |

### Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (PowerShell)
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
jupyter notebook trabalho_1/notebooks/resolucao_trabalho.ipynb
```

Para recompilar o relatório LaTeX (opcional):

```bash
python trabalho_1/relatorio/gerar_figuras.py
cd trabalho_1/relatorio
pdflatex relatorio.tex
pdflatex relatorio.tex
```

## Trabalho 2

Estrutura criada em `trabalho_2/`. Ver `trabalho_2/README.md`.

## Ferramentas

- `numpy` / `pandas` — manipulação numérica e CSVs
- `scipy.signal` — correlação, PSD (Welch), espectro cruzado e coerência
- `matplotlib` — gráficos
- `statsmodels` / `scipy` — identificação de modelos e resíduos
