# Processos Estocásticos

Repositório com análises de sistemas e sinais à luz de processos estocásticos:
caracterização no tempo e na frequência, identificação de modelos, filtragem,
detecção e estimação.

## Estrutura

```
stochastic_processes/
├── README.md
├── requirements.txt
├── trabalho_1/                 # Trabalho 1
│   ├── enunciado.pdf
│   ├── dados/
│   ├── notebooks/
│   └── relatorio/
└── trabalho_2/                 # Trabalho 2
    ├── enunciado.pdf
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
| Caderno | `trabalho_1/notebooks/resolucao_trabalho.ipynb` |
| Relatório | `trabalho_1/relatorio/relatorio.pdf` |

## Trabalho 2

Detecção e estimação de pulsos impulsivos em ruído (Wiener, filtro casado, BLUE).

| Artefato | Caminho |
|----------|---------|
| Enunciado | `trabalho_2/enunciado.pdf` |
| Caderno | `trabalho_2/notebooks/resolucao_trabalho.ipynb` |
| Relatório | `trabalho_2/relatorio/relatorio.pdf` |

Detalhes em [`trabalho_2/README.md`](trabalho_2/README.md).

## Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (PowerShell)
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

Em seguida, abra o caderno do trabalho desejado em `trabalho_1/notebooks/` ou
`trabalho_2/notebooks/`.

## Ferramentas

- `numpy` / `pandas` — manipulação numérica e CSVs
- `scipy.signal` — correlação, PSD (Welch), filtragem
- `matplotlib` — gráficos
- `statsmodels` / `scipy` — modelos e resíduos (Trabalho 1)
