# Trabalho 2 — Processos Estocásticos

Detecção e estimação de pulsos impulsivos imersos em ruído, com filtro de Wiener,
filtro casado e estimador BLUE.

## Estrutura

```
trabalho_2/
├── enunciado.pdf
├── dados/                      # CSVs + zip original
├── notebooks/
│   └── resolucao_trabalho.ipynb
└── relatorio/
    ├── relatorio.tex
    ├── relatorio.pdf
    ├── gerar_figuras.py
    ├── valores.tex
    ├── secoes/
    ├── figuras/
    └── imagens/
```

## Como reproduzir

```bash
# a partir da raiz do repositório
pip install -r requirements.txt
jupyter notebook trabalho_2/notebooks/resolucao_trabalho.ipynb
```

Para regenerar figuras e recompilar o PDF:

```bash
python trabalho_2/relatorio/gerar_figuras.py
cd trabalho_2/relatorio
pdflatex relatorio.tex
pdflatex relatorio.tex
```

## Saídas

- Relatório: `relatorio/relatorio.pdf`
- Caderno: `notebooks/resolucao_trabalho.ipynb`
