# Trabalho 1 — Processos Estocásticos

Caracterização de sinais, estimação da relação entrada–saída e identificação de um
sistema **LIT (Linear e Invariante no Tempo) discreto** excitado por processos
estocásticos. Os dados foram gerados pelo professor a partir de um sistema LIT e o
objetivo é caracterizá-los nos domínios do **tempo** e da **frequência**, identificar
um modelo, validá-lo criticamente e analisar a presença de anomalias.

## Objetivo

A partir dos sinais de entrada `u[n]` e saída `y[n]`, o trabalho consiste em:

1. Caracterizar estatisticamente os sinais (média, variância, autocorrelação).
2. Estimar a relação entrada–saída no domínio da frequência (PSD, espectro cruzado, coerência).
3. Estimar a resposta em frequência do sistema por dois métodos e compará-los.
4. Identificar pelo menos um modelo LIT (FIR, ARX ou espaço de estados).
5. Validar o modelo em um conjunto independente.
6. Analisar resíduos e detectar anomalias.

## Dados

Os conjuntos de dados estão em `dados_processos_estocasticos.zip` e, quando extraídos,
geram quatro arquivos CSV. Todos possuem três colunas:

| Coluna | Descrição |
|--------|-----------|
| `n`    | Índice de tempo discreto (amostra) |
| `u`    | Sinal de entrada `u[n]` |
| `y`    | Sinal de saída `y[n]` |

| Arquivo | Amostras | Uso |
|---------|----------|-----|
| `dados_treino_branco.csv`    | 5000 | Treino — entrada aproximadamente **branca** |
| `dados_validacao_branco.csv` | 3000 | **Validação** do modelo identificado |
| `dados_treino_colorido.csv`  | 5000 | Treino — entrada **colorida** (não excita todo o espectro) |
| `dados_teste_anomalia.csv`   | 5000 | **Teste** de detecção de anomalia |

## Tarefas

### 1. Entrada branca (`dados_treino_branco.csv`)

1. Estimar média, variância e autocorrelação de `u[n]` e `y[n]`.
2. Estimar a correlação cruzada entre `u[n]` e `y[n]`.
3. Estimar a PSD (densidade espectral de potência) de `u[n]` e `y[n]`.
4. Estimar o espectro cruzado entre `u[n]` e `y[n]`.
5. Estimar a **coerência** entrada–saída, interpretando em quais faixas de frequência
   a relação pode ser considerada mais confiável.
6. Estimar a resposta em frequência do sistema por **dois métodos** e comparar:
   - Método 1: estimativa aproximada do **módulo** (adequado para entrada branca e
     ruído de saída não dominante).
   - Método 2: estimativa **complexa** (módulo e fase), analisada em conjunto com a coerência.
7. Identificar pelo menos um modelo LIT (ex.: FIR, ARX ou espaço de estados).
8. Validar o modelo com `dados_validacao_branco.csv`: aplicar `uval[n]` para obter
   `ŷval[n]` e comparar com `yval[n]` verdadeiro.

### 2. Entrada colorida (`dados_treino_colorido.csv`)

1. Repetir a análise temporal e espectral feita para o caso de entrada branca.
2. Comparar os resultados com o caso de entrada branca.
3. Discutir por que a identificação pode piorar em algumas faixas de frequência quando
   a entrada não excita adequadamente todo o espectro.

### 3. Detecção de anomalia (`dados_teste_anomalia.csv`)

1. Usar o modelo identificado a partir dos dados normais.
2. Calcular os resíduos `e[n] = y[n] - ŷ[n]`.
3. Analisar média, variância, autocorrelação e energia local dos resíduos.
4. Detectar possíveis mudanças de regime, aumento do erro ou impulsos.
5. Discutir se a anomalia aparece mais claramente no sinal de saída ou nos resíduos.

## Estrutura do projeto

```
stochastic_processes/
├── README.md
├── requirements.txt                      # dependências Python
├── dados_processos_estocasticos.zip      # dados originais (compactados)
├── Trabalho 1 - Processos Estocásticos.pdf
├── data/                                 # CSVs extraídos
├── src/                                  # código-fonte da análise
├── results/                              # gráficos e arquivos de saída gerados
└── relatorio/                            # relatório em PDF
```

> A estrutura acima é o destino planejado; pastas e arquivos serão criados conforme o
> trabalho avança.

## Como reproduzir

1. Criar e ativar um ambiente virtual Python:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (PowerShell)
# source .venv/bin/activate # Linux/macOS
```

2. Instalar as dependências:

```bash
pip install -r requirements.txt
```

3. Extrair os dados (caso ainda não estejam em `data/`):

```bash
tar -xf dados_processos_estocasticos.zip
```

4. Executar os scripts de análise em `src/` (a serem criados).

## Ferramentas

Análise em **Python**, utilizando:

- `numpy` / `pandas` — manipulação numérica e dos CSVs.
- `scipy.signal` — correlação, PSD (Welch), espectro cruzado (CSD) e coerência.
- `matplotlib` — geração dos gráficos.
- `statsmodels` / `scipy` — identificação de modelos (FIR/ARX) e análise de resíduos.

## Entrega

- Relatório em PDF contendo gráficos, estimativas, descrição dos métodos e
  **interpretação crítica** dos resultados.
- Códigos utilizados.
- Arquivos de saída relevantes.

O relatório deve permitir a **reprodução** dos resultados, discutindo como
autocorrelação, PSD, espectro cruzado, coerência e resíduos ajudam a avaliar a
qualidade e a confiabilidade do modelo identificado.
