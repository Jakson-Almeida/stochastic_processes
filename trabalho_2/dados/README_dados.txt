

Esta pasta possui:
- dados de ruído para caracterização estatística;
- conjunto de treino rotulado para projetar filtros e ajustar limiares;
- conjunto de teste rotulado para medir o desempenho final.

Parâmetros:
- cada janela tem 256 amostras;
- frequência de amostragem: 1000 Hz;
- o template do pulso tem 64 amostras;
- o pulso pode aparecer em diferentes posições dentro da janela;

Arquivos:

1. template_pulso.csv
   Forma de onda esperada do pulso s[n].
   Colunas: n, t_s, s.

2. ruido_treino.csv
   300 janelas contendo apenas ruído.
   Deve ser usado para estimar média, variância, autocorrelação, PSD, covariância e limiares iniciais.
   Colunas: record_id, n, t_s, y.

3. sinais_treino.csv
   500 janelas que podem conter ou não um pulso.
   Deve ser usado para desenvolver os filtros, testar alternativas, ajustar parâmetros e escolher limiares.
   Colunas: record_id, n, t_s, y.

4. rotulos_treino.csv
   Rótulos do conjunto de treino.
   Colunas:
   record_id: identificador da janela.
   event_present: 1 se há pulso, 0 se há apenas ruído.
   t0_sample: amostra aproximada de início do pulso. Vale -1 quando não há pulso.
   amplitude_A: amplitude verdadeira do pulso. Vale 0 quando não há pulso.

5. sinais_teste.csv
   250 janelas que podem conter ou não um pulso.
   Deve ser usado apenas após a escolha do método final, para avaliar o desempenho em dados não usados no ajuste.
   Colunas: record_id, n, t_s, y.

6. rotulos_teste.csv
   Rótulos do conjunto de teste.
   Deve ser usado para calcular as métricas finais, como falsos positivos, falsos negativos, precisão, revocação,
   F1-score, erro na posição estimada e erro na amplitude estimada.

Modelo aproximado:
y[n] = A s[n - n0] + r[n],
onde s[n] é o pulso conhecido, A é a amplitude, n0 é o instante de ocorrência e r[n] é ruído.
