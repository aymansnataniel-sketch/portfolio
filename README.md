# portfolio

Proyectos de Nataniel Aymans.

## Modelo predictivo para apuestas deportivas

Se agregó el script `sports_betting_model.py` para entrenar un modelo de clasificación binaria (`home_win`) usando validación temporal y calibración de probabilidades.

### Qué hace

- Usa `TimeSeriesSplit` para evitar fugas de información por tiempo.
- Entrena un `HistGradientBoostingClassifier` calibrado con isotonic regression.
- Reporta métricas out-of-sample:
  - AUC
  - LogLoss
  - Brier Score
- Si el CSV contiene `odds_home` y `odds_away`, calcula:
  - EV home/away
  - Mejor lado
  - Señal de valor esperado positivo (`edge_flag`)

### Requisitos

```bash
pip install pandas numpy scikit-learn
```

### Uso

```bash
python sports_betting_model.py --data datos.csv --target home_win --pred-out predicciones.csv
```

### Formato recomendado de datos

- Objetivo: `home_win` (1 si gana local, 0 si no).
- Cuotas opcionales para detección de valor: `odds_home`, `odds_away`.
- Variables de entrada: ratings, forma reciente, ELO, xG, descanso, localía, lesiones agregadas, etc.
- Ordena el CSV por fecha ascendente antes de entrenar.

> Nota: no existe "alta precisión" garantizada en apuestas. El foco correcto es calibración de probabilidades, control de riesgo y mejora continua por ligas/mercados.
