#!/usr/bin/env python
# coding: utf-8

# # Modelo para apuestas Premier League
#
# - El predictor está basado en el ELO rating, que usamos para estimar las probablidades
#   de cada resultado.
# - Se mantiene un rating global, uno que solo tienen en cuenta los partidos de local
#   y uno que solo tiene en cuenta los partidos de visitante.
# - Primero cargamos en dataframes los datos de la temporada actual y la anterior, se usa
#   la temporada anterior para calcular los elos iniciales de la actual.

# In[69]:


import seaborn as sb
from typing import Tuple, List
import numpy as np
import pandas as pd
import pandasql as pdsql
import matplotlib.pyplot as plt
import sys

# PARAMETERS
# ------------------------------------------------------------------------------
K: int = 36  # ELO K-factor
N_BETS: int = 380  # Number of bets to simulate
HTA: int = 150  # Home team advantage
EV_CUTOFF: float = 0.6  # Cutoff to place bets on
FORM_MEMORY: int = 10  # Number of previous matches to remember
FORM_MULTIPLIER = 12.5
# ------------------------------------------------------------------------------

current_season: str = f"{sys.argv[1]}-{int(sys.argv[2]) - 2000:02d}"
last_season: str = f"{int(sys.argv[1]) - 1:02d}-{int(sys.argv[1]) - 2000:02d}"

# current_season: str = "2020-21"
# last_season: str = "2019-20"

full_df: pd.DataFrame = pd.read_csv(f"Datasets/premier/{current_season}.csv")
last_season_ds: pd.DataFrame = pd.read_csv(
    f"Datasets/premier/{last_season}.csv")
full_df.reset_index(drop=True, inplace=True)
# full_df.head(20)


# - Definimos funciones  para manipular rating.

# In[70]:


def get_expected_score(rating_1: int, rating_2: int) -> float:
    return 1 / (1 + 10**((rating_2 - rating_1) / 400))


def get_new_elo(rating_1: int, rating_2: int, score_1: int, score_2: int) -> Tuple[int, int]:
    """
    Calculates new elo ratings for both teams.
    """
    # Calculate expected score for both teams
    expected_score_1: float = get_expected_score(rating_1, rating_2)
    expected_score_2: float = get_expected_score(rating_2, rating_1)

    # Calculate new elo ratings
    result1: float = 1 if score_1 > score_2 else 0.5 if score_1 == score_2 else 0
    result2: float = 1 - result1

    new_rating_1: int = round(rating_1 + K * (result1 - expected_score_1))
    new_rating_2: int = round(rating_2 + K * (result2 - expected_score_2))

    # Bonus por golear/ Penalización por ser goleado
    bonus_rating_1: float = 0
    bonus_rating_2: float = 0

    if (score_1 - score_2 > 2):
        bonus_rating_1 = score_1 - score_2
        bonus_rating_2 = score_2 - score_1
    elif (score_2 > score_1 > 2):
        bonus_rating_2 = int((score_2 - score_1) * 1.5)
        bonus_rating_1 = int((score_1 - score_2) * 1.5)

    return (new_rating_1 + bonus_rating_1, new_rating_2 + bonus_rating_2)


# - Se juntan los equipos que participaron en la temporada
#   anterior y actual en un solo dataframe.
# - Se calculan los ratings al terminar la temporada anterior.
# - A cada equipo se le asigna un rating inicial de 1200.

# In[71]:


df_teams = pdsql.sqldf(
    "SELECT DISTINCT HomeTeam as TEAM FROM full_df UNION SELECT DISTINCT HomeTeam as TEAM from last_season_ds")
df_teams["RATING"] = df_teams["H_RATING"] = df_teams["A_RATING"] = 1200

for index, row in last_season_ds.iterrows():
    new_elos: Tuple[int, int] = get_new_elo(df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "RATING"].values[0],
                                            df_teams.loc[df_teams["TEAM"] ==
                                                         row["AwayTeam"], "RATING"].values[0],
                                            row["FTHG"], row["FTAG"])
    new_elos_localia: Tuple[int, int] = get_new_elo(df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "H_RATING"].values[0],
                                                    df_teams.loc[df_teams["TEAM"] ==
                                                                 row["AwayTeam"], "A_RATING"].values[0],
                                                    row["FTHG"], row["FTAG"])

    df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "RATING"] = new_elos[0]
    df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "RATING"] = new_elos[1]

    df_teams.loc[df_teams["TEAM"] == row["HomeTeam"],
                 "H_RATING"] = new_elos_localia[0]
    df_teams.loc[df_teams["TEAM"] == row["AwayTeam"],
                 "A_RATING"] = new_elos_localia[1]

for index, row in df_teams.iterrows():
    df_teams.loc[index, "RATING"] = (row["RATING"] + 1200) / 2
    df_teams.loc[index, "H_RATING"] = (row["H_RATING"] + 1200) / 2
    df_teams.loc[index, "A_RATING"] = (row["A_RATING"] + 1200) / 2

# df_teams.head(25)


# - Luego convertimos las odds de nuestro proveedor a probabilidad, columnas PB365H, PB365D y PB365A,
#   normalizando para que la suma de resultados posibles sea 1, ya que la suma da mayor a 1 por el
#   margen que se dejan los bookmakers.
# - Se carga cada partido, actualizando el elo y forma, dejando registro del rating de cada equipo
#   en ese partido y de la forma en la que llegaron al partido (ultimos resultados).
# - Se actualizan las columnas de jugados, ganados, empatados y perdidos de cada equipo.

# In[72]:


df_teams["FORM"] = "D" * FORM_MEMORY
df_teams["LOST"] = df_teams["TIED"] = df_teams["WON"] = df_teams["PLAYED"] = 0
full_df.index.rename('match_id', inplace=True)
df = full_df[["Date", "HomeTeam", "AwayTeam", "FTHG",
              "FTAG", "B365H", "B365D", "B365A"]].copy()
df["PB365H"] = df["PB365D"] = df["PB365A"] = np.nan
df["H_FORM"] = df["A_FORM"] = ""

for index, row in df.iterrows():

    # Odds to implied probability
    sum = 1 / row["B365H"] + 1 / row["B365D"] + 1 / row["B365A"]
    df.loc[index, "PB365H"] = 1 / row["B365H"] / sum
    df.loc[index, "PB365D"] = 1 / row["B365D"] / sum
    df.loc[index, "PB365A"] = 1 / row["B365A"] / sum

    # Calculate and set new elo ratings
    df.at[index, "HTR"] = df_teams.loc[df_teams["TEAM"]
                                       == row["HomeTeam"], "RATING"].values[0]
    df.at[index, "ATR"] = df_teams.loc[df_teams["TEAM"]
                                       == row["AwayTeam"], "RATING"].values[0]

    df.at[index, "A_RATING"] = df_teams.loc[df_teams["TEAM"]
                                            == row["AwayTeam"], "A_RATING"].values[0]
    df.at[index, "H_RATING"] = df_teams.loc[df_teams["TEAM"]
                                            == row["HomeTeam"], "H_RATING"].values[0]

    df.at[index, "A_RATING"] = df_teams.loc[df_teams["TEAM"]
                                            == row["AwayTeam"], "A_RATING"].values[0]
    df.at[index, "H_RATING"] = df_teams.loc[df_teams["TEAM"]
                                            == row["HomeTeam"], "H_RATING"].values[0]

    new_elos: Tuple[int, int] = get_new_elo(df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "RATING"].values[0],
                                            df_teams.loc[df_teams["TEAM"] ==
                                                         row["AwayTeam"], "RATING"].values[0],
                                            row["FTHG"], row["FTAG"])
    new_elos_localia: Tuple[int, int] = get_new_elo(df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "H_RATING"].values[0],
                                                    df_teams.loc[df_teams["TEAM"] ==
                                                                 row["AwayTeam"], "A_RATING"].values[0],
                                                    row["FTHG"], row["FTAG"])

    df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "RATING"] = new_elos[0]
    df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "RATING"] = new_elos[1]

    df.at[index, "H_FORM"] = df_teams.loc[df_teams["TEAM"]
                                          == row["HomeTeam"], "FORM"].values[0]
    df.at[index, "A_FORM"] = df_teams.loc[df_teams["TEAM"]
                                          == row["AwayTeam"], "FORM"].values[0]

    df_teams.loc[df_teams["TEAM"] == row["HomeTeam"],
                 "H_RATING"] = new_elos_localia[0]
    df_teams.loc[df_teams["TEAM"] == row["AwayTeam"],
                 "A_RATING"] = new_elos_localia[1]

    # Update team stats and form
    df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "PLAYED"] += 1
    df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "PLAYED"] += 1
    if row["FTHG"] > row["FTAG"]:
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "WON"] += 1
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "LOST"] += 1
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "FORM"] = "W" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["HomeTeam"], "FORM"].values[0][:-1]
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "FORM"] = "L" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["AwayTeam"], "FORM"].values[0][:-1]
    elif row["FTHG"] == row["FTAG"]:
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "TIED"] += 1
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "TIED"] += 1
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "FORM"] = "D" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["HomeTeam"], "FORM"].values[0][:-1]
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "FORM"] = "D" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["AwayTeam"], "FORM"].values[0][:-1]
    else:
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "LOST"] += 1
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "WON"] += 1
        df_teams.loc[df_teams["TEAM"] == row["HomeTeam"], "FORM"] = "L" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["HomeTeam"], "FORM"].values[0][:-1]
        df_teams.loc[df_teams["TEAM"] == row["AwayTeam"], "FORM"] = "W" + \
            df_teams.loc[df_teams["TEAM"] ==
                         row["AwayTeam"], "FORM"].values[0][:-1]


df_teams.sort_values(by="RATING", inplace=True, ascending=False)
# df_teams.head(23)


# - Se va a apostar en los últimos N partidos, ya que los ratings tardan en converger, aunque
#   es posible que los bookmakers también tengan problemas al estimar sus probabilidades ya que
#   no hay muchos datos de la temporada.
# - El elo nos da el resultado esperado, pero en actividades con empates como este caso y
#   al contar un empate como media victoria y media derrota, hay que restar de las probabilidades
#   de victoria del equipo local y del visitante una cantidad desconocida.
#     - Usamos la probabilidad de empate del bookmaker, y restamos la misma proporcionalmente a
#     las probabilidades de victoria
# - Se calcula un rating efectivo para el partido en cuestión teniendo en cuenta la forma y localía.
#   - Esto trae ciertas limitaciones, puesto que no todos equipos deberían tener la misma ventaja de
#   localía, y ésta debería ser dinámica, cambiando a lo largo de la temporada.

# In[73]:


my_bets: pd.DataFrame = df.copy()
my_bets["PH"] = my_bets["PD"] = my_bets["PA"] = my_bets["SUM"] = np.nan

for index, row in my_bets.tail(N_BETS).iterrows():
    effective_rating_H: float = row["H_RATING"] + HTA + (
        row["H_FORM"].count("W") - row["H_FORM"].count("L")) * FORM_MULTIPLIER
    effective_rating_A: float = row["A_RATING"] + (
        row["A_FORM"].count("W") - row["A_FORM"].count("L")) * FORM_MULTIPLIER

    p_home: float = get_expected_score(effective_rating_H, effective_rating_A)
    p_away: float = get_expected_score(effective_rating_A, effective_rating_H)
    proportion_ha: float = p_home / (p_home + p_away)

    p_tie: float = row["PB365D"]

    my_bets.loc[index, "PH"] = get_expected_score(
        effective_rating_H, effective_rating_A) - p_tie * proportion_ha
    my_bets.loc[index, "PA"] = get_expected_score(
        effective_rating_A, effective_rating_H) - p_tie * (1 - proportion_ha)

    my_bets.loc[index, "PD"] = p_tie
    my_bets.loc[index, "SUM"] = my_bets.loc[index, "PH"] + \
        my_bets.loc[index, "PA"] + my_bets.loc[index, "PD"]


# # Benchmarking el modelo
#
# - Vamos a simular apuestas, partiendo de un bankroll y usando un tamaño de apuesta fijo
#   - La mayor parte de nuestras ganancias/pérdidas las vamos a tener en las apuestas a
#   heavy underdogs.
# - Para decidir qué apuesta hacer, primero debemos calcular el valor esperado de cada opción.
#   - $EV=\text{Bet} * (\text{Odds} - 1) * P_{result}- \text{Bet} * (1 - P_{result})$
#   - La mejor opción va a ser la de mayor $EV$.
# - Finalmente para decidir si se apuesta o no, la mejor opción tiene que tener un $EV$ mayor
#   a dicho threshold. Idealmente ese threshold debería ser 0, pero los mejores resultados se
#   consiguen con un threshold mayor a 1.
# - Finalmente dejamos registro de cuanto dinero se ganó/perdió en cada partido y actualizamos
#   el bankroll.

# In[74]:


# Benchmarking value bets
my_bets["CHOICE"] = 'NO BET'
my_bets["GAIN_LOSS"] = np.nan

choices: List[str] = []
bet_size: float = 1.0
bankroll_real: float = 100.0
bankroll_theory: float = bankroll_real
bankroll_over_time: List[float] = [bankroll_real]
for index, row in my_bets.tail(N_BETS).iterrows():
    ev_home: float = bet_size * \
        (row["B365H"] - 1) * row["PH"] - bet_size * (1 - row["PH"])
    ev_away: float = bet_size * \
        (row["B365A"] - 1) * row["PA"] - bet_size * (1 - row["PA"])
    ev_tie: float = bet_size * \
        (row["B365D"] - 1) * row["PD"] - bet_size * (1 - row["PD"])

    if max(ev_home, ev_away, ev_tie) < EV_CUTOFF:
        # Si no hay apuesta atractiva para hacer, no apostamos
        continue

    choice: str = ("H" if ev_home == max(ev_home, ev_away, ev_tie) else
                   "A" if ev_away == max(ev_home, ev_away, ev_tie) else
                   "D")
    p_choice: float = row["PH"] if choice == "H" else row["PA"] if choice == "A" else "D"
    choices.append(choice)

    my_bets.loc[index, "CHOICE"] = choice
    my_bets.loc[index, "P_CHOICE"] = p_choice

    delta_bankroll: float = -bet_size

    if choice == "H" and row["FTHG"] > row["FTAG"]:
        delta_bankroll = bet_size * (row["B365H"] - 1)
        bankroll_real += bet_size * (row["B365H"] - 1)
    elif choice == "A" and row["FTHG"] < row["FTAG"]:
        delta_bankroll = bet_size * (row["B365A"] - 1)
        bankroll_real += bet_size * (row["B365A"] - 1)
    elif choice == "D" and row["FTHG"] == row["FTAG"]:
        delta_bankroll = bet_size * (row["B365D"] - 1)
        bankroll_real += bet_size * (row["B365D"] - 1)
    else:
        bankroll_real -= bet_size
    my_bets.loc[index, "GAIN_LOSS"] = delta_bankroll
    bankroll_over_time.append(bankroll_real)

print(bankroll_real)

# plt.rcParams["figure.figsize"] = (12, 8)
# plt.grid(True)
# plt.title
# plt.plot(bankroll_over_time)
# plt.show()
# plt.grid(True)
# plt.bar(my_bets["CHOICE"].unique(), [my_bets.tail(N_BETS).loc[my_bets["CHOICE"]
#         == choice, "CHOICE"].count() for choice in my_bets["CHOICE"].unique()])
# plt.show()

df_aux = my_bets.tail(N_BETS)


# In[75]:


# pdsql.sqldf(
#     """
#   SELECT  CHOICE,
#           AVG(GAIN_LOSS) AS PROMEDIO,
#           SUM(GAIN_LOSS) AS TOTAL,
#           MAX(GAIN_LOSS) AS MAX_GAIN_LOSS,
#           COUNT(GAIN_LOSS) AS CANTIDAD,
#           CASE WHEN P_CHOICE > 0.6 THEN "SEGURA"
#                WHEN P_CHOICE > 0.3 THEN "PAREJO"
#                ELSE "LONG SHOT"
#           END AS PROBABLE
#   FROM df_aux
#   WHERE CHOICE NOT LIKE '%NO BET%'
#   GROUP BY CHOICE, PROBABLE
#   """
# ).head()


# In[76]:


my_bets.tail(1)[["H_FORM", "A_FORM", "H_RATING", "A_RATING",
                 "B365H", "B365A", "B365D", "PH", "PA", "PD"]]
