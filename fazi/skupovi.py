import numpy as np
import skfuzzy as fuzz

x_kljucne_reci  = np.arange(0, 21, 1)
x_broj_linkova  = np.arange(0, 21, 1)
x_caps_procenat = np.arange(0, 101, 1)
x_spam_score    = np.arange(0, 101, 1)

kljucne_zanemarljive = fuzz.trapmf(x_kljucne_reci, [0,  0,  2,  5])
kljucne_zastupljene  = fuzz.trimf (x_kljucne_reci, [3,  7, 12])
kljucne_dominantne   = fuzz.trapmf(x_kljucne_reci, [10, 14, 20, 20])

linkovi_minimalni = fuzz.trapmf(x_broj_linkova, [0,  0,  1,  3])
linkovi_umereni   = fuzz.trimf (x_broj_linkova, [2,  5,  9])
linkovi_brojni    = fuzz.trapmf(x_broj_linkova, [8, 12, 20, 20])

caps_uobicajen = fuzz.trapmf(x_caps_procenat, [0,  0,  10, 25])
caps_poviseni  = fuzz.trimf (x_caps_procenat, [15, 35, 55])
caps_agresivan = fuzz.trapmf(x_caps_procenat, [45, 60, 100, 100])


score_legitiman = fuzz.trapmf(x_spam_score, [0,  0,  15, 35])
score_sumnjiv   = fuzz.trimf (x_spam_score, [25, 45, 65])
score_spam      = fuzz.trapmf(x_spam_score, [55, 70, 100, 100])


def stepen_pripadnosti(x_universe, mf, vrednost: float) -> float:
    return float(fuzz.interp_membership(x_universe, mf, vrednost))