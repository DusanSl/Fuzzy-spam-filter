import numpy as np
import skfuzzy as fuzz

x_kljucne_reci  = np.arange(0, 11, 1)
x_broj_linkova  = np.arange(0, 11, 1)
x_caps_procenat = np.arange(0, 101, 1)
x_spam_score    = np.arange(0, 101, 1)

kljucne_zanemarljive = fuzz.trapmf(x_kljucne_reci, [0, 0, 1, 3])
kljucne_zastupljene  = fuzz.trimf (x_kljucne_reci, [1, 4, 7])
kljucne_dominantne   = fuzz.trapmf(x_kljucne_reci, [5, 7, 10, 10])

linkovi_minimalni = fuzz.trapmf(x_broj_linkova, [0, 0, 1, 3])
linkovi_umereni   = fuzz.trimf (x_broj_linkova, [1, 3, 5])
linkovi_brojni    = fuzz.trapmf(x_broj_linkova, [4, 6, 10, 10])

caps_uobicajen = fuzz.trapmf(x_caps_procenat, [0,  0,  8, 20])
caps_poviseni  = fuzz.trimf (x_caps_procenat, [12, 30, 50])
caps_agresivan = fuzz.trapmf(x_caps_procenat, [35, 55, 100, 100])


score_legitiman = fuzz.trapmf(x_spam_score, [0,  0,  12, 30])
score_sumnjiv   = fuzz.trimf (x_spam_score, [25, 45, 65])
score_spam      = fuzz.trapmf(x_spam_score, [50, 65, 100, 100])


def stepen_pripadnosti(x_universe, mf, vrednost: float) -> float:
    return float(fuzz.interp_membership(x_universe, mf, vrednost))