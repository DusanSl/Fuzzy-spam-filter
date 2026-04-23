import numpy as np
from fazi.skupovi import (
    x_kljucne_reci,  kljucne_zanemarljive, kljucne_zastupljene, kljucne_dominantne,
    x_broj_linkova,  linkovi_minimalni,    linkovi_umereni,     linkovi_brojni,
    x_caps_procenat, caps_uobicajen,       caps_poviseni,       caps_agresivan,
    x_interpunkcija, interpunkcija_retka,  interpunkcija_umerena, interpunkcija_agresivna,
    score_legitiman, score_sumnjiv,        score_spam,
    stepen_pripadnosti,
)


def fuzzifikuj(
    val_kljucne:     float,
    val_linkovi:     float,
    val_caps:        float,
    val_interpunkcija: float,
) -> dict:
    return {
        "kljucne_reci": {
            "zanemarljive": stepen_pripadnosti(x_kljucne_reci, kljucne_zanemarljive, val_kljucne),
            "zastupljene":  stepen_pripadnosti(x_kljucne_reci, kljucne_zastupljene,  val_kljucne),
            "dominantne":   stepen_pripadnosti(x_kljucne_reci, kljucne_dominantne,   val_kljucne),
        },
        "broj_linkova": {
            "minimalni": stepen_pripadnosti(x_broj_linkova, linkovi_minimalni, val_linkovi),
            "umereni":   stepen_pripadnosti(x_broj_linkova, linkovi_umereni,   val_linkovi),
            "brojni":    stepen_pripadnosti(x_broj_linkova, linkovi_brojni,    val_linkovi),
        },
        "caps_procenat": {
            "uobicajen": stepen_pripadnosti(x_caps_procenat, caps_uobicajen, val_caps),
            "poviseni":  stepen_pripadnosti(x_caps_procenat, caps_poviseni,  val_caps),
            "agresivan": stepen_pripadnosti(x_caps_procenat, caps_agresivan, val_caps),
        },
        "interpunkcija": {
            "retka":     stepen_pripadnosti(x_interpunkcija, interpunkcija_retka,     val_interpunkcija),
            "umerena":   stepen_pripadnosti(x_interpunkcija, interpunkcija_umerena,   val_interpunkcija),
            "agresivna": stepen_pripadnosti(x_interpunkcija, interpunkcija_agresivna, val_interpunkcija),
        },
    }


def kontroler_spam_score(mu: dict) -> np.ndarray:
    kljucne = mu["kljucne_reci"]
    linkovi = mu["broj_linkova"]
    caps    = mu["caps_procenat"]
    inter   = mu["interpunkcija"]

    aktivacije = []

    p01 = min(kljucne["zanemarljive"], linkovi["minimalni"], caps["uobicajen"], inter["retka"])
    aktivacije.append(np.fmin(p01, score_legitiman))

    p02 = min(kljucne["zastupljene"], linkovi["umereni"], caps["poviseni"], inter["umerena"])
    aktivacije.append(np.fmin(p02, score_sumnjiv))

    p03 = np.fmax(kljucne["dominantne"], linkovi["brojni"])
    aktivacije.append(np.fmin(p03, score_spam))

    p04 = min(kljucne["zastupljene"], np.fmax(caps["agresivan"], inter["agresivna"]))
    aktivacije.append(np.fmin(p04, score_spam))

    p05 = min(linkovi["umereni"], np.fmax(caps["agresivan"], inter["agresivna"]))
    aktivacije.append(np.fmin(p05, score_spam))

    p06 = kljucne["dominantne"]
    aktivacije.append(np.fmin(p06, score_spam))

    p07 = min(kljucne["zastupljene"], linkovi["minimalni"])
    aktivacije.append(np.fmin(p07, score_sumnjiv))

    p08 = min(np.fmax(kljucne["zanemarljive"], kljucne["zastupljene"]),linkovi["umereni"])
    aktivacije.append(np.fmin(p08, score_sumnjiv))

    p09 = min(kljucne["dominantne"], caps["uobicajen"])
    aktivacije.append(np.fmin(p09, score_sumnjiv))

    return np.fmax.reduce(aktivacije)