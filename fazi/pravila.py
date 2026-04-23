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

    p01 = min(kljucne["zanemarljive"], caps["uobicajen"])
    aktivacije.append(np.fmin(p01, score_legitiman))
    p02 = min(kljucne["zanemarljive"], linkovi["minimalni"])
    aktivacije.append(np.fmin(p02, score_legitiman))
    p03 = min(kljucne["zanemarljive"], caps["poviseni"])
    aktivacije.append(np.fmin(p03, score_legitiman))
    p04 = min(inter["retka"], kljucne["zanemarljive"])
    aktivacije.append(np.fmin(p04, score_legitiman))

    p05 = min(kljucne["zastupljene"], linkovi["umereni"])
    aktivacije.append(np.fmin(p05, score_sumnjiv))
    p06 = min(linkovi["minimalni"], caps["poviseni"])
    aktivacije.append(np.fmin(p06, score_sumnjiv))
    p07 = min(kljucne["dominantne"], linkovi["minimalni"])
    aktivacije.append(np.fmin(p07, score_sumnjiv))
    p08 = min(kljucne["zastupljene"], linkovi["minimalni"])
    aktivacije.append(np.fmin(p08, score_sumnjiv))
    p09 = min(kljucne["zastupljene"], caps["poviseni"])
    aktivacije.append(np.fmin(p09, score_sumnjiv))
    p10 = kljucne["zastupljene"]
    aktivacije.append(np.fmin(p10, score_sumnjiv))
    p11 = min(linkovi["umereni"], caps["agresivan"])
    aktivacije.append(np.fmin(p11, score_sumnjiv))
    p12 = min(linkovi["umereni"], caps["poviseni"])
    aktivacije.append(np.fmin(p12, score_sumnjiv))
    p13 = min(kljucne["zanemarljive"], linkovi["brojni"])
    aktivacije.append(np.fmin(p13, score_sumnjiv))
    p14 = inter["agresivna"]
    aktivacije.append(np.fmin(p14, score_sumnjiv))
    p15 = min(inter["umerena"], kljucne["zastupljene"])
    aktivacije.append(np.fmin(p15, score_sumnjiv))
    p16 = min(caps["agresivan"], linkovi["minimalni"])
    aktivacije.append(np.fmin(p16, score_sumnjiv))

    p17 = min(kljucne["dominantne"], linkovi["brojni"])
    aktivacije.append(np.fmin(p17, score_spam))
    p18 = min(kljucne["zastupljene"], caps["agresivan"])
    aktivacije.append(np.fmin(p18, score_spam))
    p19 = min(linkovi["brojni"], caps["agresivan"])
    aktivacije.append(np.fmin(p19, score_spam))
    p20 = min(kljucne["dominantne"], caps["poviseni"])
    aktivacije.append(np.fmin(p20, score_spam))
    p21 = linkovi["brojni"]
    aktivacije.append(np.fmin(p21, score_spam))
    p22 = min(kljucne["zastupljene"], linkovi["brojni"])
    aktivacije.append(np.fmin(p22, score_spam))
    p23 = min(kljucne["dominantne"], caps["agresivan"])
    aktivacije.append(np.fmin(p23, score_spam))
    p24 = min(inter["agresivna"], kljucne["zastupljene"])
    aktivacije.append(np.fmin(p24, score_spam))
    p25 = min(inter["agresivna"], caps["agresivan"])
    aktivacije.append(np.fmin(p25, score_spam))
    p26 = min(inter["agresivna"], kljucne["dominantne"])
    aktivacije.append(np.fmin(p26, score_spam))

    return np.fmax.reduce(aktivacije)