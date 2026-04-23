import numpy as np
from fazi.pravila import (
    fuzzifikuj,
    kontroler_spam_score,
)
from fazi.defazifikacija import (
    defazifikuj,
    odredi_kategoriju,
)


def pokreni_fis(
    val_kljucne:       float,
    val_linkovi:       float,
    val_caps:          float,
    val_interpunkcija: float,
) -> dict:

    mu = fuzzifikuj(val_kljucne, val_linkovi, val_caps, val_interpunkcija)

    agregirani_skup = kontroler_spam_score(mu)

    spam_score = defazifikuj(agregirani_skup)

    kategorija = odredi_kategoriju(spam_score)

    return {
        "spam_score": spam_score,
        "kategorija": kategorija,
    }