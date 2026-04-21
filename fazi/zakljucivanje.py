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
    val_kljucne: float,
    val_linkovi: float,
    val_caps:    float,
    ispisi:      bool = False,
) -> dict:

    mu = fuzzifikuj(val_kljucne, val_linkovi, val_caps)

    agregirani_skup = kontroler_spam_score(mu)

    spam_score = defazifikuj(agregirani_skup)

    kategorija = odredi_kategoriju(spam_score)

    if ispisi:
        print(f"  Ključne reči : {val_kljucne}")
        print(f"  Broj linkova : {val_linkovi}")
        print(f"  Caps procenat: {val_caps}%")
        print(f"  Spam score   : {spam_score:.2f}")
        print(f"  Kategorija   : {kategorija}")

    return {
        "spam_score": spam_score,
        "kategorija": kategorija,
    }