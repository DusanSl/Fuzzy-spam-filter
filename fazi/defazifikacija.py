import numpy as np
import skfuzzy as fuzz
from fazi.skupovi import (
    x_spam_score,
    score_legitiman, score_sumnjiv, score_spam,
    stepen_pripadnosti,
)

def defazifikuj(agregirani_skup: np.ndarray) -> float:
    if agregirani_skup.max() == 0:
        return 0.0

    return float(fuzz.defuzz(x_spam_score, agregirani_skup, "centroid"))

def odredi_kategoriju(spam_score: float) -> str:

    pripadnost_legitiman = stepen_pripadnosti(x_spam_score, score_legitiman, spam_score)
    pripadnost_sumnjivo  = stepen_pripadnosti(x_spam_score, score_sumnjiv,   spam_score)
    pripadnost_spam      = stepen_pripadnosti(x_spam_score, score_spam,      spam_score)

    nivoi = {
        "LEGITIMAN": pripadnost_legitiman,
        "SUMNJIVO":  pripadnost_sumnjivo,
        "SPAM":      pripadnost_spam,
    }

    return max(nivoi, key=nivoi.get)