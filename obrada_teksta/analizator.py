import re

SPAM_RECNIK = [

    "win", "winner", "won", "free", "click", "offer", "prize",
    "congratulations", "claim", "cash", "bonus", "reward",
    "urgent", "limited", "exclusive", "guarantee", "deal",
    "cheap", "discount", "buy now", "act now", "order now",
    "money", "income", "profit", "rich", "million", "dollar",

    "pobedi", "pobednik", "osvojio", "besplatno", "klikni", "ponuda", "nagrada",
    "čestitamo", "preuzmi", "gotovina", "bonus", "nagrada",
    "hitno", "ograničeno", "ekskluzivno", "garantovano", "pogodba",
    "jeftino", "popust", "kupi sad", "deluj odmah", "naruči odmah",
    "novac", "prihod", "zarada", "bogat", "milion", "dolar",
]



def analiziraj_email(tekst: str) -> dict:

    tekst_mali = tekst.lower()

    broj_kljucnih_reci = 0
    for rec in SPAM_RECNIK:
        broj_kljucnih_reci += len(re.findall(r'\b' + re.escape(rec) + r'\b', tekst_mali))

    broj_kljucnih_reci = min(broj_kljucnih_reci, 20)

    pronadjeni_linkovi = re.findall(r'https?://\S+', tekst)
    broj_linkova = min(len(pronadjeni_linkovi), 20)

    sva_slova = [karakter for karakter in tekst if karakter.isalpha()]
    if len(sva_slova) > 0:
        velika_slova = sum(1 for karakter in sva_slova if karakter.isupper())
        caps_procenat = (velika_slova / len(sva_slova)) * 100
    else:
        caps_procenat = 0.0

    return {
        "kljucne_reci":  float(broj_kljucnih_reci),
        "broj_linkova":  float(broj_linkova),
        "caps_procenat": round(caps_procenat, 2),
    }


def ucitaj_random_primer() -> str:

    sadrzaj = PUTANJA_PRIMERI.read_text(encoding="utf-8")

    primeri = [p.strip() for p in sadrzaj.split("---") if p.strip()]

    return random.choice(primeri)
