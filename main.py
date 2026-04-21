from flask import Flask, request, jsonify, render_template
from obrada_teksta.analizator import analiziraj_email, ucitaj_random_primer
from fazi.zakljucivanje import pokreni_fis

app = Flask(
    __name__,
    template_folder="veb/sabloni",
    static_folder="veb/static",
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/primer")
def primer():
    tekst = ucitaj_random_primer()
    return jsonify({"tekst": tekst})


@app.route("/analiziraj", methods=["POST"])
def analiziraj():
    if request.is_json:
        podaci = request.get_json()
        tekst_emaila = podaci.get("tekst", "")
    else:
        tekst_emaila = request.form.get("tekst", "")

    if not tekst_emaila.strip():
        return jsonify({"greska": "Tekst emaila je prazan."}), 400

    ulazi = analiziraj_email(tekst_emaila)

    rezultat_fis = pokreni_fis(
        ulazi["kljucne_reci"],
        ulazi["broj_linkova"],
        ulazi["caps_procenat"],
    )

    rezultat = {
        "tekst":         tekst_emaila,
        "kljucne_reci":  ulazi["kljucne_reci"],
        "broj_linkova":  ulazi["broj_linkova"],
        "caps_procenat": ulazi["caps_procenat"],
        "spam_score":    round(rezultat_fis["spam_score"], 2),
        "kategorija":    rezultat_fis["kategorija"],
    }

    if request.is_json:
        return jsonify(rezultat)

    return render_template("rezultat.html", r=rezultat)


if __name__ == "__main__":
    app.run(debug=True, port=5000)