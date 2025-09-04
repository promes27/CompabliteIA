import os
import re
import cv2
import fitz
import pandas as pd
import easyocr
import streamlit as st
from pdf2image import convert_from_path
from io import StringIO
from rapidfuzz import fuzz
import spacy
from openai import OpenAI

from dotenv import load_dotenv   # <<--- AJOUT

# Charger .env
load_dotenv()

# ==========================
# CONFIGURATION
# ==========================
CSV_OUTPUT = "brouillard_complet.csv"
PDF_PCG = "D:/OCR/CompabliteIA/plan-comptable-general-2005.pdf"

# Lecture OCR
reader = easyocr.Reader(['fr', 'en'], gpu=False)

# Initialisation OpenAI via variable d'environnement
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ La clé OpenAI n'est pas définie. Veuillez créer la variable d'environnement OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

# Modèle NLP pour français
nlp = spacy.load("fr_core_news_sm")

# ==========================
# CHARGEMENT PCG
# ==========================
def charger_classes_1_a_7_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texte_total = ""
    for page in doc:
        texte_total += page.get_text()
    lignes = texte_total.split('\n')
    paragraphe_classe = []
    garder = False
    for ligne in lignes:
        texte = ligne.strip()
        if re.match(r"^CLASSE\s+1", texte, re.IGNORECASE):
            garder = True
        elif re.match(r"^CLASSE\s+8", texte, re.IGNORECASE):
            garder = False
        if garder and texte:
            paragraphe_classe.append(texte)
    return "\n".join(paragraphe_classe)

pcg_contenu = charger_classes_1_a_7_pdf(PDF_PCG)

# ==========================
# FONCTIONS UTILITAIRES
# ==========================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin

def nettoyer_montant(text):
    try:
        text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
        montant = text.replace(" ", "").replace(",", ".")
        montant = re.sub(r"[^\d.]", "", montant)
        return float(montant) if montant else 0.0
    except:
        return 0.0

def extraire_champs(texte):
    date_regex = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', texte)
    date_extrait = date_regex[0] if date_regex else ''
    num_facture = re.findall(r'FACTURE\s*(?:No|N°|#)?\s*([\w\-]+)', texte, flags=re.IGNORECASE)
    num_facture = num_facture[0] if num_facture else ''
    montant_ht = re.findall(r'Total\s*HT\s*[:\-]?\s*([\d\s.,]+)', texte, flags=re.IGNORECASE)
    montant_tva = re.findall(r'TVA\s*\(.*?\)\s*[:\-]?\s*([\d\s.,]+)', texte, flags=re.IGNORECASE)
    montant_ttc = re.findall(r'Total\s*TTC\s*[:\-]?\s*([\d\s.,]+)', texte, flags=re.IGNORECASE)
    ht = nettoyer_montant(montant_ht[0]) if montant_ht else 0.0
    tva = nettoyer_montant(montant_tva[0]) if montant_tva else 0.0
    ttc = nettoyer_montant(montant_ttc[0]) if montant_ttc else 0.0
    if abs(ttc - (ht + tva)) > 1000 and ht > 0 and tva > 0:
        ttc = ht + tva
    return {
        'date': date_extrait,
        'numero_piece': num_facture,
        'montant_ht': ht,
        'tva': tva,
        'montant_ttc': ttc
    }

def detecter_doublons(df, seuil_similarite=85):
    alertes = []
    for i, row in df.iterrows():
        doublon = False
        for j in range(i):
            autre = df.iloc[j]
            if (
                row.get("date") == autre.get("date") and
                row.get("numero_piece") == autre.get("numero_piece") and
                row.get("montant_ttc") == autre.get("montant_ttc")
            ):
                doublon = True
                break
            similarite = fuzz.token_sort_ratio(row.get("texte_brut",""), autre.get("texte_brut",""))
            if similarite >= seuil_similarite:
                doublon = True
                break
        alertes.append("Doublon détecté" if doublon else "Non")
    df["alerte_doublon"] = alertes
    return df

def classifier_avec_gpt(texte_brut):
    prompt = f"""
Tu es un expert en comptabilité. En te basant sur le texte suivant extrait d'une pièce comptable, indique de quel type de journal comptable il s'agit (achat, vente, banque, caisse, OD).

Texte de la pièce :
\"\"\"{texte_brut}\"\"\" 

Répond uniquement par le type de journal.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        st.warning(f"Erreur GPT : {e}")
        return "inconnu"

def detecter_compte(texte_ocr, contenu_pcg):
    prompt = f"""
Tu es un expert du Plan Comptable Général malgache 2005.

Voici un texte extrait d'une pièce comptable :
\"\"\"{texte_ocr}\"\"\" 

Voici un extrait du PCG (classes 1 à 7) :
\"\"\"{contenu_pcg[:12000]}\"\"\" 

Quel est le numéro de compte le plus approprié selon le PCG ?
Réponds uniquement par un numéro (exemple : 606).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().split()[0]
    except:
        return "000"

def generer_journal_avec_llm(texte_ocr, pcg_contenu, journal_csv="grand_journal.csv"):
    prompt = f"""
Tu es un expert en comptabilité basé sur le Plan Comptable Général malgache 2005.

Voici une pièce comptable scannée :
\"\"\"{texte_ocr}\"\"\" 

Voici le plan comptable (extrait classes 1 à 7) :
\"\"\"{pcg_contenu[:12000]}\"\"\" 

Génère l'écriture comptable correspondante au format suivant (journal à 5 colonnes) :

Date | Numéro de compte | Libellé | Débit | Crédit

Réponds uniquement avec un tableau propre.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        texte_tableau = response.choices[0].message.content.strip()

        lignes = [l for l in texte_tableau.split("\n") if "|" in l and "---" not in l]
        colonnes = ["Date", "Numéro de compte", "Libellé", "Débit (Ar)", "Crédit (Ar)"]
        donnees_corrigees = []

        for l in lignes:
            c = [x.strip() for x in l.split("|")[1:-1]]
            while len(c) < len(colonnes):
                c.append("")
            c = c[:len(colonnes)]
            donnees_corrigees.append(c)

        df_journal = pd.DataFrame(donnees_corrigees, columns=colonnes)
        df_journal.to_csv(journal_csv, index=False, encoding="utf-8-sig")
        st.write(f"✅ Journal sauvegardé dans {journal_csv}")

        return df_journal
    except Exception as e:
        st.warning(f"Erreur journal GPT : {e}")
        return pd.DataFrame()

def traiter_image(image_path, pcg_contenu):
    st.markdown(f"### Traitement : {os.path.basename(image_path)}")
    img = preprocess_image(image_path)
    result = reader.readtext(img)
    texte_complet = " ".join([text for _, text, _ in result])
    st.text("Texte OCR extrait :")
    st.write(texte_complet)

    champs = extraire_champs(texte_complet)
    champs['texte_brut'] = texte_complet
    champs['fichier'] = os.path.basename(image_path)
    champs['type_journal'] = classifier_avec_gpt(texte_complet)
    champs['compte'] = detecter_compte(texte_complet, pcg_contenu)
    champs['journal_markdown'] = generer_journal_avec_llm(texte_complet, pcg_contenu)
    return champs

# ==========================
# INTERFACE STREAMLIT
# ==========================
st.markdown('<h2 style="color: green; font-size:20px; text-align: center;">Automatisation des Processus Comptables</h2>', unsafe_allow_html=True)

fichiers = st.file_uploader("Upload fichiers (images, PDF, CSV, Excel)", type=["png","jpg","jpeg","pdf","csv","xlsx"], accept_multiple_files=True)

if st.button("Traiter") and fichiers:
    donnees = []

    for fichier in fichiers:
        extension = os.path.splitext(fichier.name)[1].lower()
        temp_path = "temp_" + fichier.name
        with open(temp_path, "wb") as f:
            f.write(fichier.read())

        if extension in [".png",".jpg",".jpeg"]:
            donnees.append(traiter_image(temp_path, pcg_contenu))
        elif extension == ".pdf":
            doc = fitz.open(temp_path)
            texte_total = "".join([page.get_text() for page in doc])
            if texte_total.strip():
                champs = extraire_champs(texte_total)
                champs['texte_brut'] = texte_total
                champs['fichier'] = os.path.basename(temp_path)
                champs['type_journal'] = classifier_avec_gpt(texte_total)
                champs['compte'] = detecter_compte(texte_total, pcg_contenu)
                champs['journal_markdown'] = generer_journal_avec_llm(texte_total, pcg_contenu)
                donnees.append(champs)
            else:
                pages = convert_from_path(temp_path)
                for i, page in enumerate(pages):
                    img_path = f"temp_page_{i}.png"
                    page.save(img_path,"PNG")
                    donnees.append(traiter_image(img_path, pcg_contenu))
                    os.remove(img_path)
        elif extension == ".xlsx":
            df = pd.read_excel(temp_path)
            donnees.extend(df.to_dict(orient="records"))
        elif extension == ".csv":
            df = pd.read_csv(temp_path)
            donnees.extend(df.to_dict(orient="records"))

        os.remove(temp_path)

    if donnees:
        df = pd.DataFrame(donnees)
        df = detecter_doublons(df, seuil_similarite=85)
        df_final = pd.DataFrame(donnees)
        df_final.to_csv(CSV_OUTPUT,index=False,encoding="utf-8-sig")
        st.success(f"✅ Extraction terminée, exporté dans {CSV_OUTPUT}")
        st.dataframe(df)
        st.download_button("⬇️ Télécharger le CSV", data=df_final.to_csv(index=False, encoding="utf-8-sig"), file_name=CSV_OUTPUT, mime="text/csv")





