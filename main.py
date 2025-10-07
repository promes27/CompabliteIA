
import os
import re
import cv2
import fitz
import pandas as pd
import easyocr
import streamlit as st
from pdf2image import convert_from_path
from rapidfuzz import fuzz
import spacy
from openai import OpenAI
from dotenv import load_dotenv
from collections import deque
from PIL import Image
import io
import numpy as np

# ==========================
# CONFIGURATION
# ==========================


load_dotenv()
CSV_OUTPUT = "grand_journal.csv"
PDF_PCG = "./plan-comptable-general-2005.pdf"

# OCR et NLP
reader = easyocr.Reader(['fr', 'en'], gpu=False)
nlp = spacy.load("fr_core_news_sm")

# OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ La clé OpenAI n'est pas définie. Créez OPENAI_API_KEY dans .env")
    st.stop()
client = OpenAI(api_key=api_key)

# ==========================
# INITIALISATION DE L'ÉTAT
# ==========================
if 'df_final' not in st.session_state:
    st.session_state.df_final = pd.DataFrame()

if 'df_modifie' not in st.session_state:
    st.session_state.df_modifie = pd.DataFrame()

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
    if date_extrait:
        try:
            date_extrait = pd.to_datetime(date_extrait, dayfirst=True).date()
        except:
            pass

    num_facture = re.findall(r'(?:FACTURE|facture)?\s*(?:N°|N["\'%]?)?\s*([0-9]{4}-[0-9]+)', texte)
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
    """
    Classe une pièce comptable en type de journal : achat, vente, banque, caisse, OD.
    Utilise GPT avec un prompt plus précis et valide la réponse.
    """
    # Nettoyage du texte OCR pour réduire le bruit
    texte_propre = re.sub(r'\s+', ' ', texte_brut)  # supprimer les sauts de ligne multiples
    texte_propre = re.sub(r'[^a-zA-Z0-9 /.,\-]', '', texte_propre)  # garder caractères utiles

    prompt = f"""
Tu es un expert en comptabilité malgache selon le PCG 2005. 
Tu dois classer une pièce comptable selon son type exact. 
Les types possibles sont : 
- achat
- vente
- banque
- caisse
- OD (opérations diverses)

Ignore les mots trompeurs et base-toi sur le contexte réel de la pièce 
(client ou fournisseur, paiement reçu ou effectué, type de document, etc.)

Texte de la pièce comptable (nettoyé) :
\"\"\"{texte_propre}\"\"\" 

Réponds uniquement par un mot exact parmi : achat, vente, banque, caisse, OD
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        type_journal = response.choices[0].message.content.strip().lower()

        # Validation stricte : si GPT renvoie autre chose, on met "inconnu"
        if type_journal not in ["achat", "vente", "banque", "caisse", "od"]:
            type_journal = "inconnu"
        return type_journal

    except Exception as e:
        print(f"Erreur classifier_avec_gpt : {e}")
        return "inconnu"

def detecter_compte(texte_ocr, contenu_pcg, client=None):
    prompt = f"""
    Tu es un expert du Plan Comptable Général malgache 2005.

    Texte extrait :
    \"\"\"{texte_ocr}\"\"\"

    Client/Fournisseur :
    \"\"\"{client}\"\"\"

    Extrait PCG (classes 1 à 7) :
    \"\"\"{contenu_pcg[:12000]}\"\"\"

    Quel est le numéro de compte le plus approprié ?
    Réponds uniquement par un numéro.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().split()[0]
    except:
        return "000"


def generer_journal_avec_llm(texte_ocr, pcg_contenu, type_journal="vente"):
    prompt = f"""
Tu es un expert comptable selon le PCG 2005 Madagascar.

Voici une pièce comptable :
\"\"\"{texte_ocr}\"\"\"

Consignes :
- Si c’est une **vente** :
  - Utiliser 411 pour le client (jamais 512 Banque dans le journal de vente).
  - Utiliser 70x pour les ventes et 44571 pour la TVA collectée.
- Si c’est un **achat** :
  - Utiliser 401 pour le fournisseur (jamais 512 Banque).
  - Utiliser 60x pour les charges et 44566 pour la TVA déductible.
- Si c’est une opération de **banque** :
  - Utiliser 512 (banque) avec contrepartie 411 (client) ou 401 (fournisseur).
- Le montant TTC doit être équilibré entre débits et crédits.
- Donne un tableau avec 5 colonnes : Date, Numéro de compte, Libellé, Débit, Crédit.
- La Date doit être au format JJ/MM/AAAA.
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
        
        # Nettoyage
        if len(df_journal) > 0:
            premiere_ligne = df_journal.iloc[0]
            mots_cles = ['date','compte','libellé','libelle','débit','debit','crédit','credit']
            if any(mot in " ".join(str(v).lower() for v in premiere_ligne.values) for mot in mots_cles):
                df_journal = df_journal.drop(0).reset_index(drop=True)

        df_journal = df_journal.dropna(how='all').reset_index(drop=True)
        df_journal.index = df_journal.index + 1
        
        if "Date" in df_journal.columns:
            df_journal["Date"] = pd.to_datetime(df_journal["Date"], errors="coerce", dayfirst=True).dt.date

        # Vérification métier
        comptes_autorises = {
                "achat": [
                    "401","403","404","408","409",
                    "601","602","606","607","608","609",
                    "44562","44566"
                ],
                "vente": [
                    "411","413","418","419",
                    "701","702","703","706","707","708","709",
                    "44571"
                ],
                "banque": [
                    "512","514","401","411","421","431","445","581"
                ],
                "caisse": [
                    "53","401","411","512"
                ]
        }

        comptes_ok = comptes_autorises.get(type_journal, [])
        for idx, row in df_journal.iterrows():
            compte = str(row["Numéro de compte"])
            if not any(compte.startswith(c) for c in comptes_ok):
                df_journal.at[idx, "Numéro de compte"] = "???"

        return df_journal
    except:
        return pd.DataFrame()

def traiter_image(image_path, pcg_contenu):
    st.markdown(f"### Traitement : {os.path.basename(image_path)}")
    img = preprocess_image(image_path)
    result = reader.readtext(img)
    texte_complet = " ".join([text for _, text, _ in result])
    st.write("Texte OCR extrait :")
    st.write(texte_complet)

    champs = extraire_champs(texte_complet)
    champs['texte_brut'] = texte_complet
    champs['fichier'] = os.path.basename(image_path)
    champs['type_journal'] = classifier_avec_gpt(texte_complet)
    champs['compte'] = detecter_compte(texte_complet, pcg_contenu)
    champs['journal_markdown'] = generer_journal_avec_llm(texte_complet, pcg_contenu, champs['type_journal'])
    return champs

def aggreger_en_grand_journal(donnees, fichier_sortie="grand_journal.csv"):
    journaux = []
    for ligne in donnees:
        if isinstance(ligne.get("journal_markdown"), pd.DataFrame):
            df_piece = ligne["journal_markdown"].copy()
            df_piece["Type journal"] = ligne.get("type_journal","inconnu")
            df_piece["Référence"] = ligne.get("numero_piece","")
            journaux.append(df_piece)
    if not journaux:
        return pd.DataFrame()
    
    grand_journal = pd.concat(journaux, ignore_index=True)
    colonnes = ["Date","Référence","Numéro de compte","Libellé","Débit (Ar)","Crédit (Ar)","Type journal"]
    grand_journal = grand_journal.reindex(columns=colonnes)
    
    # Nettoyer les colonnes Débit et Crédit (enlever "Ar", espaces, etc.)
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        grand_journal[col] = (
            grand_journal[col]
            .astype(str)
            .str.replace(r"[^\d.,-]", "", regex=True)  # garder seulement chiffres
            .str.replace(",", ".", regex=False)        # remplacer , par .
        )
        grand_journal[col] = pd.to_numeric(grand_journal[col], errors="coerce").fillna(0)

    # Normaliser les dates
    grand_journal["Date"] = pd.to_datetime(grand_journal["Date"], errors="coerce", dayfirst=True).dt.date
    
    # Trier par date
    grand_journal = grand_journal.sort_values(by="Date").reset_index(drop=True)
    grand_journal.index = grand_journal.index + 1
    
    # Export CSV
    grand_journal.to_csv(fichier_sortie,index=False,encoding="utf-8-sig")
    return grand_journal

def generer_grand_livre(df_grand_journal, fichier_sortie="grand_livre.csv"):
    """
    Génère le Grand Livre à partir du Grand Journal.
    """
    if df_grand_journal.empty:
        return pd.DataFrame()

    # Nettoyer les colonnes Débit et Crédit pour enlever "Ar" et espaces
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        df_grand_journal[col] = (
            df_grand_journal[col]
            .astype(str)
            .str.replace(r"[^\d.,-]", "", regex=True)  # garder que chiffres
            .str.replace(",", ".", regex=False)        # remplacer , par .
        )
        df_grand_journal[col] = pd.to_numeric(df_grand_journal[col], errors="coerce").fillna(0)

    # Liste des comptes uniques
    comptes = df_grand_journal["Numéro de compte"].unique()
    grand_livre = []

    for compte in comptes:
        df_compte = df_grand_journal[df_grand_journal["Numéro de compte"] == compte].copy()
        df_compte = df_compte.sort_values(by="Date").reset_index(drop=True)
        # Calcul du solde cumulé
        df_compte["Solde"] = (df_compte["Débit (Ar)"] - df_compte["Crédit (Ar)"]).cumsum()
        df_compte["Compte"] = compte
        grand_livre.append(df_compte)

    df_grand_livre = pd.concat(grand_livre, ignore_index=True)
    colonnes = ["Compte","Date","Référence","Libellé","Débit (Ar)","Crédit (Ar)","Solde"]
    df_grand_livre = df_grand_livre.reindex(columns=colonnes)

    # ✅ Version affichage formatée (ex : "200 000 Ar")
    for col in ["Débit (Ar)", "Crédit (Ar)", "Solde"]:
        df_grand_livre[col] = df_grand_livre[col].apply(lambda x: f"{int(x):,} Ar".replace(",", " ") if x != 0 else "")

    # Export CSV (avec nombres bruts pour analyse)
    df_export = pd.concat(grand_livre, ignore_index=True)
    df_export = df_export.reindex(columns=colonnes)
    df_export.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

    return df_grand_livre

def generer_lettrage(df_grand_journal, fichier_sortie="lettrage.csv"):
    """
    Rapproche débits/crédits par compte (lettrage).
    - supporte lettrage exact et partiel (fractionnement logique).
    - retourne un dataframe avec colonnes: Numéro de compte, Date, Référence, Libellé,
      Débit (Ar), Crédit (Ar), Solde partiel, Statut, Lettrage
    """
    if df_grand_journal is None or df_grand_journal.empty:
        return pd.DataFrame()

    df = df_grand_journal.copy()

    # Nettoyage colonnes numériques
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^\d\-,.]", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Normaliser date pour le tri (si existante)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        df["Date"] = pd.NaT

    résultats = []

    # Pour chaque compte, on tente de lettrer
    for compte, grp in df.groupby("Numéro de compte", sort=False):
        df_c = grp.sort_values(by=["Date", "Référence"]).reset_index(drop=True)

        # Montant signé et absolu
        df_c["amount"] = df_c["Débit (Ar)"] - df_c["Crédit (Ar)"]
        df_c["abs_amount"] = df_c["amount"].abs()
        df_c["sign"] = df_c["amount"].apply(lambda x: "D" if x > 0 else ("C" if x < 0 else "0"))
        df_c["remaining"] = df_c["abs_amount"].copy()
        df_c["Lettrage"] = ""  # on stockera des id L1;L2...
        
        # files: debit queue (positive), credit queue (positive absolut)
        deb_q = deque([(i, row["remaining"]) for i, row in df_c[df_c["sign"] == "D"].iterrows()])
        cred_q = deque([(i, row["remaining"]) for i, row in df_c[df_c["sign"] == "C"].iterrows()])

        let_idx = 1
        # rapprochement (greedy) : on consomme les files
        while deb_q and cred_q:
            i_idx, i_amt = deb_q.popleft()
            j_idx, j_amt = cred_q.popleft()

            # tolérance (par sécurité float)
            if abs(i_amt - j_amt) < 1e-6:
                tag = f"L{let_idx}"
                # append tag
                df_c.at[i_idx, "Lettrage"] = (df_c.at[i_idx, "Lettrage"] + ";" + tag) if df_c.at[i_idx, "Lettrage"] else tag
                df_c.at[j_idx, "Lettrage"] = (df_c.at[j_idx, "Lettrage"] + ";" + tag) if df_c.at[j_idx, "Lettrage"] else tag
                df_c.at[i_idx, "remaining"] = 0
                df_c.at[j_idx, "remaining"] = 0
                let_idx += 1
            elif i_amt > j_amt:
                tag = f"L{let_idx}"
                df_c.at[j_idx, "Lettrage"] = (df_c.at[j_idx, "Lettrage"] + ";" + tag) if df_c.at[j_idx, "Lettrage"] else tag
                df_c.at[j_idx, "remaining"] = 0
                new_i = i_amt - j_amt
                df_c.at[i_idx, "remaining"] = new_i
                # remettre le débit restant en tête (on continue à le rapprocher)
                deb_q.appendleft((i_idx, new_i))
                let_idx += 1
            else:  # i_amt < j_amt
                tag = f"L{let_idx}"
                df_c.at[i_idx, "Lettrage"] = (df_c.at[i_idx, "Lettrage"] + ";" + tag) if df_c.at[i_idx, "Lettrage"] else tag
                df_c.at[i_idx, "remaining"] = 0
                new_j = j_amt - i_amt
                df_c.at[j_idx, "remaining"] = new_j
                cred_q.appendleft((j_idx, new_j))
                let_idx += 1

        # Statut final selon remaining
        def statut_ligne(r):
            if r["abs_amount"] == 0:
                return "Zéro"
            if r["remaining"] == 0:
                return "Soldé"
            if r["remaining"] < r["abs_amount"]:
                return "Partiellement lettré"
            return "Non soldé"

        df_c["Statut"] = df_c.apply(statut_ligne, axis=1)

        # Solde partiel signé : remaining avec signe
        def solde_partiel_signed(r):
            if r["sign"] == "D":
                return r["remaining"]
            if r["sign"] == "C":
                return -r["remaining"]
            return 0
        df_c["Solde partiel"] = df_c.apply(solde_partiel_signed, axis=1)

        résultats.append(df_c)

    if len(résultats) == 0:
        return pd.DataFrame()

    df_out = pd.concat(résultats, ignore_index=True, sort=False)

    # Colonnes de sortie dans l'ordre souhaité
    colonnes = ["Numéro de compte", "Date", "Référence", "Libellé",
                "Débit (Ar)", "Crédit (Ar)", "Solde partiel", "Statut", "Lettrage"]
    for c in colonnes:
        if c not in df_out.columns:
            df_out[c] = ""

    df_out = df_out[colonnes]
    df_out.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    return df_out


# =====================
# Rapprochement bancaire
# =====================

reader = easyocr.Reader(['fr'])

def pdf_image_to_df(file):
    pages = convert_from_path(file) if file.name.endswith(".pdf") else [file]
    lignes = []
    for page in pages:
        img = np.array(page)
        result = reader.readtext(img)
        for (_, texte, _) in result:
            lignes.append(texte)
    data = []
    for l in lignes:
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', l)
        montant_match = re.search(r'[-]?\d+(?:[.,]\d{2})?', l)
        if date_match and montant_match:
            date = pd.to_datetime(date_match.group(), dayfirst=True).date()
            montant = float(montant_match.group().replace(',', '.'))
            libelle = l.replace(date_match.group(), '').replace(montant_match.group(), '').strip()
            data.append([date, libelle, montant])
    df = pd.DataFrame(data, columns=["Date", "Libellé", "Montant"])
    return df


def rapprochement_bancaire(df_journal, df_releve, seuil_jours=3, seuil_montant=100):
    df_journal = df_journal.copy()
    df_releve = df_releve.copy()
    df_journal["Date"] = pd.to_datetime(df_journal["Date"], errors="coerce").dt.date
    df_journal["Montant"] = df_journal.get("Débit (Ar)", 0) - df_journal.get("Crédit (Ar)", 0)
    df_journal["Statut"] = "Non rapproché"
    df_releve["Statut"] = "Non rapproché"

    for i, row_j in df_journal.iterrows():
        for j, row_r in df_releve.iterrows():
            if row_r["Statut"] == "Rapproché":
                continue
            date_ok = abs((row_j["Date"] - row_r["Date"]).days) <= seuil_jours
            montant_ok = abs(row_j["Montant"] - row_r["Montant"]) <= seuil_montant
            if date_ok and montant_ok:
                df_journal.at[i, "Statut"] = "Rapproché"
                df_releve.at[j, "Statut"] = "Rapproché"
                break

    df_result = pd.concat([
        df_journal[["Date", "Libellé", "Montant", "Statut"]],
        df_releve.rename(columns={"Libellé":"Libellé_releve"})[["Date","Libellé_releve","Montant","Statut"]]
    ], axis=1)
    return df_result


import pandas as pd

def generer_balance(df_grand_journal, fichier_sortie="balance.csv"):
    """
    Génère la balance comptable à partir du Grand Journal.
    Retourne un DataFrame avec :
    - Numéro de compte
    - Débit total
    - Crédit total
    - Solde Débiteur
    - Solde Créditeur
    """

    if df_grand_journal.empty:
        return pd.DataFrame()

    df = df_grand_journal.copy()

    # Vérifier colonnes obligatoires
    colonnes_attendues = ["Numéro de compte", "Débit (Ar)", "Crédit (Ar)"]
    for col in colonnes_attendues:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le fichier : {col}")

    # Nettoyage colonnes numériques
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[^\d\-,.]", "", regex=True)   # garder seulement chiffres, signe, point, virgule
            .str.replace(",", ".", regex=False)         # uniformiser les décimales
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Grouper par compte
    df_balance = df.groupby("Numéro de compte", as_index=False).agg({
        "Débit (Ar)": "sum",
        "Crédit (Ar)": "sum"
    })

    # Calcul solde et séparation Débiteur/Créditeur
    df_balance["Solde"] = df_balance["Débit (Ar)"] - df_balance["Crédit (Ar)"]
    df_balance["Solde Débiteur"] = df_balance["Solde"].apply(lambda x: x if x > 0 else 0)
    df_balance["Solde Créditeur"] = df_balance["Solde"].apply(lambda x: -x if x < 0 else 0)
    df_balance.drop(columns=["Solde"], inplace=True)

    # Formatage en Ar
    for col in ["Débit (Ar)", "Crédit (Ar)", "Solde Débiteur", "Solde Créditeur"]:
        df_balance[col] = df_balance[col].apply(lambda x: f"{int(x):,} Ar".replace(",", " "))

    # Export CSV
    df_balance.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

    return df_balance

def parse_montant(val):
    """
    Convertit un montant (ex: '1 200,50 Ar') en float.
    Retourne 0.0 si la valeur n'est pas convertible.
    """
    if pd.isna(val):  # si valeur NaN
        return 0.0
    if isinstance(val, str):
        # Nettoyage du texte
        val = val.replace("Ar", "").replace(" ", "").replace("\u202f", "")
        val = val.replace(",", ".")
    try:
        return float(val)
    except ValueError:
        return 0.0
    
def generer_bilan(df_balance_brut, fichier_sortie="bilan.csv"):
    """
    Génère un bilan simplifié (Actif / Passif) à partir de la balance fournie.
    df_balance_brut : DataFrame retourné par generer_balance (colonnes Débit (Ar), Crédit (Ar))
    Retour : DataFrame du bilan (Actif / Passif) et sauvegarde CSV.
    """
    if df_balance_brut is None or df_balance_brut.empty:
        return pd.DataFrame()

    df = df_balance_brut.copy()

    # Vérifier colonnes
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        if col not in df.columns:
            df[col] = ""

    # Convertir en numérique
    df["_debit_num"] = df["Débit (Ar)"].apply(parse_montant)
    df["_credit_num"] = df["Crédit (Ar)"].apply(parse_montant)
    df["_net"] = df["_debit_num"] - df["_credit_num"]  # positif = Actif, négatif = Passif

    # Classification simple par préfixe
    def classifier_compte(numero):
        n = str(numero).strip()
        if not n:
            return "Non classé"
        if n.startswith("2"):
            return "Actif immobilisé"
        if n.startswith("3"):
            return "Stocks"
        if n.startswith("41") or n.startswith("46"):
            return "Créances clients et divers"
        if n.startswith("40") or n.startswith("42") or n.startswith("44"):
            return "Dettes fournisseurs / Tiers"
        if n.startswith("5") or n.startswith("53") or n.startswith("512"):
            return "Disponibilités (Banque / Caisse)"
        if n.startswith("1"):
            return "Capitaux propres et assimilés"
        if n.startswith("16") or n.startswith("17") or n.startswith("19"):
            return "Emprunts et dettes financières"
        return "Autres"

    df["Poste"] = df["Numéro de compte"].apply(classifier_compte)

    # Construire total par poste
    postes = {}
    for _, row in df.iterrows():
        poste = row["Poste"]
        net = row["_net"]
        if poste not in postes:
            postes[poste] = {"actif": 0.0, "passif": 0.0}
        if net > 0:
            postes[poste]["actif"] += net
        elif net < 0:
            postes[poste]["passif"] += -net

    # Construire DataFrame bilan
    lignes = []
    for poste, vals in postes.items():
        lignes.append({
            "Poste": poste,
            "Actif (Ar)": vals["actif"],
            "Passif (Ar)": vals["passif"]
        })
    df_bilan = pd.DataFrame(lignes).sort_values(by="Poste").reset_index(drop=True)

    # Totaux
    total_actif = df_bilan["Actif (Ar)"].sum()
    total_passif = df_bilan["Passif (Ar)"].sum()

    totals_row = pd.DataFrame([{
        "Poste": "TOTAL",
        "Actif (Ar)": total_actif,
        "Passif (Ar)": total_passif
    }])
    df_bilan_aff = pd.concat([df_bilan, totals_row], ignore_index=True)

    # Mise en forme
    df_bilan_aff["Actif (Ar)"] = df_bilan_aff["Actif (Ar)"].apply(lambda x: f"{int(round(x)):,}".replace(",", " ") + " Ar" if x != 0 else "")
    df_bilan_aff["Passif (Ar)"] = df_bilan_aff["Passif (Ar)"].apply(lambda x: f"{int(round(x)):,}".replace(",", " ") + " Ar" if x != 0 else "")

    # Sauvegarde CSV
    df_bilan_aff.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

    return df_bilan_aff, total_actif, total_passif


def generer_compte_resultat(df_balance, fichier_sortie="compte_resultat.csv"):
    """
    Génère un compte de résultat simplifié (Charges / Produits).
    """
    if df_balance is None or df_balance.empty:
        return pd.DataFrame()

    df = df_balance.copy()

    # Nettoyer montants
    df["_debit_num"] = df["Débit (Ar)"].apply(parse_montant)
    df["_credit_num"] = df["Crédit (Ar)"].apply(parse_montant)
    df["_net"] = df["_debit_num"] - df["_credit_num"]

    charges = df[df["Numéro de compte"].astype(str).str.startswith("6")]
    produits = df[df["Numéro de compte"].astype(str).str.startswith("7")]

    total_charges = charges["_debit_num"].sum()
    total_produits = produits["_credit_num"].sum()
    resultat_net = total_produits - total_charges

    # Préparer tableau résultat
    data = {
        "Charges (classe 6)": [f"{int(round(total_charges)):,}".replace(",", " ") + " Ar"],
        "Produits (classe 7)": [f"{int(round(total_produits)):,}".replace(",", " ") + " Ar"],
        "Résultat Net": [f"{int(round(resultat_net)):,}".replace(",", " ") + " Ar"]
    }

    df_cr = pd.DataFrame(data)

    # Export CSV
    df_cr.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

    return df_cr, total_charges, total_produits, resultat_net


def generer_annexe(df_grand_livre, fichier_sortie="annexe.csv"):
    """
    Génère une annexe simplifiée à partir du grand livre.
    """
    if df_grand_livre.empty:
        return pd.DataFrame()
    
    # Immobilisations (comptes 2xx)
    immobilisations = df_grand_livre[df_grand_livre["Compte"].str.startswith("2")]
    immobilisations_detail = immobilisations.groupby("Compte").agg({
        "Débit (Ar)": "sum",
        "Crédit (Ar)": "sum"
    }).reset_index()
    immobilisations_detail["Valeur nette"] = immobilisations_detail["Débit (Ar)"] - immobilisations_detail["Crédit (Ar)"]

    # Clients (411) et Fournisseurs (401)
    clients = df_grand_livre[df_grand_livre["Compte"].str.startswith("411")].groupby("Compte").agg({"Solde": "sum"}).reset_index()
    fournisseurs = df_grand_livre[df_grand_livre["Compte"].str.startswith("401")].groupby("Compte").agg({"Solde": "sum"}).reset_index()

    # Provisions (compte 15xx)
    provisions = df_grand_livre[df_grand_livre["Compte"].str.startswith("15")].groupby("Compte").agg({"Solde": "sum"}).reset_index()

    # Export CSV
    with pd.ExcelWriter(fichier_sortie) as writer:
        immobilisations_detail.to_excel(writer, sheet_name="Immobilisations", index=False)
        clients.to_excel(writer, sheet_name="Clients", index=False)
        fournisseurs.to_excel(writer, sheet_name="Fournisseurs", index=False)
        provisions.to_excel(writer, sheet_name="Provisions", index=False)
    
    return {
        "Immobilisations": immobilisations_detail,
        "Clients": clients,
        "Fournisseurs": fournisseurs,
        "Provisions": provisions
    }

def generer_amortissement(df_grand_journal, fichier_sortie="amortissement.csv", duree_defaut=5):
    """
    Génère automatiquement un plan d’amortissement linéaire
    pour toutes les factures d’immobilisations (classe 2).
    - duree_defaut = 5 ans si la durée n’est pas précisée
    """
    if df_grand_journal.empty:
        return pd.DataFrame()

    # Nettoyer colonnes numériques
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        df_grand_journal[col] = (
            df_grand_journal[col].astype(str)
            .str.replace(r"[^\d\-,.]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df_grand_journal[col] = pd.to_numeric(df_grand_journal[col], errors="coerce").fillna(0)

    # Filtrer immobilisations (comptes 2xxx en Débit)
    
            # Si compte 2xxx
    df_immo = df_grand_journal[df_grand_journal["Compte"].astype(str).str.startswith("2")].copy()
    
            # Si n'importe quel compte
    # df_immo = df_grand_journal[df_grand_journal['Compte'].str.match(r'^\d+$')].copy()
    # Vérification debug
    st.write("Nombre de lignes immobilisations :", df_immo.shape[0])
    
    plans = []
    for _, row in df_immo.iterrows():
        date_acq = pd.to_datetime(row["Date"], errors="coerce")
        montant = row["Débit (Ar)"] if row["Débit (Ar)"] > 0 else row["Crédit (Ar)"]

        if pd.isna(date_acq) or montant <= 0:
            continue

        duree = duree_defaut
        annuite = montant / duree

        for i in range(1, duree + 1):
            annee = date_acq.year + i - 1
            plans.append({
                "Référence": row.get("Référence", ""),
                "Date acquisition": date_acq.date(),
                "Compte immobilisation": row["Compte"], 
                "Montant acquisition": montant,
                "Année": annee,
                "Durée (ans)": duree,
                "Annuité": round(annuite, 2),
                "Cumul amortissement": round(annuite * i, 2),
                "Valeur nette comptable": round(montant - annuite * i, 2)
            })

    # Créer le DataFrame final
    df_amort = pd.DataFrame(plans)

    # Sauvegarde si non vide
    if not df_amort.empty:
        df_amort.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

    return df_amort 

# ==========================
# INTERFACE STREAMLIT
# ==========================
st.markdown('<h2 style="color: green; text-align:center;">Automatisation des Processus Comptables</h2>', unsafe_allow_html=True)

# Section 1: Upload et traitement des fichiers
st.markdown("<h5>Upload et Traitement des Documents</h5>", unsafe_allow_html=True)
fichiers = st.file_uploader("Upload fichiers (images, PDF, CSV, Excel)", type=["png","jpg","jpeg","pdf","csv","xlsx"], accept_multiple_files=True)

if st.button("Traiter") and fichiers:
    donnees = []
    for fichier in fichiers:
        extension = os.path.splitext(fichier.name)[1].lower()
        temp_path = "temp_" + fichier.name
        with open(temp_path,"wb") as f:
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
                for i,page in enumerate(pages):
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
        df = detecter_doublons(df)
        st.session_state.df_final = aggreger_en_grand_journal(donnees, CSV_OUTPUT)
        st.success(f"✅ Extraction terminée, exporté dans {CSV_OUTPUT}")

        types_dispo = st.session_state.df_final["Type journal"].unique().tolist()
        for tj in types_dispo:
            st.markdown(f"Type Journal : {tj.capitalize()}")
            df_tj = st.session_state.df_final[st.session_state.df_final["Type journal"]==tj].copy()
            df_tj = df_tj.sort_values(by="Date").reset_index(drop=True)
            st.dataframe(df_tj)

        st.markdown("<h4> Grand Journal complet </h4>", unsafe_allow_html=True)
        st.dataframe(st.session_state.df_final)
        st.download_button(
            "⬇️ Télécharger le Grand Journal",
            data=st.session_state.df_final.to_csv(index=False, encoding="utf-8-sig"),
            file_name=CSV_OUTPUT,
            mime="text/csv"
        )

        st.markdown("<h4> Grand Livre </h4>", unsafe_allow_html=True)
        df_grand_livre = generer_grand_livre(st.session_state.df_final, "grand_livre.csv")
        st.dataframe(df_grand_livre)
        st.download_button(
            "⬇️ Télécharger le Grand Livre",
            data=df_grand_livre.to_csv(index=False, encoding="utf-8-sig"),
            file_name="grand_livre.csv",
            mime="text/csv"
        )

        # Titre
        st.markdown("<h4>Lettrage des comptes</h4>", unsafe_allow_html=True)

        # Génération du lettrage à partir du grand journal final
        df_lettrage = generer_lettrage(st.session_state.df_final, "lettrage.csv")

        # Affichage dans Streamlit
        st.dataframe(
            df_lettrage.style.format({
                "Débit (Ar)": "{:,.0f}".format,
                "Crédit (Ar)": "{:,.0f}".format,
                "Solde partiel": "{:,.0f}".format
            })
        )

        # Bouton de téléchargement
        st.download_button(
            "⬇️ Télécharger le Lettrage",
            data=df_lettrage.to_csv(index=False, encoding="utf-8-sig"),
            file_name="lettrage.csv",
            mime="text/csv"
        )

    st.markdown("<h4>Rapprochement bancaire</h4>", unsafe_allow_html=True)

        # Upload Grand Livre
    uploaded_journal = st.file_uploader("Importer Grand Livre (CSV ou Excel)", type=["csv","xlsx"])
    # Upload Relevé bancaire
    uploaded_releve = st.file_uploader("Importer Relevé bancaire (CSV, Excel ou PDF/image)", type=["csv","xlsx","pdf","png","jpg","jpeg"])

    df_journal, df_releve = None, None

    if uploaded_journal:
        if uploaded_journal.name.endswith(".csv"):
            df_journal = pd.read_csv(uploaded_journal)
        else:
            df_journal = pd.read_excel(uploaded_journal)

    if uploaded_releve:
        if uploaded_releve.name.endswith((".csv", ".xlsx")):
            df_releve = pd.read_csv(uploaded_releve) if uploaded_releve.name.endswith(".csv") else pd.read_excel(uploaded_releve)
            df_releve.rename(columns={"Débit (Ar)":"Montant","Crédit (Ar)":"Montant"}, inplace=True)
        else:
            df_releve = pdf_image_to_df(uploaded_releve)


    # Lancer rapprochement
    if df_journal is not None and df_releve is not None:
        df_result = rapprochement_bancaire(df_journal, df_releve)
        st.dataframe(df_result)
        st.download_button("⬇️ Télécharger le rapprochement", df_result.to_csv(index=False, encoding="utf-8-sig"), "rapprochement.csv", "text/csv")



    st.markdown("<h4>Balance comptable</h4>", unsafe_allow_html=True)

    # Balance brute (avec nombres pour export CSV)
    df_balance_brut = generer_balance(st.session_state.df_final, "balance.csv")

    # Créer une copie formatée pour affichage
    df_balance_affichage = df_balance_brut.copy()
    for col in ["Débit (Ar)", "Crédit (Ar)", "Solde Débiteur", "Solde Créditeur"]:
        df_balance_affichage[col] = df_balance_affichage[col].apply(
            lambda x: f"{int(str(x).replace(' Ar','').replace(' ','') or 0):,} Ar"
        )

    # Afficher dans Streamlit
    st.dataframe(df_balance_affichage)

    # Bouton téléchargement CSV (avec les valeurs numériques brutes)
    st.download_button(
        "⬇️ Télécharger la Balance",
        data=df_balance_brut.to_csv(index=False, encoding="utf-8-sig"),
        file_name="balance.csv",
        mime="text/csv"
    )


    # Générer le bilan
    df_bilan, tot_actif, tot_passif = generer_bilan(df_balance_brut, "bilan.csv")

    st.markdown("<h4 style='margin-top:20px;'>📊 Bilan simplifié</h4>", unsafe_allow_html=True)

    # Affichage du tableau
    st.dataframe(df_bilan, use_container_width=True)

    # Vérification équilibre
    if int(round(tot_actif)) == int(round(tot_passif)):
        st.success(f"✅ Bilan équilibré : Actif = Passif = {int(round(tot_actif)):,} Ar".replace(",", " "))
    else:
        st.warning(f"⚠️ Bilan déséquilibré : Actif = {int(round(tot_actif)):,} Ar, "
                f"Passif = {int(round(tot_passif)):,} Ar".replace(",", " "))

    # Bouton de téléchargement
    st.download_button(
        "⬇️ Télécharger le Bilan",
        data=df_bilan.to_csv(index=False, encoding="utf-8-sig"),
        file_name="bilan.csv",
        mime="text/csv"
    )

    # ============================
    # 2. Compte de Résultat
    # ============================

    st.markdown("<h3 style='margin-top:30px;'>📗 Compte de Résultat</h3>", unsafe_allow_html=True)

    df_cr, total_charges, total_produits, resultat_net = generer_compte_resultat(df_balance_brut, "compte_resultat.csv")

    if df_cr is not None and not df_cr.empty:
        st.dataframe(df_cr, use_container_width=True)

        if resultat_net > 0:
            st.success(f"✅ Résultat Net : Bénéfice de {int(round(resultat_net)):,} Ar".replace(",", " "))
        elif resultat_net < 0:
            st.error(f"❌ Résultat Net : Perte de {int(round(abs(resultat_net))):,} Ar".replace(",", " "))
        else:
            st.info(" Résultat Net = 0 (ni bénéfice ni perte)")

        st.download_button(
            "⬇️ Télécharger le Compte de Résultat",
            data=df_cr.to_csv(index=False, encoding="utf-8-sig"),
            file_name="compte_resultat.csv",
            mime="text/csv"
        )
    else:
        st.info(" Aucun compte de résultat généré. Vérifiez vos données.")


    # Supposons que tu as déjà généré ton annexe
    annexe = generer_annexe(df_grand_livre)

    # Affichage par onglet pour chaque section
    tabs = st.tabs(["Immobilisations", "Clients", "Fournisseurs", "Provisions"])

    with tabs[0]:
        st.markdown("<h4>Immobilisations</h4>", unsafe_allow_html=True)
        st.dataframe(annexe["Immobilisations"])

    with tabs[1]:
        st.markdown("<h4>Clients</h4>", unsafe_allow_html=True)
        st.dataframe(annexe["Clients"])

    with tabs[2]:
        st.markdown("<h4>Fournisseurs</h4>", unsafe_allow_html=True)
        st.dataframe(annexe["Fournisseurs"])

    with tabs[3]:
        st.markdown("<h4>Provisions</h4>", unsafe_allow_html=True)
        st.dataframe(annexe["Provisions"])
        
            # Génération du plan d'amortissement
        df_amortissement = generer_amortissement(df_grand_livre)
        st.markdown("<h4>📊 Plan d'Amortissement des Immobilisations</h4>", unsafe_allow_html=True)
        if df_amortissement is not None and not df_amortissement.empty:
        # Affichage complet dans un seul tableau
            st.dataframe(df_amortissement)
            st.download_button(
                "⬇️ Télécharger le Plan d'Amortissement",
                data=df_amortissement.to_csv(index=False, encoding="utf-8-sig"),
                file_name="amortissement.csv",
                mime="text/csv"
            )
        else:
            st.info("Aucun amortissement généré.")