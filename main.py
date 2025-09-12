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

# ==========================
# CONFIGURATION
# ==========================
load_dotenv()
CSV_OUTPUT = "grand_journal.csv"
PDF_PCG = "D:/OCR/CompabliteIA/plan-comptable-general-2005.pdf"

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

Répond uniquement par un mot exact parmi : achat, vente, banque, caisse, OD
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


def detecter_compte(texte_ocr, contenu_pcg):
    prompt = f"""
Tu es un expert du Plan Comptable Général malgache 2005.

Texte extrait :
\"\"\"{texte_ocr}\"\"\" 

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

def generer_journal_avec_llm(texte_ocr, pcg_contenu):
    prompt = f"""
Tu es un expert comptable selon le PCG 2005.

Pièce :
\"\"\"{texte_ocr}\"\"\" 

Génère un tableau avec 5 colonnes : Date, Numéro de compte, Libellé, Débit, Crédit
- La colonne Date doit contenir uniquement la date au format JJ/MM/AAAA
- Ne répète pas les en-têtes
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
        if len(df_journal) > 0:
            premiere_ligne = df_journal.iloc[0]
            mots_cles = ['date','compte','libellé','libelle','débit','debit','crédit','credit']
            if any(mot in " ".join(str(v).lower() for v in premiere_ligne.values) for mot in mots_cles):
                df_journal = df_journal.drop(0).reset_index(drop=True)
        df_journal = df_journal.dropna(how='all').reset_index(drop=True)
        df_journal.index = df_journal.index + 1
        if "Date" in df_journal.columns:
            df_journal["Date"] = pd.to_datetime(df_journal["Date"], errors="coerce", dayfirst=True).dt.date
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
    champs['journal_markdown'] = generer_journal_avec_llm(texte_complet, pcg_contenu)
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


def ocr_releve_bancaire(fichier):
    """
    Extrait un relevé bancaire via OCR (PDF ou image).
    Retourne un DataFrame avec colonnes Date, Libellé, Montant
    """
    texte_total = ""
    extension = os.path.splitext(fichier)[1].lower()

    if extension == ".pdf":
        doc = fitz.open(fichier)
        for page in doc:
            texte_total += page.get_text()
    else:  # image (png, jpg, jpeg)
        img = preprocess_image(fichier)
        result = reader.readtext(img)
        texte_total = " ".join([text for _, text, _ in result])

    # Extraction des dates et montants
    lignes = texte_total.split("\n")
    donnees = []
    for ligne in lignes:
        date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", ligne)
        montant_match = re.search(r"[-+]?\d[\d\s.,]*", ligne)
        if date_match and montant_match:
            date_op = pd.to_datetime(date_match.group(), dayfirst=True, errors="coerce").date()
            montant = nettoyer_montant(montant_match.group())
            libelle = ligne.replace(date_match.group(), "").replace(montant_match.group(), "").strip()
            donnees.append({"Date": date_op, "Libellé": libelle, "Montant": montant})

    return pd.DataFrame(donnees)


def rapprochement_bancaire(df_grand_journal, fichier_releve, seuil_jours=3):
    """
    Compare les écritures du compte 512 Banque avec le relevé bancaire (CSV ou OCR).
    """
    extension = os.path.splitext(fichier_releve)[1].lower()

    if extension == ".csv":
        df_releve = pd.read_csv(fichier_releve)
        df_releve["Date"] = pd.to_datetime(df_releve["Date"], errors="coerce", dayfirst=True).dt.date
        df_releve["Montant"] = pd.to_numeric(df_releve["Montant"], errors="coerce").fillna(0)
    else:
        df_releve = ocr_releve_bancaire(fichier_releve)

    # Filtrer le compte banque dans le Grand Journal
    df_banque = df_grand_journal[df_grand_journal["Numéro de compte"].astype(str).str.startswith("512")].copy()
    df_banque["Montant"] = df_banque["Débit (Ar)"] - df_banque["Crédit (Ar)"]

    resultats = []
    for _, op in df_banque.iterrows():
        match = df_releve[
            (abs(pd.to_datetime(df_releve["Date"]) - pd.to_datetime(op["Date"])).dt.days <= seuil_jours) &
            (df_releve["Montant"].round(0) == round(op["Montant"], 0))
        ]
        if not match.empty:
            statut = "OK"
        else:
            statut = "Écart (non trouvé sur relevé)"
        resultats.append(statut)

    df_banque["Statut rapprochement"] = resultats

    # Vérifier aussi les opérations du relevé absentes en compta
    ecarts_releve = []
    for _, op in df_releve.iterrows():
        match = df_banque[
            (abs(pd.to_datetime(df_banque["Date"]) - pd.to_datetime(op["Date"])).dt.days <= seuil_jours) &
            (df_banque["Montant"].round(0) == round(op["Montant"], 0))
        ]
        if match.empty:
            ecarts_releve.append(op)
    df_ecarts_releve = pd.DataFrame(ecarts_releve)

    return df_banque, df_ecarts_releve

def generer_balance(df_grand_journal):
    """
    Génère la balance comptable à partir du Grand Journal.
    
    df_grand_journal : DataFrame issu de aggreger_en_grand_journal()
    
    Retourne : DataFrame balance
    """
    # Nettoyer colonnes montants
    df_grand_journal["Débit (Ar)"] = pd.to_numeric(df_grand_journal["Débit (Ar)"], errors="coerce").fillna(0)
    df_grand_journal["Crédit (Ar)"] = pd.to_numeric(df_grand_journal["Crédit (Ar)"], errors="coerce").fillna(0)

    # Regrouper par compte
    balance = df_grand_journal.groupby("Numéro de compte").agg({
        "Débit (Ar)": "sum",
        "Crédit (Ar)": "sum"
    }).reset_index()

    # Calcul des soldes
    balance["Solde Débiteur"] = balance["Débit (Ar)"] - balance["Crédit (Ar)"]
    balance["Solde Créditeur"] = balance["Crédit (Ar)"] - balance["Débit (Ar)"]

    # Si le solde est négatif, on remet à 0 dans la mauvaise colonne
    balance["Solde Débiteur"] = balance["Solde Débiteur"].apply(lambda x: x if x > 0 else 0)
    balance["Solde Créditeur"] = balance["Solde Créditeur"].apply(lambda x: x if x > 0 else 0)

    # Vérification équilibre
    total_debit = balance["Débit (Ar)"].sum()
    total_credit = balance["Crédit (Ar)"].sum()

    return balance, total_debit, total_credit

def travaux_inventaire(df_grand_journal, amortissements=[], provisions=[], charges_avance=[], produits_attendus=[]):
    """
    df_grand_journal : Grand Journal
    Les autres listes contiennent des dictionnaires de type :
    {"compte": "218", "libelle": "Amortissement ordinateur", "montant": 1000000, "date": "2025-12-31"}
    """
    df_modif = df_grand_journal.copy()
    
    for ecriture in amortissements + provisions + charges_avance + produits_attendus:
        df_modif = pd.concat([df_modif, pd.DataFrame([{
            "Date": ecriture["date"],
            "Numéro de compte": ecriture["compte"],
            "Libellé": ecriture["libelle"],
            "Débit (Ar)": ecriture.get("debit",0),
            "Crédit (Ar)": ecriture.get("credit",0),
            "Type journal": "OD",
            "Référence": ""
        }])], ignore_index=True)
    
    # Convertir les colonnes Débit/Crédit en numérique
    df_modif["Débit (Ar)"] = pd.to_numeric(df_modif["Débit (Ar)"], errors="coerce").fillna(0)
    df_modif["Crédit (Ar)"] = pd.to_numeric(df_modif["Crédit (Ar)"], errors="coerce").fillna(0)
    
    return df_modif

def generer_bilan(df_grand_journal):
    """
    Génère un Bilan simplifié à partir du Grand Journal.
    """
    df = df_grand_journal.copy()
    df["Débit (Ar)"] = pd.to_numeric(df["Débit (Ar)"], errors="coerce").fillna(0)
    df["Crédit (Ar)"] = pd.to_numeric(df["Crédit (Ar)"], errors="coerce").fillna(0)

    # Actif : comptes commençant par 1 à 5
    actif = df[df["Numéro de compte"].astype(str).str.match(r"^[1-5]")]
    actif_total = (actif["Débit (Ar)"] - actif["Crédit (Ar)"]).sum()

    # Passif : comptes commençant par 2 à 5
    passif = df[df["Numéro de compte"].astype(str).str.match(r"^[2-5]")]
    passif_total = (passif["Crédit (Ar)"] - passif["Débit (Ar)"]).sum()

    bilan = pd.DataFrame({
        "Catégorie": ["Actif", "Passif"],
        "Montant (Ar)": [actif_total, passif_total]
    })

    return bilan, actif_total, passif_total
def generer_compte_resultat(df_grand_journal):
    """
    Génère un Compte de Résultat simplifié à partir du Grand Journal.
    """
    df = df_grand_journal.copy()
    df["Débit (Ar)"] = pd.to_numeric(df["Débit (Ar)"], errors="coerce").fillna(0)
    df["Crédit (Ar)"] = pd.to_numeric(df["Crédit (Ar)"], errors="coerce").fillna(0)

    # Charges : comptes 6
    charges = df[df["Numéro de compte"].astype(str).str.startswith("6")]
    total_charges = charges["Débit (Ar)"].sum()

    # Produits : comptes 7
    produits = df[df["Numéro de compte"].astype(str).str.startswith("7")]
    total_produits = produits["Crédit (Ar)"].sum()

    resultat_net = total_produits - total_charges

    compte_resultat = pd.DataFrame({
        "Catégorie": ["Produits", "Charges", "Résultat Net"],
        "Montant (Ar)": [total_produits, total_charges, resultat_net]
    })

    return compte_resultat, total_produits, total_charges, resultat_net


# ==========================
# INTERFACE STREAMLIT
# ==========================
st.markdown('<h2 style="color: green; text-align:center;">Automatisation des Processus Comptables</h2>', unsafe_allow_html=True)
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
        df_final = aggreger_en_grand_journal(donnees, CSV_OUTPUT)
        st.success(f"✅ Extraction terminée, exporté dans {CSV_OUTPUT}")

        types_dispo = df_final["Type journal"].unique().tolist()
        for tj in types_dispo:
            st.markdown(f"Type Journal : {tj.capitalize()}")
            df_tj = df_final[df_final["Type journal"]==tj].copy()
            df_tj = df_tj.sort_values(by="Date").reset_index(drop=True)
            st.dataframe(df_tj)

        st.markdown("# Grand Journal complet")
        st.dataframe(df_final)
        st.download_button(
            "⬇️ Télécharger le Grand Journal",
            data=df_final.to_csv(index=False, encoding="utf-8-sig"),
            file_name=CSV_OUTPUT,
            mime="text/csv"
        )

        st.markdown("# Grand Livre")
        df_grand_livre = generer_grand_livre(df_final, "grand_livre.csv")
        st.dataframe(df_grand_livre)
        st.download_button(
            "⬇️ Télécharger le Grand Livre",
            data=df_grand_livre.to_csv(index=False, encoding="utf-8-sig"),
            file_name="grand_livre.csv",
            mime="text/csv"
        )


st.markdown("Rapprochement Bancaire")
fichier_releve = st.file_uploader("Uploader le relevé bancaire (CSV, PDF ou image)", type=["csv","pdf","png","jpg","jpeg"])

if fichier_releve:
    temp_releve = "temp_releve" + os.path.splitext(fichier_releve.name)[1]
    with open(temp_releve,"wb") as f:
        f.write(fichier_releve.read())

    df_banque, df_ecarts_releve = rapprochement_bancaire(df_final, temp_releve)

    st.markdown("Écritures du compte 512 avec statut de rapprochement")
    st.dataframe(df_banque)

    st.markdown("Opérations du relevé absentes en compta")
    st.dataframe(df_ecarts_releve)

    st.download_button(
        "⬇️ Exporter Résultat Rapprochement",
        data=df_banque.to_csv(index=False, encoding="utf-8-sig"),
        file_name="rapprochement_bancaire.csv",
        mime="text/csv"
    )

    os.remove(temp_releve)

st.markdown("Balance Comptable")

if st.button("Générer la Balance"):
    balance, total_debit, total_credit = generer_balance(df_final)

    st.dataframe(balance)

    st.write(f"✅ Total Débit : {total_debit:,.0f} Ar")
    st.write(f"✅ Total Crédit : {total_credit:,.0f} Ar")

    if total_debit == total_credit:
        st.success("La balance est équilibrée ✅")
    else:
        st.error("⚠️ La balance n'est pas équilibrée !")

    st.download_button(
        "⬇️ Exporter la Balance en CSV",
        data=balance.to_csv(index=False, encoding="utf-8-sig"),
        file_name="balance_comptable.csv",
        mime="text/csv"
    )

st.markdown(" Travaux d'inventaire")

if st.button("Appliquer Travaux d'Inventaire"):
    # Exemple simple : 1 amortissement et 1 provision
    amortissements = [{"compte":"218", "libelle":"Amortissement ordinateur", "credit":1000000, "date":"2025-12-31"}]
    provisions = [{"compte":"681", "libelle":"Provision clients douteux", "debit":200000, "date":"2025-12-31"}]

    df_modifie = travaux_inventaire(df_final, amortissements=amortissements, provisions=provisions)
    st.dataframe(df_modifie)

    st.download_button(
        "⬇️ Exporter Grand Journal après inventaire",
        data=df_modifie.to_csv(index=False, encoding="utf-8-sig"),
        file_name="grand_journal_inventaire.csv",
        mime="text/csv"
    )

st.markdown("États Financiers")

if st.button("Générer Bilan et Compte de Résultat"):
    bilan, actif_total, passif_total = generer_bilan(df_modifie)
    compte_resultat, total_produits, total_charges, resultat_net = generer_compte_resultat(df_modifie)

    st.markdown("### Bilan")
    st.dataframe(bilan)
    st.write(f"Total Actif : {actif_total:,.0f} Ar")
    st.write(f"Total Passif : {passif_total:,.0f} Ar")

    st.markdown("### Compte de Résultat")
    st.dataframe(compte_resultat)
    st.write(f"Total Produits : {total_produits:,.0f} Ar")
    st.write(f"Total Charges : {total_charges:,.0f} Ar")
    st.write(f"Résultat Net : {resultat_net:,.0f} Ar")

    st.download_button(
        "⬇️ Exporter Bilan",
        data=bilan.to_csv(index=False, encoding="utf-8-sig"),
        file_name="bilan.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Exporter Compte de Résultat",
        data=compte_resultat.to_csv(index=False, encoding="utf-8-sig"),
        file_name="compte_resultat.csv",
        mime="text/csv"
    )

