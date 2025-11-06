
import os
import re
import cv2
import fitz
import psycopg2
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
from psycopg2.extras import execute_values
import psycopg2
import os
import hashlib


# ==========================
# CONFIGURATION
# ==========================


# CHARGEMENT VARIABLES .ENV
# ============================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL non trouvé dans .env")
    exit()

# ==========================
# CRÉATION TABLES + CONNEXION OUVERTE
# ==========================
def creer_toutes_les_tables_supabase():
    """
    Crée toutes les tables comptables dans Supabase.
    Retourne la connexion OUVERTE pour réutilisation.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS public.grand_journal (
                id SERIAL PRIMARY KEY,
                "Date" DATE,
                "Référence" VARCHAR,
                "Numéro de compte" VARCHAR,
                "Libellé" VARCHAR,
                "Débit (Ar)" DECIMAL(15,2),
                "Crédit (Ar)" DECIMAL(15,2),
                "Type journal" VARCHAR,
                "Fichier source" VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.grand_livre (
                id SERIAL PRIMARY KEY,
                "Compte" VARCHAR,
                "Date" DATE,
                "Référence" VARCHAR,
                "Libellé" VARCHAR,
                "Débit (Ar)" DECIMAL(15,2),
                "Crédit (Ar)" DECIMAL(15,2),
                "Solde" DECIMAL(15,2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.balance (
                id SERIAL PRIMARY KEY,
                "Libellé " TEXT,
                "Numéro de compte" VARCHAR UNIQUE,
                "Débit (Ar)" DECIMAL(15,2),
                "Crédit (Ar)" DECIMAL(15,2),
                "Solde Débiteur" DECIMAL(15,2),
                "Solde Créditeur" DECIMAL(15,2),
                "Période" VARCHAR(50),
                derniere_maj TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.lettrage (
                id SERIAL PRIMARY KEY,
                "Numéro de compte" VARCHAR,
                "Date" DATE,
                "Référence" VARCHAR,
                "Libellé" VARCHAR,
                "Débit (Ar)" DECIMAL(15,2),
                "Crédit (Ar)" DECIMAL(15,2),
                "Solde partiel" DECIMAL(15,2),
                "Statut" VARCHAR,
                "Lettrage" VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.compte_resultat (
                id SERIAL PRIMARY KEY,
                "Charges (classe 6)" DECIMAL(15,2),
                "Produits (classe 7)" DECIMAL(15,2),
                "Résultat Net" DECIMAL(15,2),
                periode_debut DATE,
                periode_fin DATE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.bilan (
                id SERIAL PRIMARY KEY,
                "Poste" VARCHAR,
                "Actif (Ar)" DECIMAL(15,2),
                "Passif (Ar)" DECIMAL(15,2),
                date_generation DATE,
                file_hash VARCHAR(32),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS public.annexe (
                id SERIAL PRIMARY KEY,
                "Compte" VARCHAR,
                "Date" DATE,
                "Référence" VARCHAR,
                "Libellé" VARCHAR,
                "Débit (Ar)" DECIMAL(15,2),
                "Crédit (Ar)" DECIMAL(15,2),
                "Solde" DECIMAL(15,2),
                "categorie" VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]

        for command in sql_commands:
            cursor.execute(command)

        conn.commit()
        cursor.close()  # ✅ On ferme seulement le curseur
        print("✅ Tables créées avec succès dans Supabase")
        print("✅ Connexion maintenue ouverte pour l'analyse")
        
        return conn  # ✅ Retourne la connexion OUVERTE

    except Exception as error:
        print(f"❌ Erreur création tables : {error}")
        return None

# ✅ Créer les tables et garder la connexion ouverte
CONN_SUPABASE = creer_toutes_les_tables_supabase()

CSV_OUTPUT = "grand_journal.csv"
PDF_PCG = "./plan-comptable-general-2005.pdf"

# OCR et NLP
reader = easyocr.Reader(['fr', 'en'], gpu=False, model_storage_directory='.EasyOCR')
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

# Verification d'image si une pièce comptable ou non 
def verifier_image_piece(image_path):
    """
    Vérifie si l'image uploadée est bien une pièce comptable
    (facture, reçu, relevé, etc.).
    Si ce n’est pas le cas, affiche une alerte et stoppe le traitement.
    """
    # Extraire texte via OCR
    img = preprocess_image(image_path)
    result = reader.readtext(img)
    texte = " ".join([t for _, t, _ in result])

    # Si peu de texte, on considère déjà que ce n’est pas une pièce comptable
    if len(texte.strip()) < 30:
        st.error("⚠️ Cette image ne semble pas être une pièce comptable (texte illisible ou vide).")
        st.stop()

    prompt = f"""
Tu es un expert en comptabilité. 
Analyse le texte suivant et détermine s'il s'agit d'une pièce comptable (facture, reçu, bon de livraison, relevé, etc.)
Réponds uniquement par 'oui' ou 'non'.

Texte :
\"\"\"{texte}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        reponse = response.choices[0].message.content.strip().lower()

        if not reponse.startswith("oui"):
            st.error("Cette image ne semble pas être une pièce comptable.")
            st.stop()
        else:
            st.success("Pièce comptable reconnue.")
    except Exception as e:
        st.warning(f"Erreur lors de la vérification de la pièce : {e}")
        st.stop()
        
def traiter_image(image_path, pcg_contenu):
    verifier_image_piece(image_path)
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
# ==========================
# FONCTION AGRÉGATION AVEC CONNEXION RÉUTILISÉE
# ==========================


def aggreger_en_grand_journal(donnees, fichier_sortie="grand_journal.csv", conn_pg=None):
    """
    Agrège les données en Grand Journal, exporte en CSV et insère dans PostgreSQL (Supabase).
    Évite les doublons en vérifiant les pièces déjà insérées.
    """
    journaux = []
    for ligne in donnees:
        if isinstance(ligne.get("journal_markdown"), pd.DataFrame):
            df_piece = ligne["journal_markdown"].copy()
            df_piece["Type journal"] = ligne.get("type_journal", "inconnu")
            df_piece["Référence"] = ligne.get("numero_piece", "")
            df_piece["Fichier source"] = ligne.get("fichier", "")
            journaux.append(df_piece)

    if not journaux:
        st.warning("⚠️ Aucune donnée à traiter")
        return pd.DataFrame()

    # ✅ Agrégation
    grand_journal = pd.concat(journaux, ignore_index=True)
    colonnes = [
        "Date", "Référence", "Numéro de compte", "Libellé",
        "Débit (Ar)", "Crédit (Ar)", "Type journal", "Fichier source"
    ]
    grand_journal = grand_journal.reindex(columns=colonnes)

    # ✅ Nettoyage montants
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        grand_journal[col] = (
            grand_journal[col]
            .astype(str)
            .str.replace(r"[^\d.,-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        grand_journal[col] = pd.to_numeric(grand_journal[col], errors="coerce").fillna(0)

    # ✅ Dates
    grand_journal["Date"] = pd.to_datetime(grand_journal["Date"], errors="coerce", dayfirst=True).dt.date
    grand_journal = grand_journal.sort_values(by="Date").reset_index(drop=True)
    grand_journal.index = grand_journal.index + 1

    # ✅ Export CSV local
    grand_journal.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV exporté : {fichier_sortie}")

    # ✅ Insertion PostgreSQL Supabase (si connexion donnée)
    if conn_pg:
        try:
            cursor = conn_pg.cursor()

            # 🔍 Récupérer les pièces déjà insérées dans la BD
            cursor.execute("""
                SELECT DISTINCT "Référence", "Fichier source" 
                FROM grand_journal
            """)
            pieces_existantes = set(cursor.fetchall())
            
            st.info(f"📊 Pièces déjà en base : {len(pieces_existantes)}")

            # 🆕 Filtrer uniquement les NOUVELLES pièces
            nouvelles_pieces = []
            for _, row in grand_journal.iterrows():
                cle_piece = (str(row["Référence"]), str(row["Fichier source"]))
                if cle_piece not in pieces_existantes:
                    nouvelles_pieces.append((
                        row["Date"] if pd.notna(row["Date"]) else None,
                        str(row["Référence"]) if pd.notna(row["Référence"]) else "",
                        str(row["Numéro de compte"]) if pd.notna(row["Numéro de compte"]) else "",
                        str(row["Libellé"]) if pd.notna(row["Libellé"]) else "",
                        float(row["Débit (Ar)"]) if pd.notna(row["Débit (Ar)"]) else 0.0,
                        float(row["Crédit (Ar)"]) if pd.notna(row["Crédit (Ar)"]) else 0.0,
                        str(row["Type journal"]) if pd.notna(row["Type journal"]) else "inconnu",
                        str(row["Fichier source"]) if pd.notna(row["Fichier source"]) else ""
                    ))

            if not nouvelles_pieces:
                st.warning("⚠️ Aucune nouvelle pièce à insérer (toutes déjà en base)")
                cursor.close()
                return grand_journal

            # Compter avant insertion
            cursor.execute("SELECT COUNT(*) FROM grand_journal")
            count_avant = cursor.fetchone()[0]

            # ✅ Insertion groupée des NOUVELLES pièces uniquement
            execute_values(
                cursor,
                """
                INSERT INTO grand_journal 
                ("Date", "Référence", "Numéro de compte", "Libellé", 
                 "Débit (Ar)", "Crédit (Ar)", "Type journal", "Fichier source")
                VALUES %s
                """,
                nouvelles_pieces
            )

            conn_pg.commit()

            # Compter après insertion
            cursor.execute("SELECT COUNT(*) FROM grand_journal")
            count_apres = cursor.fetchone()[0]
            nb_nouvelles = count_apres - count_avant

            st.success(f"✅ {nb_nouvelles} nouvelles écritures insérées → Total en base : {count_apres}")
            st.info(f"📝 Écritures ignorées (déjà en base) : {len(grand_journal) - nb_nouvelles}")

            cursor.close()

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ Erreur PostgreSQL : {e}")
            import traceback
            st.error(traceback.format_exc())

    return grand_journal
def generer_grand_livre(df_grand_journal, fichier_sortie="grand_livre.csv", conn_pg=None):
    """
    Génère le Grand Livre à partir du Grand Journal, exporte en CSV et insère dans PostgreSQL (Supabase).
    Évite les doublons en vérifiant les entrées déjà insérées.
    """
    if df_grand_journal.empty:
        st.warning("⚠️ Grand Journal vide")
        return pd.DataFrame()

    # Nettoyer les colonnes Débit et Crédit
    df_grand_journal = df_grand_journal.copy()
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        df_grand_journal[col] = (
            df_grand_journal[col]
            .astype(str)
            .str.replace(r"[^\d.,-]", "", regex=True)
            .str.replace(",", ".", regex=False)
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
    colonnes = ["Compte", "Date", "Référence", "Libellé", "Débit (Ar)", "Crédit (Ar)", "Solde"]
    df_grand_livre = df_grand_livre.reindex(columns=colonnes)

    # ✅ Export CSV (avec nombres bruts pour analyse)
    df_export = df_grand_livre.copy()
    df_export.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV Grand Livre exporté : {fichier_sortie}")

    # ✅ Insertion PostgreSQL Supabase (si connexion donnée)
    if conn_pg:
        cursor = None
        try:
            cursor = conn_pg.cursor()

            # 🔍 Récupérer les entrées déjà insérées dans la BD
            cursor.execute("""
                SELECT DISTINCT "Compte", "Date", "Référence" 
                FROM grand_livre
            """)
            entrees_existantes = set(cursor.fetchall())
            
            st.info(f"📊 Entrées déjà en base (Grand Livre) : {len(entrees_existantes)}")

            # 🆕 Filtrer uniquement les NOUVELLES entrées
            nouvelles_entrees = []
            for _, row in df_grand_livre.iterrows():
                cle_entree = (
                    str(row["Compte"]), 
                    row["Date"] if pd.notna(row["Date"]) else None,
                    str(row["Référence"])
                )
                if cle_entree not in entrees_existantes:
                    nouvelles_entrees.append((
                        str(row["Compte"]) if pd.notna(row["Compte"]) else "",
                        row["Date"] if pd.notna(row["Date"]) else None,
                        str(row["Référence"]) if pd.notna(row["Référence"]) else "",
                        str(row["Libellé"]) if pd.notna(row["Libellé"]) else "",
                        float(row["Débit (Ar)"]) if pd.notna(row["Débit (Ar)"]) else 0.0,
                        float(row["Crédit (Ar)"]) if pd.notna(row["Crédit (Ar)"]) else 0.0,
                        float(row["Solde"]) if pd.notna(row["Solde"]) else 0.0
                    ))

            if not nouvelles_entrees:
                st.warning("⚠️ Aucune nouvelle entrée à insérer dans le Grand Livre (toutes déjà en base)")
            else:
                # Compter avant insertion
                cursor.execute("SELECT COUNT(*) FROM grand_livre")
                count_avant = cursor.fetchone()[0]

                # ✅ Insertion groupée des NOUVELLES entrées uniquement
                execute_values(
                    cursor,
                    """
                    INSERT INTO grand_livre 
                    ("Compte", "Date", "Référence", "Libellé", 
                     "Débit (Ar)", "Crédit (Ar)", "Solde")
                    VALUES %s
                    """,
                    nouvelles_entrees
                )

                conn_pg.commit()

                # Compter après insertion
                cursor.execute("SELECT COUNT(*) FROM grand_livre")
                count_apres = cursor.fetchone()[0]
                nb_nouvelles = count_apres - count_avant

                st.success(f"✅ {nb_nouvelles} nouvelles entrées insérées dans le Grand Livre → Total en base : {count_apres}")
                st.info(f"📝 Entrées ignorées (déjà en base) : {len(df_grand_livre) - nb_nouvelles}")

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ Erreur PostgreSQL Grand Livre : {e}")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            if cursor:
                cursor.close()

    # ✅ Version affichage formatée (ex : "200 000 Ar")
    df_grand_livre_affichage = df_grand_livre.copy()
    for col in ["Débit (Ar)", "Crédit (Ar)", "Solde"]:
        df_grand_livre_affichage[col] = df_grand_livre_affichage[col].apply(
            lambda x: f"{int(x):,} Ar".replace(",", " ") if x != 0 else ""
        )

    return df_grand_livre_affichage

def generer_lettrage(df_grand_journal, fichier_sortie="lettrage.csv", conn_pg=None):
    """
    Rapproche débits/crédits par compte (lettrage).
    Exporte en CSV et insère dans PostgreSQL (Supabase).
    - supporte lettrage exact et partiel (fractionnement logique).
    - retourne un dataframe avec colonnes: Numéro de compte, Date, Référence, Libellé,
      Débit (Ar), Crédit (Ar), Solde partiel, Statut, Lettrage
    """
    if df_grand_journal is None or df_grand_journal.empty:
        st.warning("⚠️ Grand Journal vide")
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
        df_c["Lettrage"] = ""
        
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
    
    # ✅ Export CSV
    df_out.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV Lettrage exporté : {fichier_sortie}")

    # ✅ Insertion PostgreSQL Supabase (si connexion donnée)
    if conn_pg:
        cursor = None
        try:
            cursor = conn_pg.cursor()

            # 🔍 Récupérer les entrées déjà insérées dans la BD
            cursor.execute("""
                SELECT DISTINCT "Numéro de compte", "Date", "Référence", "Lettrage"
                FROM lettrage
            """)
            entrees_existantes = set(cursor.fetchall())
            
            st.info(f"📊 Entrées déjà en base (Lettrage) : {len(entrees_existantes)}")

            # 🆕 Filtrer uniquement les NOUVELLES entrées
            nouvelles_entrees = []
            for _, row in df_out.iterrows():
                cle_entree = (
                    str(row["Numéro de compte"]),
                    row["Date"] if pd.notna(row["Date"]) else None,
                    str(row["Référence"]),
                    str(row["Lettrage"])
                )
                if cle_entree not in entrees_existantes:
                    nouvelles_entrees.append((
                        str(row["Numéro de compte"]) if pd.notna(row["Numéro de compte"]) else "",
                        row["Date"] if pd.notna(row["Date"]) else None,
                        str(row["Référence"]) if pd.notna(row["Référence"]) else "",
                        str(row["Libellé"]) if pd.notna(row["Libellé"]) else "",
                        float(row["Débit (Ar)"]) if pd.notna(row["Débit (Ar)"]) else 0.0,
                        float(row["Crédit (Ar)"]) if pd.notna(row["Crédit (Ar)"]) else 0.0,
                        float(row["Solde partiel"]) if pd.notna(row["Solde partiel"]) else 0.0,
                        str(row["Statut"]) if pd.notna(row["Statut"]) else "",
                        str(row["Lettrage"]) if pd.notna(row["Lettrage"]) else ""
                    ))

            if not nouvelles_entrees:
                st.warning("⚠️ Aucune nouvelle entrée à insérer dans le Lettrage (toutes déjà en base)")
            else:
                # Compter avant insertion
                cursor.execute("SELECT COUNT(*) FROM lettrage")
                count_avant = cursor.fetchone()[0]

                # ✅ Insertion groupée des NOUVELLES entrées uniquement
                execute_values(
                    cursor,
                    """
                    INSERT INTO lettrage 
                    ("Numéro de compte", "Date", "Référence", "Libellé", 
                     "Débit (Ar)", "Crédit (Ar)", "Solde partiel", "Statut", "Lettrage")
                    VALUES %s
                    """,
                    nouvelles_entrees
                )

                conn_pg.commit()

                # Compter après insertion
                cursor.execute("SELECT COUNT(*) FROM lettrage")
                count_apres = cursor.fetchone()[0]
                nb_nouvelles = count_apres - count_avant

                st.success(f"✅ {nb_nouvelles} nouvelles entrées insérées dans le Lettrage → Total en base : {count_apres}")
                st.info(f"📝 Entrées ignorées (déjà en base) : {len(df_out) - nb_nouvelles}")

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ Erreur PostgreSQL Lettrage : {e}")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            if cursor:
                cursor.close()

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
# a partir de ici , y a encore des code a rectifier
def generer_balance(df_grand_journal, fichier_sortie="balance.csv", conn_pg=None):
    """
    Génère la balance comptable à partir du Grand Journal.
    Exporte en CSV et insère/met à jour dans PostgreSQL (Supabase).
    Inclut le libellé de la dernière écriture pour chaque compte.
    """

    if df_grand_journal.empty:
        st.warning("⚠️ Grand Journal vide")
        return pd.DataFrame()

    df = df_grand_journal.copy()

    # Vérifier colonnes obligatoires
    colonnes_attendues = ["Numéro de compte", "Débit (Ar)", "Crédit (Ar)"]
    for col in colonnes_attendues:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    # Nettoyage colonnes numériques
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[^\d\-,.]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ✅ Grouper par compte avec agrégation pour récupérer le libellé
    df_balance = df.groupby("Numéro de compte", as_index=False).agg({
        "Débit (Ar)": "sum",
        "Crédit (Ar)": "sum",
        "Libellé": "last"  # ✅ Prend le dernier libellé pour ce compte
    })

    # Calcul solde et séparation Débiteur/Créditeur
    df_balance["Solde"] = df_balance["Débit (Ar)"] - df_balance["Crédit (Ar)"]
    df_balance["Solde Débiteur"] = df_balance["Solde"].apply(lambda x: x if x > 0 else 0)
    df_balance["Solde Créditeur"] = df_balance["Solde"].apply(lambda x: -x if x < 0 else 0)
    
    # ✅ Ajouter période (jour/mois/année actuel)
    periode_actuelle = pd.Timestamp.now().strftime("%Y-%m-%d")  # Format: 2025-01
    df_balance["Période"] = periode_actuelle

    # ✅ Réorganiser colonnes
    df_balance_numeric = df_balance[[
        "Numéro de compte", 
        "Libellé",
        "Débit (Ar)", 
        "Crédit (Ar)", 
        "Solde Débiteur", 
        "Solde Créditeur",
        "Période"
    ]].copy()

    # ✅ Export CSV (valeurs numériques brutes)
    df_balance_numeric.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV Balance exporté : {fichier_sortie}")

    # Calcul total pour vérification
    total_debit = df_balance_numeric["Débit (Ar)"].sum()
    total_credit = df_balance_numeric["Crédit (Ar)"].sum()
    
    if abs(total_debit - total_credit) < 0.01:
        st.success(f"✅ Balance équilibrée : {total_debit:,.2f} Ar")
    else:
        st.warning(f"⚠️ Balance déséquilibrée : Débit={total_debit:,.2f} Ar, Crédit={total_credit:,.2f} Ar")

    # ✅ Insertion/Mise à jour PostgreSQL Supabase
    if conn_pg:
        cursor = None
        try:
            cursor = conn_pg.cursor()

            st.info(f"📊 Synchronisation avec Supabase...")

            # 🔄 UPSERT pour chaque compte
            for _, row in df_balance_numeric.iterrows():
                compte = str(row["Numéro de compte"])
                libelle = str(row["Libellé"]) if pd.notna(row["Libellé"]) else ""
                debit = float(row["Débit (Ar)"])
                credit = float(row["Crédit (Ar)"])
                solde_deb = float(row["Solde Débiteur"])
                solde_cred = float(row["Solde Créditeur"])
                periode = str(row["Période"])

                cursor.execute("""
                    INSERT INTO balance 
                    ("Numéro de compte", "Libellé ", "Débit (Ar)", "Crédit (Ar)", 
                     "Solde Débiteur", "Solde Créditeur", "Période", derniere_maj)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT ("Numéro de compte") 
                    DO UPDATE SET
                        "Libellé " = EXCLUDED."Libellé ",
                        "Débit (Ar)" = EXCLUDED."Débit (Ar)",
                        "Crédit (Ar)" = EXCLUDED."Crédit (Ar)",
                        "Solde Débiteur" = EXCLUDED."Solde Débiteur",
                        "Solde Créditeur" = EXCLUDED."Solde Créditeur",
                        "Période" = EXCLUDED."Période",
                        derniere_maj = CURRENT_TIMESTAMP
                """, (compte, libelle, debit, credit, solde_deb, solde_cred, periode))

            conn_pg.commit()

            # Compter total en base
            cursor.execute("SELECT COUNT(*) FROM balance")
            total = cursor.fetchone()[0]

            st.success(f"✅ Balance synchronisée avec Supabase → {total} comptes en base")

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ Erreur PostgreSQL Balance : {e}")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            if cursor:
                cursor.close()

    # ✅ Formatage pour affichage (après insertion Supabase)
    df_balance_affichage = df_balance_numeric.copy()
    
    for col in ["Débit (Ar)", "Crédit (Ar)", "Solde Débiteur", "Solde Créditeur"]:
        df_balance_affichage[col] = df_balance_affichage[col].apply(
            lambda x: f"{int(x):,} Ar".replace(",", " ") if pd.notna(x) and x != 0 else ""
        )

    return df_balance_affichage
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
def generer_bilan(df_balance_brut, fichier_sortie="bilan.csv", conn_pg=None):
    """
    Génère un bilan simplifié (Actif / Passif) à partir de la balance fournie.
    Exporte en CSV et insère dans PostgreSQL (Supabase) avec détection de doublons.
    
    ✅ Utilise un hash MD5 du fichier pour éviter les réinsertions du même fichier
    
    Parameters:
    -----------
    df_balance_brut : pd.DataFrame
        DataFrame contenant la balance avec colonnes : 
        "Numéro de compte", "Débit (Ar)", "Crédit (Ar)"
    fichier_sortie : str
        Chemin du fichier CSV de sortie
    conn_pg : psycopg2.connection
        Connexion PostgreSQL/Supabase (optionnel)
    
    Returns:
    --------
    tuple : (df_bilan_affichage, total_actif, total_passif)
    """
    
    # ═══════════════════════════════════════════════════════════
    # 1. VALIDATION DES DONNÉES
    # ═══════════════════════════════════════════════════════════
    if df_balance_brut is None or df_balance_brut.empty:
        st.warning("⚠️ Balance vide, impossible de générer le bilan")
        return pd.DataFrame(), 0, 0

    df = df_balance_brut.copy()

    # ═══════════════════════════════════════════════════════════
    # 2. CALCUL DU HASH UNIQUE DU FICHIER
    # ═══════════════════════════════════════════════════════════
    file_hash = None
    try:
        # Colonnes essentielles pour identifier le fichier de manière unique
        colonnes_hash = ["Numéro de compte", "Débit (Ar)", "Crédit (Ar)"]
        
        # Vérifier que les colonnes existent
        colonnes_disponibles = [col for col in colonnes_hash if col in df.columns]
        
        if not colonnes_disponibles:
            st.error("❌ Colonnes nécessaires introuvables pour calculer le hash")
        else:
            # Sérialiser les données en JSON puis calculer MD5
            data_to_hash = df[colonnes_disponibles].to_json(
                orient='records', 
                force_ascii=False,
                default_handler=str  # Gère les types non-JSON
            )
            file_hash = hashlib.md5(data_to_hash.encode('utf-8')).hexdigest()
            st.info(f"🔐 Empreinte du fichier : `{file_hash[:12]}...`")
    
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul du hash : {e}")
        st.warning("⚠️ Le système continuera sans détection de doublons")

    # ═══════════════════════════════════════════════════════════
    # 3. PRÉPARATION DES COLONNES
    # ═══════════════════════════════════════════════════════════
    for col in ["Débit (Ar)", "Crédit (Ar)"]:
        if col not in df.columns:
            df[col] = ""
            st.warning(f"⚠️ Colonne '{col}' manquante, initialisée à vide")

    # ═══════════════════════════════════════════════════════════
    # 4. CONVERSION DES MONTANTS EN NUMÉRIQUE
    # ═══════════════════════════════════════════════════════════
    df["_debit_num"] = df["Débit (Ar)"].apply(parse_montant)
    df["_credit_num"] = df["Crédit (Ar)"].apply(parse_montant)
    df["_net"] = df["_debit_num"] - df["_credit_num"]

    # ═══════════════════════════════════════════════════════════
    # 5. CLASSIFICATION DES COMPTES PAR POSTE
    # ═══════════════════════════════════════════════════════════
    def classifier_compte(numero):
        """
        Classifie un numéro de compte selon le plan comptable OHADA/PCG
        """
        n = str(numero).strip()
        if not n:
            return "Non classé"
        
        # ACTIF
        if n.startswith("2"):
            return "Actif immobilisé"
        if n.startswith("3"):
            return "Stocks"
        if n.startswith("41") or n.startswith("46"):
            return "Créances clients et divers"
        if n.startswith("5") or n.startswith("53") or n.startswith("512"):
            return "Disponibilités (Banque / Caisse)"
        
        # PASSIF
        if n.startswith("1"):
            return "Capitaux propres et assimilés"
        if n.startswith("16") or n.startswith("17") or n.startswith("19"):
            return "Emprunts et dettes financières"
        if n.startswith("40") or n.startswith("42") or n.startswith("44"):
            return "Dettes fournisseurs / Tiers"
        
        return "Autres"

    df["Poste"] = df["Numéro de compte"].apply(classifier_compte)

    # ═══════════════════════════════════════════════════════════
    # 6. AGRÉGATION PAR POSTE (ACTIF / PASSIF)
    # ═══════════════════════════════════════════════════════════
    postes = {}
    for _, row in df.iterrows():
        poste = row["Poste"]
        net = row["_net"]
        
        if poste not in postes:
            postes[poste] = {"actif": 0.0, "passif": 0.0}
        
        # Solde positif → Actif
        if net > 0:
            postes[poste]["actif"] += net
        # Solde négatif → Passif
        elif net < 0:
            postes[poste]["passif"] += abs(net)

    # ═══════════════════════════════════════════════════════════
    # 7. CONSTRUCTION DU DATAFRAME BILAN
    # ═══════════════════════════════════════════════════════════
    lignes = []
    for poste, vals in postes.items():
        lignes.append({
            "Poste": poste,
            "Actif (Ar)": vals["actif"],
            "Passif (Ar)": vals["passif"]
        })
    
    df_bilan = pd.DataFrame(lignes).sort_values(by="Poste").reset_index(drop=True)

    # Calcul des totaux
    total_actif = df_bilan["Actif (Ar)"].sum()
    total_passif = df_bilan["Passif (Ar)"].sum()

    # Ligne de total
    totals_row = pd.DataFrame([{
        "Poste": "TOTAL",
        "Actif (Ar)": total_actif,
        "Passif (Ar)": total_passif
    }])
    
    df_bilan_numeric = pd.concat([df_bilan, totals_row], ignore_index=True)

    # ═══════════════════════════════════════════════════════════
    # 8. EXPORT CSV
    # ═══════════════════════════════════════════════════════════
    try:
        df_bilan_numeric.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
        st.success(f"✅ Fichier CSV exporté : `{fichier_sortie}`")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'export CSV : {e}")

    # ═══════════════════════════════════════════════════════════
    # 9. INSERTION DANS SUPABASE (AVEC DÉTECTION DE DOUBLONS)
    # ═══════════════════════════════════════════════════════════
    if conn_pg and file_hash:
        cursor = None
        try:
            cursor = conn_pg.cursor()

            # ───────────────────────────────────────────────────
            # 9.1. VÉRIFIER SI CE FICHIER A DÉJÀ ÉTÉ TRAITÉ
            # ───────────────────────────────────────────────────
            cursor.execute("""
                SELECT COUNT(*), MAX(date_generation) 
                FROM bilan 
                WHERE file_hash = %s
            """, (file_hash,))
            
            result = cursor.fetchone()
            count_existing = result[0] if result else 0
            last_date = result[1] if result and result[1] else None

            # ───────────────────────────────────────────────────
            # 9.2. FICHIER DÉJÀ TRAITÉ → BLOQUER L'INSERTION
            # ───────────────────────────────────────────────────
            if count_existing > 0:
                st.warning(f"⚠️ **Ce fichier a déjà été analysé**")
                st.info(f"📅 Première analyse : {last_date}")
                st.info(f"📊 {count_existing} enregistrements existants pour ce fichier")
                st.info("✅ Aucune insertion effectuée (doublons évités)")
                
                # Afficher un aperçu des données existantes
                cursor.execute("""
                    SELECT "Poste", "Actif (Ar)", "Passif (Ar)"
                    FROM bilan 
                    WHERE file_hash = %s
                    ORDER BY "Poste"
                """, (file_hash,))
                
                existing_data = cursor.fetchall()
                st.write("**Données existantes :**")
                df_existing = pd.DataFrame(
                    existing_data, 
                    columns=["Poste", "Actif (Ar)", "Passif (Ar)"]
                )
                st.dataframe(df_existing)

            # ───────────────────────────────────────────────────
            # 9.3. NOUVEAU FICHIER → INSERTION
            # ───────────────────────────────────────────────────
            else:
                st.info(f"📊 Nouveau fichier détecté. Insertion dans Supabase...")

                nouvelles_entrees = []
                date_aujourd_hui = pd.Timestamp.now().date()
                
                for _, row in df_bilan_numeric.iterrows():
                    nouvelles_entrees.append((
                        str(row["Poste"]),
                        float(row["Actif (Ar)"]) if row["Actif (Ar)"] != 0 else 0.0,
                        float(row["Passif (Ar)"]) if row["Passif (Ar)"] != 0 else 0.0,
                        date_aujourd_hui,  # date_generation (DATE)
                        file_hash  # Hash unique du fichier
                    ))

                execute_values(
                    cursor,
                    """
                    INSERT INTO bilan 
                    ("Poste", "Actif (Ar)", "Passif (Ar)", date_generation, file_hash)
                    VALUES %s
                    """,
                    nouvelles_entrees
                )

                conn_pg.commit()

                # ───────────────────────────────────────────────
                # 9.4. STATISTIQUES POST-INSERTION
                # ───────────────────────────────────────────────
                cursor.execute("SELECT COUNT(*) FROM bilan")
                total_enregistrements = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT file_hash) FROM bilan WHERE file_hash IS NOT NULL")
                nb_fichiers_uniques = cursor.fetchone()[0]

                st.success(f"✅ **Bilan inséré avec succès dans Supabase**")
                st.success(f"📊 Total : **{total_enregistrements}** enregistrements | **{nb_fichiers_uniques}** fichiers uniques")

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ **Erreur PostgreSQL lors de l'insertion du bilan :**")
            st.error(f"```{str(e)}```")
            import traceback
            st.error(f"```{traceback.format_exc()}```")
        
        finally:
            if cursor:
                cursor.close()

    elif conn_pg and not file_hash:
        st.warning("⚠️ Impossible de calculer le hash du fichier.")
        st.warning("⚠️ Insertion annulée pour éviter les doublons potentiels.")
        st.info("💡 Vérifiez que votre fichier contient les colonnes nécessaires.")

    # ═══════════════════════════════════════════════════════════
    # 10. MISE EN FORME POUR AFFICHAGE
    # ═══════════════════════════════════════════════════════════
    df_bilan_aff = df_bilan_numeric.copy()
    
    # Formater les montants avec séparateurs de milliers
    df_bilan_aff["Actif (Ar)"] = df_bilan_aff["Actif (Ar)"].apply(
        lambda x: f"{int(round(x)):,}".replace(",", " ") + " Ar" if x != 0 else ""
    )
    df_bilan_aff["Passif (Ar)"] = df_bilan_aff["Passif (Ar)"].apply(
        lambda x: f"{int(round(x)):,}".replace(",", " ") + " Ar" if x != 0 else ""
    )

    # ═══════════════════════════════════════════════════════════
    # 11. RETOUR DES RÉSULTATS
    # ═══════════════════════════════════════════════════════════
    return df_bilan_aff, total_actif, total_passif


# ═══════════════════════════════════════════════════════════════
# FONCTION BONUS : AFFICHER L'HISTORIQUE DES FICHIERS
# ═══════════════════════════════════════════════════════════════
def afficher_historique_bilans(conn_pg):
    """
    Affiche l'historique des fichiers uniques analysés avec leurs statistiques.
    
    Parameters:
    -----------
    conn_pg : psycopg2.connection
        Connexion PostgreSQL/Supabase
    """
    if not conn_pg:
        st.warning("⚠️ Pas de connexion à la base de données")
        return
    
    cursor = None
    try:
        cursor = conn_pg.cursor()
        
        # Récupérer les statistiques par fichier
        cursor.execute("""
            SELECT 
                file_hash,
                COUNT(*) as nb_lignes,
                MIN(date_generation) as premiere_analyse,
                MAX(date_generation) as derniere_analyse,
                SUM("Actif (Ar)") as total_actif,
                SUM("Passif (Ar)") as total_passif
            FROM bilan
            WHERE file_hash IS NOT NULL
            GROUP BY file_hash
            ORDER BY derniere_analyse DESC
        """)
        
        resultats = cursor.fetchall()
        
        if not resultats:
            st.info("ℹ️ Aucun fichier analysé pour le moment")
            return
        
        st.subheader(f"📂 Historique des fichiers analysés ({len(resultats)} fichiers)")
        
        for idx, (hash_val, nb, premiere, derniere, actif, passif) in enumerate(resultats, 1):
            with st.expander(f"📄 Fichier #{idx} - {hash_val[:12]}... (analysé le {premiere})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nombre de lignes", nb)
                with col2:
                    st.metric("Total Actif", f"{int(actif):,}".replace(",", " ") + " Ar")
                with col3:
                    st.metric("Total Passif", f"{int(passif):,}".replace(",", " ") + " Ar")
                
                # Afficher le détail
                cursor.execute("""
                    SELECT "Poste", "Actif (Ar)", "Passif (Ar)"
                    FROM bilan
                    WHERE file_hash = %s
                    ORDER BY "Poste"
                """, (hash_val,))
                
                detail = cursor.fetchall()
                df_detail = pd.DataFrame(detail, columns=["Poste", "Actif (Ar)", "Passif (Ar)"])
                st.dataframe(df_detail, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération de l'historique : {e}")
    
    finally:
        if cursor:
            cursor.close()
def generer_compte_resultat(df_balance, fichier_sortie="compte_resultat.csv", conn_pg=None):
    """
    Génère un compte de résultat simplifié (Charges / Produits).
    Exporte en CSV et insère dans PostgreSQL (Supabase) SANS supprimer l'ancien.
    """
    if df_balance is None or df_balance.empty:
        st.warning("⚠️ Balance vide")
        return pd.DataFrame(), 0, 0, 0

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

    df_cr_numeric = pd.DataFrame({
        "Charges (classe 6)": [total_charges],
        "Produits (classe 7)": [total_produits],
        "Résultat Net": [resultat_net]
    })

    # ✅ Export CSV
    df_cr_numeric.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV Compte de Résultat exporté : {fichier_sortie}")

    # ✅ Insertion PostgreSQL Supabase SANS SUPPRESSION
    if conn_pg:
        cursor = None
        try:
            cursor = conn_pg.cursor()

            st.info(f"📊 Insertion du compte de résultat dans Supabase (historique conservé)...")

            # ✅ INSERTION DIRECTE sans DELETE
            cursor.execute("""
                INSERT INTO compte_resultat 
                ("Charges (classe 6)", "Produits (classe 7)", "Résultat Net", 
                 periode_debut, periode_fin)
                VALUES (%s, %s, %s, CURRENT_DATE, CURRENT_DATE)
            """, (float(total_charges), float(total_produits), float(resultat_net)))

            conn_pg.commit()

            # Compter total en base
            cursor.execute("SELECT COUNT(*) FROM compte_resultat")
            total = cursor.fetchone()[0]

            st.success(f"✅ Compte de Résultat inséré → Total analyses historisées : {total}")

        except Exception as e:
            conn_pg.rollback()
            st.error(f"❌ Erreur PostgreSQL Compte de Résultat : {e}")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            if cursor:
                cursor.close()

    # Préparer tableau résultat formaté
    data = {
        "Charges (classe 6)": [f"{int(round(total_charges)):,}".replace(",", " ") + " Ar"],
        "Produits (classe 7)": [f"{int(round(total_produits)):,}".replace(",", " ") + " Ar"],
        "Résultat Net": [f"{int(round(resultat_net)):,}".replace(",", " ") + " Ar"]
    }

    df_cr = pd.DataFrame(data)

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
    # with pd.ExcelWriter(fichier_sortie) as writer:
    #     immobilisations_detail.to_excel(writer, sheet_name="Immobilisations", index=False)
    #     clients.to_excel(writer, sheet_name="Clients", index=False)
    #     fournisseurs.to_excel(writer, sheet_name="Fournisseurs", index=False)
    #     provisions.to_excel(writer, sheet_name="Provisions", index=False)
    # Concaténer tout dans un seul DataFrame
    df_annexe = pd.concat([immobilisations, clients, fournisseurs, provisions], ignore_index=True)

    # Sauvegarde CSV
    df_annexe.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")
    
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
        st.session_state.df_final = aggreger_en_grand_journal(donnees, CSV_OUTPUT,
            conn_pg=CONN_SUPABASE )
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
        df_grand_livre = generer_grand_livre(st.session_state.df_final, "grand_livre.csv",
            conn_pg=CONN_SUPABASE)
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
        df_lettrage = generer_lettrage(st.session_state.df_final, "lettrage.csv",conn_pg=CONN_SUPABASE )

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
    df_balance_brut = generer_balance(st.session_state.df_final, "balance.csv",
    conn_pg=CONN_SUPABASE )

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
    df_bilan, tot_actif, tot_passif = generer_bilan(df_balance_brut, "bilan.csv",conn_pg=CONN_SUPABASE)

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

    df_cr, total_charges, total_produits, resultat_net = generer_compte_resultat(df_balance_brut, "compte_resultat.csv",
    conn_pg=CONN_SUPABASE )

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