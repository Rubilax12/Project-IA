import os
import re
import discord
from discord.ext import commands
import openai
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Installer les dépendances nécessaires
nltk.download("punkt")
nltk.download("wordnet")

# Charger les bibliothèques NLP
nlp = spacy.load("fr_core_news_sm")
lemmatizer = WordNetLemmatizer()

# Dossier contenant les fichiers TXT
TXT_FOLDER = "toacrd/TXT"

# Clé API OpenAI
openai.api_key = os.getenv("OPENAI_TOKEN")

# Initialisation du bot Discord
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Fonction pour extraire les mots-clés
def extraire_mots_cles(question):
    """
    Extrait les mots-clés d'une question de l'utilisateur.
    """
    doc = nlp(question)
    mots_cles = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return mots_cles

# Fonction pour trouver des synonymes
def trouver_synonymes(mot):
    """
    Trouve des synonymes d'un mot via WordNet.
    """
    synonymes = set()
    for syn in wordnet.synsets(mot, lang="fra"):  # Utilise la langue française
        for lemma in syn.lemmas("fra"):
            synonymes.add(lemma.name())
    return list(synonymes)

# Fonction pour chercher dans les fichiers TXT
def rechercher_fenetres_texte(mots_cles, dossier_txt, fenetre=200):
    """
    Recherche les mots-clés ou leurs synonymes dans les fichiers TXT.
    """
    fenetres = []
    for filename in os.listdir(dossier_txt):
        if filename.endswith(".txt"):
            filepath = os.path.join(dossier_txt, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                contenu = f.read()

                # Recherche des mots-clés et synonymes
                for mot in mots_cles:
                    mots_recherche = set([mot] + trouver_synonymes(mot))
                    pattern = r'\b(' + '|'.join(re.escape(m) for m in mots_recherche) + r')\b'

                    for match in re.finditer(pattern, contenu, re.IGNORECASE):
                        start = max(0, match.start() - fenetre)
                        end = min(len(contenu), match.end() + fenetre)
                        fenetres.append(contenu[start:end])
    return fenetres

# Fonction pour générer une réponse via OpenAI
def generer_reponse(question, fenetres, model="gpt-4"):
    """
    Génère une réponse basée sur la question et le contexte extrait.
    """
    contexte = "\n---\n".join(fenetres)
    prompt = (
        f"Voici une question posée par un utilisateur : {question}\n"
        f"Voici des extraits de ma base de données pour répondre à cette question :\n{contexte}\n"
        f"Fournis une réponse claire et concise en utilisant les informations ci-dessus."
    )
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui fournit des réponses basées sur des informations contextuelles."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"], response.model
    except Exception as e:
        return f"Erreur : {str(e)}", None

# Commande pour interagir avec le bot
@bot.command(name="ask")
async def ask_question(ctx, *, question):
    """
    Répond à une question posée par l'utilisateur en utilisant OpenAI et les fichiers TXT.
    """
    await ctx.send("Analyse de la question...")
    
    # Extraction des mots-clés
    mots_cles = extraire_mots_cles(question)
    await ctx.send(f"Mots-clés extraits : {mots_cles}")
    
    # Recherche des fenêtres contextuelles
    fenetres = rechercher_fenetres_texte(mots_cles, TXT_FOLDER)
    await ctx.send(f"{len(fenetres)} fenêtres trouvées dans la base de données.")
    
    # Génération de la réponse
    if fenetres:
        reponse, model_utilise = generer_reponse(question, fenetres)
        await ctx.send(f"Modèle utilisé : {model_utilise}")
        await ctx.send(f"Réponse : {reponse}")
    else:
        await ctx.send("Aucune information pertinente trouvée dans la base de données.")

# Commande pour tester les fonctionnalités
@bot.command(name="test")
async def test_functionalities(ctx):
    """
    Teste indépendamment les fonctionnalités d'extraction et de recherche.
    """
    question = "Comment fonctionne l'apprentissage supervisé ?"
    mots_cles = extraire_mots_cles(question)
    fenetres = rechercher_fenetres_texte(mots_cles, TXT_FOLDER)
    await ctx.send(f"Mots-clés extraits : {mots_cles}")
    await ctx.send(f"{len(fenetres)} fenêtres trouvées :")
    for fenetre in fenetres[:3]:  # Limiter à 3 fenêtres pour éviter les longs messages
        await ctx.send(f"Fenêtre : {fenetre}")

# Lancer le bot
@bot.event
async def on_ready():
    print(f"{bot.user} est connecté et prêt à répondre !")

# Ajouter une commande pour quitter proprement
@bot.command(name="quit")
@commands.is_owner()
async def quit_bot(ctx):
    """
    Arrête le bot proprement.
    """
    await ctx.send("Arrêt du bot...")
    await bot.close()

# Démarrage du bot avec le token Discord
if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    if not DISCORD_TOKEN:
        print("Erreur : DISCORD_TOKEN non trouvé. Assurez-vous de définir DISCORD_TOKEN dans les variables d'environnement.")
    else:
        bot.run(DISCORD_TOKEN)
