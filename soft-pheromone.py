import discord
import os
import openai
from dotenv import load_dotenv
import platform
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import re

# Fonction pour effacer le terminal en fonction de l'OS
def clear_terminal():
    system_name = platform.system()
    if system_name == "Windows":
        os.system('cls')  # Pour Windows
    else:
        os.system('clear')  # Pour Linux et macOS

# Appel de la fonction pour effacer l'écran au démarrage
clear_terminal()

# Charger les variables d'environnement
load_dotenv()

# Configuration des clés API
openai.api_key = os.getenv("OPENAI_TOKEN")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not openai.api_key or not DISCORD_TOKEN:
    raise ValueError("Les clés API OpenAI ou Discord sont manquantes dans les variables d'environnement.")

# Afficher la version d'OpenAI utilisée
print(f"Version d'OpenAI utilisée : {openai.__version__}")

# Configurer les intents Discord
intents = discord.Intents.default()
intents.message_content = True

# Initialisation du client Discord
client = discord.Client(intents=intents)

# Variable pour s'assurer que le modèle n'est affiché qu'une seule fois
model_displayed = False

# Gestion de l'historique des conversations
user_histories = {}
max_chat_history = 20
role_message = {"role": "system", "content": "Tu es mon assistant. Sois concis dans tes réponses."}

# Charger les bibliothèques NLP
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")  # Télécharger la ressource omw-1.4
nlp = spacy.load("fr_core_news_sm")
lemmatizer = WordNetLemmatizer()

# Dossier contenant les fichiers TXT
TXT_FOLDER = "toacrd/TXT"

# Liste de mots à ignorer
mots_a_ignorer = [
    # Déterminants
    'le', 'la', 'les', 'un', 'une', 'des', 'ce', 'cet', 'cette', 'ces', 'celui', 'celle', 'ceux', 'celles',
    # Conjonctions
    'et', 'ou', 'mais', 'donc', 'car', 'or', 'ni', 'ainsi', 'parce que', 'puisque', 'quand', 'lorsque', 'tandis que',
    # Prépositions
    'à', 'en', 'dans', 'sur', 'sous', 'par', 'pour', 'contre', 'de', 'du', 'des', 'avec', 'sans', 'entre', 'vers', 'chez',
    # Pronoms
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'moi', 'toi', 'lui', 'elle', 'soi', 'y', 'en', 'leur', 'leurs',
    # Mots grossiers (exemples)
    'merde', 'putain', 'con', 'connard', 'salope', 'bordel', 'chier', 'foutre', 'bite', 'cul', 'enculé',
    # Autres mots peu importants
    'a', 'an', 'the', 'of', 'in', 'to', 'and', 'or', 'but', 'so', 'if', 'then', 'because', 'when', 'while', 'although', 'though',
    'this', 'that', 'these', 'those', 'it', 'its', 'their', 'theirs', 'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'themselves',
    'me', 'you', 'your', 'yours', 'yourself', 'yourselves', 'myself', 'ourselves', 'ours', 'we', 'us', 'ourselves', 'ours', 'i', 'my', 'mine',
    'for', 'with', 'without', 'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
    'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
]

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
    if mot in mots_a_ignorer:
        return []
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
    file_usage_count = {}
    for filename in os.listdir(dossier_txt):
        if filename.endswith(".txt"):
            filepath = os.path.join(dossier_txt, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    contenu = f.read()

                    # Recherche des mots-clés et synonymes
                    for mot in mots_cles:
                        mots_recherche = set([mot] + trouver_synonymes(mot))
                        pattern = r'\b(' + '|'.join(re.escape(m) for m in mots_recherche) + r')\b'

                        for match in re.finditer(pattern, contenu, re.IGNORECASE):
                            start = max(0, match.start() - fenetre)
                            end = min(len(contenu), match.end() + fenetre)
                            fenetres.append((filename, contenu[start:end]))
                            if filename not in file_usage_count:
                                file_usage_count[filename] = 0
                            file_usage_count[filename] += 1
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {filename}: {e}")
    return fenetres, file_usage_count

# Fonction pour générer une réponse via OpenAI
async def generer_reponse(question, fenetres, file_usage_count, model="gpt-3.5-turbo"):
    """
    Génère une réponse basée sur la question et le contexte extrait.
    """
    if fenetres:
        print("Utilisation des données de la base de données.")
        for filename, count in file_usage_count.items():
            print(f"Fichier utilisé : {filename} ({count} fois)")
        contexte = "\n---\n".join([f"Fichier: {filename}\nContenu: {contenu}" for filename, contenu in fenetres])
    else:
        print("Aucune donnée pertinente trouvée dans la base de données. Utilisation de la base de connaissances d'OpenAI.")
        contexte = ""

    prompt = (
        f"Voici une question posée par un utilisateur : {question}\n"
        f"Voici des extraits de ma base de données pour répondre à cette question :\n{contexte}\n"
        f"Fournis une réponse claire et concise en utilisant les informations ci-dessus."
    )
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui fournit des réponses basées sur des informations contextuelles."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"], response.model
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API OpenAI: {e}")
        return f"Erreur : {str(e)}", None

# Fonction pour reformuler la réponse via OpenAI
async def reformuler_reponse(reponse_initiale, model="gpt-3.5-turbo"):
    """
    Reformule la réponse initiale en utilisant la base de connaissances d'OpenAI.
    """
    prompt = (
        f"Voici une réponse initiale : {reponse_initiale}\n"
        f"Reformule cette réponse en utilisant tes connaissances pour la rendre plus naturelle et complète."
    )
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui reformule les réponses pour les rendre plus naturelles et complètes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"], response.model
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API OpenAI pour reformuler la réponse: {e}")
        return f"Erreur : {str(e)}", None

# Fonction pour obtenir une réponse à une question
async def get_reply_for_question(user_id, question):
    """Génère une réponse pour une question utilisateur en utilisant OpenAI."""
    try:
        if user_id not in user_histories:
            user_histories[user_id] = []

        user_histories[user_id].append({"role": "user", "content": question})

        # Extraction des mots-clés
        mots_cles = extraire_mots_cles(question)

        # Recherche des fenêtres contextuelles
        fenetres, file_usage_count = rechercher_fenetres_texte(mots_cles, TXT_FOLDER)

        # Trier les fichiers par ordre décroissant de leur nombre d'utilisations
        sorted_files = sorted(file_usage_count.items(), key=lambda x: x[1], reverse=True)

        # Limiter le nombre de fenêtres pour éviter de dépasser la limite de tokens
        fenetres = [(filename, contenu) for filename, contenu in fenetres if filename in [f[0] for f in sorted_files]]
        fenetres = fenetres[:5]  # Limiter à 5 fenêtres pour réduire la taille de l'entrée

        # Génération de la réponse initiale
        reponse_initiale, model_utilise = await generer_reponse(question, fenetres, file_usage_count)

        # Reformuler la réponse initiale
        reponse_finale, _ = await reformuler_reponse(reponse_initiale)

        user_histories[user_id].append({"role": "assistant", "content": reponse_finale})

        while len(user_histories[user_id]) > max_chat_history:
            user_histories[user_id].pop(0)

        return reponse_finale

    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {e}")
        return f"❌ Une erreur s'est produite : {str(e)}"

@client.event
async def on_ready():
    """Événement déclenché lorsque le bot est prêt."""
    print(f"{client.user.name} est prêt et connecté à Discord!")

@client.event
async def on_message(message):
    """Événement déclenché lorsqu'un message est reçu."""
    if message.author == client.user:
        return  # Ignorer les messages du bot lui-même

    # Vérifier si le message commence par '!'
    if message.content.startswith("!"):
        # Si c'est le cas, envoie la réponse générée
        question = message.content[1:]  # Enlève le '!' du début
        async with message.channel.typing():
            response = await get_reply_for_question(message.author.id, question)
            await message.channel.send(response)

# Lancer le bot
client.run(DISCORD_TOKEN)
