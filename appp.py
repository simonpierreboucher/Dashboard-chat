import os
import re
import uuid
import json
import hashlib
import requests
import threading
import tiktoken
import numpy as np
import datetime
import csv
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlparse
import openai
import markdown
import faiss
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Télécharger les ressources NLTK nécessaires
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Configuration
PORT = 8000
HOST = "localhost"
OPENAI_API_KEY = "METTEZ_VOTRE_CLE_API_OPENAI_ICI"  # Remplacez par votre clé API
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
MAX_CRAWL_DEPTH = 3
MAX_PAGES = 100
CHUNK_SIZE = 500
VECTOR_DIMENSION = 1536  # Dimension pour le modèle d'embedding d'OpenAI

# Initialisation de l'API OpenAI
openai.api_key = OPENAI_API_KEY

# Base de données en mémoire
users_db = {}
sessions = {}
chatbots = {}
vector_stores = {}
conversations = {}  # Pour stocker l'historique des conversations
urls_to_crawl = {}  # Pour stocker les URLs découvertes avant extraction

# Initialisation du tokenizer et de l'analyseur de sentiment
tokenizer = tiktoken.get_encoding("cl100k_base")
sia = SentimentIntensityAnalyzer()

# Dossier pour stocker les données
os.makedirs("data", exist_ok=True)
os.makedirs("data/vector_stores", exist_ok=True)
os.makedirs("data/chatbots", exist_ok=True)
os.makedirs("data/users", exist_ok=True)
os.makedirs("data/conversations", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)

# Charger les données existantes
try:
    with open("data/users.json", "r") as f:
        users_db = json.load(f)
except FileNotFoundError:
    pass

try:
    with open("data/chatbots.json", "r") as f:
        chatbots = json.load(f)
except FileNotFoundError:
    pass

try:
    with open("data/conversations.json", "r") as f:
        conversations = json.load(f)
except FileNotFoundError:
    pass

try:
    with open("data/urls_to_crawl.json", "r") as f:
        urls_to_crawl = json.load(f)
except FileNotFoundError:
    pass

# Web Crawler amélioré
class WebCrawler:
    def __init__(self, start_url, max_depth=MAX_CRAWL_DEPTH, max_pages=MAX_PAGES):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.urls_to_crawl = {}
        self.base_domain = urlparse(start_url).netloc
        self.pages_content = []
        
    def is_valid_url(self, url):
        parsed_url = urlparse(url)
        return bool(parsed_url.netloc and parsed_url.scheme and parsed_url.netloc == self.base_domain)
    
    def get_page_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Suppression des éléments non pertinents
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extraction du titre et du contenu principal
                title = soup.title.string if soup.title else "Sans titre"
                
                # Extraction du texte
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\n+', '\n', text)  # Nettoyer les sauts de ligne multiples
                
                # Création d'une version markdown du contenu
                md_content = f"# {title}\n\n{text}\n\nURL: {url}"
                
                return md_content
        except Exception as e:
            print(f"Erreur lors de l'extraction de {url}: {e}")
        return None
    
    def extract_links(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                links = []
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    absolute_url = urljoin(url, href)
                    if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                        links.append(absolute_url)
                
                return links
        except Exception as e:
            print(f"Erreur lors de l'extraction des liens de {url}: {e}")
        return []
    
    def discover_urls(self, url, depth=0):
        """
        Découvre toutes les URLs sans extraire le contenu
        """
        if depth > self.max_depth or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        print(f"Découverte de: {url} (profondeur: {depth})")
        
        # Ajouter l'URL à la liste à crawler
        self.urls_to_crawl[url] = {
            "url": url,
            "depth": depth,
            "title": self.get_page_title(url)
        }
        
        if depth < self.max_depth:
            links = self.extract_links(url)
            for link in links:
                self.discover_urls(link, depth + 1)
    
    def get_page_title(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup.title.string if soup.title else "Sans titre"
        except:
            pass
        return "Sans titre"
    
    def extract_selected_urls(self, selected_urls):
        """
        Extrait le contenu uniquement des URLs sélectionnées
        """
        for url in selected_urls:
            if url in self.urls_to_crawl:
                content = self.get_page_content(url)
                if content:
                    self.pages_content.append({
                        "url": url,
                        "content": content,
                        "depth": self.urls_to_crawl[url]["depth"]
                    })
        
        return self.pages_content
    
    def start_discovery(self):
        """
        Démarre la découverte des URLs sans extraction
        """
        self.discover_urls(self.start_url)
        return self.urls_to_crawl

# Traitement du texte et embeddings
def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk = tokenizer.decode(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def get_embedding(text):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response["data"][0]["embedding"]

def create_vector_store(texts, metadatas):
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    # Création de l'index FAISS
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index_data = np.array(embeddings).astype('float32')
    index.add(index_data)
    
    return {
        "index": index,
        "texts": texts,
        "metadatas": metadatas
    }

def search_vector_store(vector_store, query, k=5):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype('float32')
    
    # Recherche des k plus proches voisins
    distances, indices = vector_store["index"].search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(vector_store["texts"]):
            results.append({
                "text": vector_store["texts"][idx],
                "metadata": vector_store["metadatas"][idx],
                "distance": float(distances[0][i])
            })
    
    return results

# Analyse de sentiment et classification de requêtes
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def classify_query(text):
    """
    Classifie le type de requête
    """
    # Analyse avec l'API d'OpenAI pour classification
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui doit classifier une question dans une seule des catégories suivantes: question_information, demande_aide, reclamation, feedback, autre. Réponds uniquement avec le nom de la catégorie."},
                {"role": "user", "content": text}
            ],
            max_tokens=20
        )
        classification = response.choices[0].message['content'].strip().lower()
        
        # Normaliser la classification
        for category in ["question_information", "demande_aide", "reclamation", "feedback", "autre"]:
            if category in classification:
                return category
                
        return "autre"
    except:
        # Méthode de secours basée sur des règles simples
        text = text.lower()
        if any(word in text for word in ["comment", "quoi", "qui", "quand", "où", "pourquoi", "combien"]):
            return "question_information"
        elif any(word in text for word in ["aide", "aider", "assister", "besoin", "problème"]):
            return "demande_aide"
        elif any(word in text for word in ["plainte", "problème", "bug", "erreur", "mauvais", "cassé"]):
            return "reclamation"
        elif any(word in text for word in ["j'aime", "super", "génial", "excellent", "merci", "apprécier"]):
            return "feedback"
        else:
            return "autre"

# Gestion des chatbots
def create_chatbot(user_id, name, description, welcome_message):
    chatbot_id = str(uuid.uuid4())
    share_token = hashlib.md5(f"{chatbot_id}-{datetime.now()}".encode()).hexdigest()
    
    chatbot = {
        "id": chatbot_id,
        "user_id": user_id,
        "name": name,
        "description": description,
        "welcome_message": welcome_message,
        "created_at": datetime.now().isoformat(),
        "share_token": share_token,
        "knowledge_sources": [],
        "appearance": {
            "primary_color": "#0084ff",
            "secondary_color": "#f0f0f0",
            "bubble_style": "rounded",
            "font": "Arial, sans-serif",
            "logo": None,
            "title": name,
            "button_text": "Envoyer",
            "placeholder_text": "Posez votre question..."
        },
        "suggested_messages": [
            "Qu'est-ce que vous proposez ?",
            "Comment puis-je vous contacter ?",
            "Quels sont vos horaires ?"
        ]
    }
    
    chatbots[chatbot_id] = chatbot
    save_chatbots()
    
    # Initialiser la structure pour les conversations de ce chatbot
    conversations[chatbot_id] = []
    save_conversations()
    
    return chatbot

def update_chatbot_appearance(chatbot_id, appearance_data):
    """
    Met à jour l'apparence du chatbot
    """
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
    
    chatbot = chatbots[chatbot_id]
    
    # Mettre à jour les champs fournis
    for key, value in appearance_data.items():
        if key in chatbot["appearance"]:
            chatbot["appearance"][key] = value
    
    # Si des messages suggérés sont fournis
    if "suggested_messages" in appearance_data:
        chatbot["suggested_messages"] = appearance_data["suggested_messages"]
    
    save_chatbots()
    return True, "Apparence mise à jour avec succès"

def discover_website_urls(chatbot_id, website_url):
    """
    Découvre toutes les URLs d'un site web sans extraire le contenu
    """
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
    
    # Crawl le site web pour découvrir les URLs
    crawler = WebCrawler(website_url)
    discovered_urls = crawler.start_discovery()
    
    # Enregistrer les URLs découvertes
    urls_to_crawl[chatbot_id] = discovered_urls
    save_urls_to_crawl()
    
    return True, list(discovered_urls.keys())

def extract_content_from_selected_urls(chatbot_id, selected_urls, website_url):
    """
    Extrait le contenu des URLs sélectionnées
    """
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
        
    if chatbot_id not in urls_to_crawl:
        return False, "Aucune URL découverte pour ce chatbot"
    
    # Crawl le site web pour les URLs sélectionnées
    crawler = WebCrawler(website_url)
    crawler.urls_to_crawl = urls_to_crawl[chatbot_id]
    pages = crawler.extract_selected_urls(selected_urls)
    
    if not pages:
        return False, "Aucune page n'a pu être extraite des URLs sélectionnées"
    
    # Chunking et embeddings
    all_chunks = []
    all_metadatas = []
    
    for page in pages:
        chunks = chunk_text(page["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "url": page["url"],
                "depth": page["depth"]
            })
    
    if not all_chunks:
        return False, "Aucun contenu à traiter"
    
    # Créer le vector store
    vector_store = create_vector_store(all_chunks, all_metadatas)
    
    # Ajouter la source de connaissance au chatbot
    knowledge_id = str(uuid.uuid4())
    chatbots[chatbot_id]["knowledge_sources"].append({
        "id": knowledge_id,
        "url": website_url,
        "pages_count": len(pages),
        "chunks_count": len(all_chunks),
        "added_at": datetime.now().isoformat(),
        "urls": selected_urls
    })
    
    # Sauvegarder le vector store
    vector_stores[knowledge_id] = vector_store
    save_chatbots()
    
    # Sauvegarder l'index FAISS sur disque
    os.makedirs(f"data/vector_stores/{knowledge_id}", exist_ok=True)
    faiss.write_index(vector_store["index"], f"data/vector_stores/{knowledge_id}/index.faiss")
    
    with open(f"data/vector_stores/{knowledge_id}/data.json", "w") as f:
        json.dump({
            "texts": all_chunks,
            "metadatas": all_metadatas
        }, f)
    
    return True, knowledge_id

def add_knowledge_to_chatbot(chatbot_id, website_url):
    """
    Version simplifiée pour la rétrocompatibilité
    """
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
    
    # Crawl le site web
    crawler = WebCrawler(website_url)
    discovered_urls = crawler.start_discovery()
    
    # Enregistrer les URLs découvertes
    urls_to_crawl[chatbot_id] = discovered_urls
    save_urls_to_crawl()
    
    # Extraire le contenu de toutes les URLs
    return extract_content_from_selected_urls(chatbot_id, list(discovered_urls.keys()), website_url)

def chat_with_bot(chatbot_id, user_input, conversation_history=None, user_id=None, session_id=None):
    if chatbot_id not in chatbots:
        return "Ce chatbot n'existe pas."
    
    chatbot = chatbots[chatbot_id]
    
    if not conversation_history:
        conversation_history = []
    
    # Rechercher les connaissances pertinentes
    relevant_context = []
    
    for source in chatbot["knowledge_sources"]:
        knowledge_id = source["id"]
        
        # Charger le vector store s'il n'est pas en mémoire
        if knowledge_id not in vector_stores:
            try:
                # Charger l'index FAISS
                index = faiss.read_index(f"data/vector_stores/{knowledge_id}/index.faiss")
                
                with open(f"data/vector_stores/{knowledge_id}/data.json", "r") as f:
                    data = json.load(f)
                
                vector_stores[knowledge_id] = {
                    "index": index,
                    "texts": data["texts"],
                    "metadatas": data["metadatas"]
                }
            except Exception as e:
                print(f"Erreur lors du chargement du vector store {knowledge_id}: {e}")
                continue
        
        # Rechercher les contenus pertinents
        results = search_vector_store(vector_stores[knowledge_id], user_input)
        for result in results:
            relevant_context.append(result["text"])
    
    # Construire le contexte pour GPT-4
    context = "\n\n".join(relevant_context)
    
    system_prompt = f"""Tu es un assistant IA nommé "{chatbot['name']}". {chatbot['description']}
Tu dois répondre aux questions en te basant sur le contexte fourni et en ignorant toute information hors contexte.
Si tu ne trouves pas l'information dans le contexte, réponds poliment que tu n'as pas cette information.
Ne mentionne pas le contexte qui t'a été fourni dans ta réponse."""
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Ajouter l'historique de conversation
    for message in conversation_history:
        messages.append(message)
    
    # Ajouter le contexte et la question actuelle
    if context:
        messages.append({"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {user_input}"})
    else:
        messages.append({"role": "user", "content": user_input})
    
    # Appel à l'API ChatGPT
    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=messages
    )
    
    ai_response = response["choices"][0]["message"]["content"]
    
    # Enregistrer cette conversation
    if chatbot_id not in conversations:
        conversations[chatbot_id] = []
    
    # Analyser le sentiment et classifier la requête
    sentiment = analyze_sentiment(user_input)
    query_type = classify_query(user_input)
    
    # Enregistrer la conversation avec les métadonnées
    conversation_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "user_id": user_id,  # Pourrait être None pour les visiteurs anonymes
        "session_id": session_id,  # Pour grouper les conversations d'une même session
        "metadata": {
            "sentiment": sentiment,
            "query_type": query_type,
            "ip_address": None,  # À remplir plus tard si nécessaire
            "user_agent": None   # À remplir plus tard si nécessaire
        }
    }
    
    conversations[chatbot_id].append(conversation_entry)
    save_conversations()
    
    return ai_response

def upload_logo(chatbot_id, file_data, file_name):
    """
    Enregistre le logo uploadé et met à jour le chatbot
    """
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
    
    # Vérifier l'extension du fichier
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    file_ext = os.path.splitext(file_name)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, "Format de fichier non supporté. Utilisez PNG, JPG, JPEG, GIF ou SVG."
    
    # Créer un nom de fichier unique
    unique_filename = f"{chatbot_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
    file_path = f"data/uploads/{unique_filename}"
    
    # Enregistrer le fichier
    with open(file_path, 'wb') as f:
        f.write(file_data)
    
    # Mettre à jour le chatbot
    chatbots[chatbot_id]["appearance"]["logo"] = unique_filename
    save_chatbots()
    
    return True, unique_filename

def get_chatbot_metrics(chatbot_id):
    """
    Génère des métriques pour le chatbot
    """
    if chatbot_id not in conversations:
        return {
            "total_conversations": 0,
            "sentiment_distribution": {},
            "query_types": {},
            "conversations_over_time": {},
            "busiest_hour": None,
            "avg_response_length": 0
        }
    
    chatbot_conversations = conversations[chatbot_id]
    
    # Nombre total de conversations
    total_conversations = len(chatbot_conversations)
    
    # Distribution des sentiments
    sentiment_distribution = Counter()
    for conv in chatbot_conversations:
        sentiment = conv["metadata"]["sentiment"]
        sentiment_distribution[sentiment] += 1
    
    # Types de requêtes
    query_types = Counter()
    for conv in chatbot_conversations:
        query_type = conv["metadata"]["query_type"]
        query_types[query_type] += 1
    
    # Conversations dans le temps (par jour)
    conversations_over_time = defaultdict(int)
    for conv in chatbot_conversations:
        date = datetime.fromisoformat(conv["timestamp"]).strftime("%Y-%m-%d")
        conversations_over_time[date] += 1
    
    # Heure la plus chargée
    hour_distribution = Counter()
    for conv in chatbot_conversations:
        hour = datetime.fromisoformat(conv["timestamp"]).hour
        hour_distribution[hour] += 1
    
    busiest_hour = hour_distribution.most_common(1)[0][0] if hour_distribution else None
    
    # Longueur moyenne des réponses
    total_response_length = sum(len(conv["ai_response"]) for conv in chatbot_conversations)
    avg_response_length = total_response_length / total_conversations if total_conversations > 0 else 0
    
    return {
        "total_conversations": total_conversations,
        "sentiment_distribution": dict(sentiment_distribution),
        "query_types": dict(query_types),
        "conversations_over_time": dict(sorted(conversations_over_time.items())),
        "busiest_hour": busiest_hour,
        "avg_response_length": round(avg_response_length, 2)
    }

def generate_metrics_charts(chatbot_id):
    """
    Génère des graphiques pour les métriques du chatbot
    """
    metrics = get_chatbot_metrics(chatbot_id)
    
    # Créer un dossier pour les métriques de ce chatbot
    os.makedirs(f"data/metrics/{chatbot_id}", exist_ok=True)
    
    charts = {}
    
    # 1. Graphique de distribution des sentiments
    if metrics["sentiment_distribution"]:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(metrics["sentiment_distribution"].keys()), 
                    y=list(metrics["sentiment_distribution"].values()))
        plt.title('Distribution des Sentiments')
        plt.xlabel('Sentiment')
        plt.ylabel('Nombre de conversations')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        sentiment_chart_path = f"data/metrics/{chatbot_id}/sentiment_distribution.png"
        plt.savefig(sentiment_chart_path)
        plt.close()
        
        # Convertir en base64 pour l'affichage HTML
        with open(sentiment_chart_path, "rb") as image_file:
            charts["sentiment_distribution"] = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 2. Graphique des types de requêtes
    if metrics["query_types"]:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(metrics["query_types"].keys()), 
                    y=list(metrics["query_types"].values()))
        plt.title('Types de Requêtes')
        plt.xlabel('Type de requête')
        plt.ylabel('Nombre de conversations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        query_types_chart_path = f"data/metrics/{chatbot_id}/query_types.png"
        plt.savefig(query_types_chart_path)
        plt.close()
        
        with open(query_types_chart_path, "rb") as image_file:
            charts["query_types"] = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 3. Graphique des conversations dans le temps
    if metrics["conversations_over_time"]:
        plt.figure(figsize=(12, 6))
        dates = list(metrics["conversations_over_time"].keys())
        counts = list(metrics["conversations_over_time"].values())
        
        plt.plot(dates, counts, marker='o')
        plt.title('Conversations par Jour')
        plt.xlabel('Date')
        plt.ylabel('Nombre de conversations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        time_chart_path = f"data/metrics/{chatbot_id}/conversations_over_time.png"
        plt.savefig(time_chart_path)
        plt.close()
        
        with open(time_chart_path, "rb") as image_file:
            charts["conversations_over_time"] = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 4. Graphique de distribution par heure
    hours = [i for i in range(24)]
    hour_counts = [0] * 24
    
    if chatbot_id in conversations:
        for conv in conversations[chatbot_id]:
            hour = datetime.fromisoformat(conv["timestamp"]).hour
            hour_counts[hour] += 1
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hours, y=hour_counts)
    plt.title('Distribution des Conversations par Heure')
    plt.xlabel('Heure')
    plt.ylabel('Nombre de conversations')
    plt.tight_layout()
    
    hour_chart_path = f"data/metrics/{chatbot_id}/hour_distribution.png"
    plt.savefig(hour_chart_path)
    plt.close()
    
    with open(hour_chart_path, "rb") as image_file:
        charts["hour_distribution"] = base64.b64encode(image_file.read()).decode('utf-8')
    
    return charts

def export_conversations(chatbot_id, format_type="csv"):
    """
    Exporte les conversations au format CSV ou JSON
    """
    if chatbot_id not in conversations:
        return None, "Aucune conversation trouvée pour ce chatbot"
    
    chatbot_conversations = conversations[chatbot_id]
    
    if format_type == "csv":
        output = io.StringIO()
        fieldnames = ["id", "timestamp", "user_input", "ai_response", "sentiment", "query_type"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for conv in chatbot_conversations:
            writer.writerow({
                "id": conv["id"],
                "timestamp": conv["timestamp"],
                "user_input": conv["user_input"],
                "ai_response": conv["ai_response"],
                "sentiment": conv["metadata"]["sentiment"],
                "query_type": conv["metadata"]["query_type"]
            })
        
        return output.getvalue(), "text/csv"
    
    elif format_type == "json":
        return json.dumps(chatbot_conversations, indent=2), "application/json"
    
    else:
        return None, "Format non pris en charge"

def generate_widget_code(chatbot_id):
    chatbot = chatbots.get(chatbot_id)
    if not chatbot:
        return None
    
    share_token = chatbot["share_token"]
    host = f"http://{HOST}:{PORT}"
    appearance = chatbot["appearance"]
    suggested_messages = chatbot["suggested_messages"]
    
    # Code HTML et JS pour le widget
    widget_code = f"""
<!-- {chatbot['name']} Chatbot Widget -->
<div id="ai-chatbot-widget">
  <div id="ai-chatbot-toggle" style="background-color: {appearance['primary_color']};">
    <img id="ai-chatbot-toggle-icon" src="{host}/uploads/{appearance['logo']}" onerror="this.src='{host}/static/default_logo.png'; this.onerror=null;" alt="Chat" style="width: 60%; height: 60%; object-fit: contain;">
  </div>
  <div id="ai-chatbot-container" style="display:none; font-family: {appearance['font']};">
    <div id="ai-chatbot-header" style="background-color: {appearance['primary_color']};">
      <h3>{appearance['title']}</h3>
      <button id="ai-chatbot-close">×</button>
    </div>
    <div id="ai-chatbot-messages"></div>
    <div id="ai-chatbot-suggested-questions" style="display: none;"></div>
    <div id="ai-chatbot-input-container">
      <input type="text" id="ai-chatbot-input" placeholder="{appearance['placeholder_text']}">
      <button id="ai-chatbot-send" style="background-color: {appearance['primary_color']};">{appearance['button_text']}</button>
    </div>
  </div>
</div>

<style>
  #ai-chatbot-widget {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-family: {appearance['font']};
    z-index: 9999;
  }}
  
  #ai-chatbot-toggle {{
    width: 60px;
    height: 60px;
    background-color: {appearance['primary_color']};
    color: white;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 24px;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  }}
  
  #ai-chatbot-container {{
    position: absolute;
    bottom: 70px;
    right: 0;
    width: 350px;
    height: 500px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.2);
    display: flex;
    flex-direction: column;
  }}
  
  #ai-chatbot-header {{
    padding: 15px;
    background-color: {appearance['primary_color']};
    color: white;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  
  #ai-chatbot-header h3 {{
    margin: 0;
  }}
  
  #ai-chatbot-close {{
    background: none;
    border: none;
    color: white;
    font-size: 24px;
    cursor: pointer;
  }}
  
  #ai-chatbot-messages {{
    flex-grow: 1;
    padding: 15px;
    overflow-y: auto;
  }}
  
  .ai-chatbot-message {{
    margin-bottom: 10px;
    padding: 10px;
    border-radius: {appearance['bubble_style'] == 'rounded' ? '10px' : '0'};
    max-width: 80%;
  }}
  
  .ai-chatbot-user-message {{
    background-color: {appearance['primary_color']};
    color: white;
    margin-left: auto;
  }}
  
  .ai-chatbot-bot-message {{
    background-color: {appearance['secondary_color']};
    margin-right: auto;
  }}
  
  #ai-chatbot-suggested-questions {{
    padding: 10px 15px;
    border-top: 1px solid #eee;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }}
  
  .ai-chatbot-suggested-question {{
    background-color: #f5f5f5;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    cursor: pointer;
  }}
  
  .ai-chatbot-suggested-question:hover {{
    background-color: {appearance['primary_color']};
    color: white;
  }}
  
  #ai-chatbot-input-container {{
    padding: 15px;
    display: flex;
    border-top: 1px solid #eee;
  }}
  
  #ai-chatbot-input {{
    flex-grow: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 20px;
    outline: none;
  }}
  
  #ai-chatbot-send {{
    margin-left: 10px;
    background-color: {appearance['primary_color']};
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 15px;
    cursor: pointer;
  }}
</style>

<script>
  (function() {{
    const widget = document.getElementById('ai-chatbot-widget');
    const toggle = document.getElementById('ai-chatbot-toggle');
    const container = document.getElementById('ai-chatbot-container');
    const closeBtn = document.getElementById('ai-chatbot-close');
    const messages = document.getElementById('ai-chatbot-messages');
    const suggestedQuestions = document.getElementById('ai-chatbot-suggested-questions');
    const input = document.getElementById('ai-chatbot-input');
    const sendBtn = document.getElementById('ai-chatbot-send');
    
    const chatbotId = '{chatbot_id}';
    const shareToken = '{share_token}';
    const apiUrl = '{host}/api/chat/' + chatbotId;
    
    let conversation = [];
    
    // Messages suggérés
    const suggestedMessagesData = {json.dumps(suggested_messages)};
    
    // Afficher le message de bienvenue
    function showWelcomeMessage() {{
      addMessage('{chatbot["welcome_message"]}', 'bot');
      showSuggestedQuestions();
    }}
    
    function showSuggestedQuestions() {{
      suggestedQuestions.innerHTML = '';
      suggestedQuestions.style.display = 'flex';
      
      suggestedMessagesData.forEach(question => {{
        const questionElement = document.createElement('div');
        questionElement.classList.add('ai-chatbot-suggested-question');
        questionElement.textContent = question;
        questionElement.addEventListener('click', function() {{
          input.value = question;
          sendMessage();
        }});
        suggestedQuestions.appendChild(questionElement);
      }});
    }}
    
    // Afficher/masquer le chatbot
    toggle.addEventListener('click', function() {{
      container.style.display = 'flex';
      toggle.style.display = 'none';
      if (messages.childElementCount === 0) {{
        showWelcomeMessage();
      }}
    }});
    
    closeBtn.addEventListener('click', function() {{
      container.style.display = 'none';
      toggle.style.display = 'flex';
    }});
    
    // Envoyer un message
    function sendMessage() {{
      const text = input.value.trim();
      if (!text) return;
      
      addMessage(text, 'user');
      input.value = '';
      
      // Masquer les questions suggérées pendant la conversation
      suggestedQuestions.style.display = 'none';
      
      conversation.push({{ role: 'user', content: text }});
      
      fetch(apiUrl, {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json'
        }},
        body: JSON.stringify({{
          token: shareToken,
          message: text,
          conversation: conversation
        }})
      }})
      .then(response => response.json())
      .then(data => {{
        const reply = data.response;
        addMessage(reply, 'bot');
        conversation.push({{ role: 'assistant', content: reply }});
        
        // Réafficher les questions suggérées après la réponse
        showSuggestedQuestions();
      }})
      .catch(error => {{
        console.error('Error:', error);
        addMessage('Désolé, une erreur est survenue. Veuillez réessayer plus tard.', 'bot');
      }});
    }}
    
    // Ajouter un message à la conversation
    function addMessage(text, sender) {{
      const messageElement = document.createElement('div');
      messageElement.classList.add('ai-chatbot-message');
      messageElement.classList.add(sender === 'user' ? 'ai-chatbot-user-message' : 'ai-chatbot-bot-message');
      messageElement.textContent = text;
      messages.appendChild(messageElement);
      messages.scrollTop = messages.scrollHeight;
    }}
    
    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', function(e) {{
      if (e.key === 'Enter') sendMessage();
    }});
  }})();
</script>
<!-- End of {chatbot['name']} Chatbot Widget -->
"""
    
    return widget_code

# Gestion des utilisateurs et sessions
def register_user(username, password):
    if username in users_db:
        return False, "Nom d'utilisateur déjà pris"
    
    salt = os.urandom(16).hex()
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    
    user = {
        "id": str(uuid.uuid4()),
        "username": username,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now().isoformat()
    }
    
    users_db[username] = user
    save_users()
    
    return True, user["id"]

def login_user(username, password):
    if username not in users_db:
        return False, "Utilisateur non trouvé"
    
    user = users_db[username]
    password_hash = hashlib.sha256((password + user["salt"]).encode()).hexdigest()
    
    if password_hash != user["password_hash"]:
        return False, "Mot de passe incorrect"
    
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_id": user["id"],
        "expires": (datetime.now() + timedelta(hours=24)).isoformat()
    }
    
    return True, session_id

def get_user_chatbots(user_id):
    return {id: bot for id, bot in chatbots.items() if bot["user_id"] == user_id}

# Sauvegarde des données
def save_users():
    with open("data/users.json", "w") as f:
        json.dump(users_db, f)

def save_chatbots():
    with open("data/chatbots.json", "w") as f:
        json.dump(chatbots, f)

def save_conversations():
    with open("data/conversations.json", "w") as f:
        json.dump(conversations, f)

def save_urls_to_crawl():
    with open("data/urls_to_crawl.json", "w") as f:
        json.dump(urls_to_crawl, f)

# Serveur HTTP
class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="text/html", status_code=200):
        self.send_response(status_code)
        self.send_header("Content-type", content_type)
        self.end_headers()
    
    def _get_session(self):
        cookie = SimpleCookie()
        if "Cookie" in self.headers:
            cookie.load(self.headers["Cookie"])
        
        if "session_id" in cookie:
            session_id = cookie["session_id"].value
            if session_id in sessions:
                session = sessions[session_id]
                if datetime.fromisoformat(session["expires"]) > datetime.now():
                    return session_id, session["user_id"]
        
        return None, None
    
    def _send_redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()
    
    def _get_post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        return parse_qs(post_data)
    
    def _get_json_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        return json.loads(post_data)
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        session_id, user_id = self._get_session()
        
        # Routes d'API
        if path.startswith("/api/"):
            return self._handle_api_get(path, query, user_id)
        
        # Route pour les ressources statiques
        if path.startswith("/static/"):
            return self._serve_static_file(path)
            
        # Route pour les uploads
        if path.startswith("/uploads/"):
            return self._serve_upload_file(path)
        
        # Route pour les exports
        if path.startswith("/export/"):
            return self._handle_export(path, user_id)
        
        # Route de partage de chatbot
        if path.startswith("/share/"):
            return self._serve_shared_chatbot(path)
        
        # Routes principales
        if path == "/":
            if user_id:
                return self._serve_dashboard(user_id)
            else:
                return self._serve_homepage()
        
        elif path == "/login":
            return self._serve_login_page()
        
        elif path == "/register":
            return self._serve_register_page()
        
        elif path == "/dashboard":
            if user_id:
                return self._serve_dashboard(user_id)
            else:
                return self._send_redirect("/login")
        
        elif path == "/create-chatbot":
            if user_id:
                return self._serve_create_chatbot_page(user_id)
            else:
                return self._send_redirect("/login")
        
        elif path.startswith("/chatbot/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._serve_chatbot_page(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
                
        elif path.startswith("/conversations/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._serve_conversations_page(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
                
        elif path.startswith("/metrics/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._serve_metrics_page(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
                
        elif path.startswith("/appearance/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._serve_appearance_page(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
        
        else:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
    
    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        session_id, user_id = self._get_session()
        
        # Routes d'API
        if path.startswith("/api/"):
            return self._handle_api_post(path, user_id, session_id)
        
        # Routes d'authentification
        if path == "/register":
            return self._handle_register()
        
        elif path == "/login":
            return self._handle_login()
        
        elif path == "/logout":
            return self._handle_logout(session_id)
        
        # Routes de gestion des chatbots
        elif path == "/create-chatbot":
            if user_id:
                return self._handle_create_chatbot(user_id)
            else:
                return self._send_redirect("/login")
        
        elif path.startswith("/chatbot/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._handle_chatbot_action(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
                
        elif path.startswith("/appearance/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._handle_appearance_update(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
                
        elif path.startswith("/upload-logo/"):
            if user_id:
                chatbot_id = path.split("/")[-1]
                return self._handle_logo_upload(user_id, chatbot_id)
            else:
                return self._send_redirect("/login")
        
        else:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
    
    # Handlers d'API
    def _handle_api_get(self, path, query, user_id):
        if path == "/api/chatbots" and user_id:
            user_chatbots = get_user_chatbots(user_id)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(user_chatbots).encode())
        
        elif path.startswith("/api/chatbot/") and user_id:
            chatbot_id = path.split("/")[-1]
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                self._set_headers("application/json")
                self.wfile.write(json.dumps(chatbots[chatbot_id]).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
                
        elif path.startswith("/api/conversations/") and user_id:
            chatbot_id = path.split("/")[-1]
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                # Récupérer les conversations du chatbot
                chatbot_conversations = conversations.get(chatbot_id, [])
                self._set_headers("application/json")
                self.wfile.write(json.dumps(chatbot_conversations).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
                
        elif path.startswith("/api/metrics/") and user_id:
            chatbot_id = path.split("/")[-1]
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                # Récupérer les métriques du chatbot
                metrics = get_chatbot_metrics(chatbot_id)
                self._set_headers("application/json")
                self.wfile.write(json.dumps(metrics).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
        
        else:
            self._set_headers("application/json", 404)
            self.wfile.write(json.dumps({"error": "API endpoint not found"}).encode())
    
    def _handle_api_post(self, path, user_id, session_id):
        if path == "/api/add-knowledge" and user_id:
            data = self._get_json_data()
            chatbot_id = data.get("chatbot_id")
            website_url = data.get("website_url")
            
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                success, message = add_knowledge_to_chatbot(chatbot_id, website_url)
                self._set_headers("application/json")
                self.wfile.write(json.dumps({
                    "success": success,
                    "message": message
                }).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
        
        elif path == "/api/discover-urls" and user_id:
            data = self._get_json_data()
            chatbot_id = data.get("chatbot_id")
            website_url = data.get("website_url")
            
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                success, urls = discover_website_urls(chatbot_id, website_url)
                self._set_headers("application/json")
                self.wfile.write(json.dumps({
                    "success": success,
                    "urls": urls if isinstance(urls, list) else []
                }).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
                
        elif path == "/api/extract-content" and user_id:
            data = self._get_json_data()
            chatbot_id = data.get("chatbot_id")
            selected_urls = data.get("selected_urls", [])
            website_url = data.get("website_url")
            
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                success, message = extract_content_from_selected_urls(chatbot_id, selected_urls, website_url)
                self._set_headers("application/json")
                self.wfile.write(json.dumps({
                    "success": success,
                    "message": message
                }).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
                
        elif path == "/api/update-appearance" and user_id:
            data = self._get_json_data()
            chatbot_id = data.get("chatbot_id")
            appearance_data = data.get("appearance", {})
            
            if chatbot_id in chatbots and chatbots[chatbot_id]["user_id"] == user_id:
                success, message = update_chatbot_appearance(chatbot_id, appearance_data)
                self._set_headers("application/json")
                self.wfile.write(json.dumps({
                    "success": success,
                    "message": message
                }).encode())
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Chatbot not found"}).encode())
        
        elif path.startswith("/api/chat/"):
            chatbot_id = path.split("/")[-1]
            data = self._get_json_data()
            message = data.get("message", "")
            token = data.get("token", "")
            conversation = data.get("conversation", [])
            
            # Vérifier si c'est un accès par token ou par user_id
            is_authorized = False
            if chatbot_id in chatbots:
                if token and chatbots[chatbot_id]["share_token"] == token:
                    is_authorized = True
                elif user_id and chatbots[chatbot_id]["user_id"] == user_id:
                    is_authorized = True
            
            if is_authorized:
                response = chat_with_bot(chatbot_id, message, conversation, user_id, session_id)
                self._set_headers("application/json")
                self.wfile.write(json.dumps({
                    "response": response
                }).encode())
            else:
                self._set_headers("application/json", 403)
                self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
        
        else:
            self._set_headers("application/json", 404)
            self.wfile.write(json.dumps({"error": "API endpoint not found"}).encode())
    
    # Handlers de pages
    def _serve_static_file(self, path):
        try:
            file_path = path[1:]  # Supprimer le / initial
            
            if file_path.endswith(".css"):
                self._set_headers("text/css")
            elif file_path.endswith(".js"):
                self._set_headers("application/javascript")
            elif file_path.endswith(".png"):
                self._set_headers("image/png")
            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                self._set_headers("image/jpeg")
            else:
                self._set_headers()
            
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
            
    def _serve_upload_file(self, path):
        try:
            file_name = path.split("/")[-1]
            file_path = f"data/uploads/{file_name}"
            
            if file_path.endswith(".png"):
                self._set_headers("image/png")
            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                self._set_headers("image/jpeg")
            elif file_path.endswith(".gif"):
                self._set_headers("image/gif")
            elif file_path.endswith(".svg"):
                self._set_headers("image/svg+xml")
            else:
                self._set_headers()
            
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
            
    def _handle_export(self, path, user_id):
        parts = path.split("/")
        if len(parts) < 4:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
            return
        
        chatbot_id = parts[2]
        format_type = parts[3]
        
        # Vérifier l'autorisation
        if not user_id or chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._set_headers(status_code=403)
            self.wfile.write(b"403 Forbidden")
            return
        
        data, content_type = export_conversations(chatbot_id, format_type)
        
        if not data:
            self._set_headers(status_code=404)
            self.wfile.write(b"No data available")
            return
        
        filename = f"conversations_{chatbot_id}.{format_type}"
        
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        
        self.wfile.write(data.encode() if isinstance(data, str) else data)
    
    def _serve_shared_chatbot(self, path):
        parts = path.split("/")
        if len(parts) < 3:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
            return
        
        share_token = parts[2]
        
        # Trouver le chatbot par le token de partage
        shared_chatbot = None
        for chatbot_id, chatbot in chatbots.items():
            if chatbot["share_token"] == share_token:
                shared_chatbot = chatbot
                break
        
        if not shared_chatbot:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
            return
        
        # Récupérer les paramètres d'apparence
        appearance = shared_chatbot["appearance"]
        suggested_messages = shared_chatbot["suggested_messages"]
        
        # Servir l'interface du chatbot partagé
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{shared_chatbot['name']} - Chatbot IA</title>
            <style>
                body {{
                    font-family: {appearance['font']};
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .chat-container {{
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    overflow: hidden;
                    background-color: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chat-header {{
                    background-color: {appearance['primary_color']};
                    color: white;
                    padding: 15px;
                    text-align: center;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .chat-header img {{
                    width: 30px;
                    height: 30px;
                    margin-right: 10px;
                    object-fit: contain;
                }}
                .chat-messages {{
                    height: 400px;
                    padding: 15px;
                    overflow-y: scroll;
                    display: flex;
                    flex-direction: column;
                }}
                .message {{
                    max-width: 80%;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: {appearance['bubble_style'] == 'rounded' ? '10px' : '0'};
                }}
                .bot {{
                    background-color: {appearance['secondary_color']};
                    align-self: flex-start;
                }}
                .user {{
                    background-color: {appearance['primary_color']};
                    color: white;
                    align-self: flex-end;
                }}
                .chat-input {{
                    display: flex;
                    padding: 15px;
                    border-top: 1px solid #eee;
                }}
                .chat-input input {{
                    flex-grow: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 20px;
                    outline: none;
                }}
                .chat-input button {{
                    margin-left: 10px;
                    background-color: {appearance['primary_color']};
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 10px 15px;
                    cursor: pointer;
                }}
                .suggested-questions {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    padding: 10px 15px;
                    border-top: 1px solid #eee;
                }}
                .suggested-question {{
                    background-color: #f5f5f5;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                    cursor: pointer;
                }}
                .suggested-question:hover {{
                    background-color: {appearance['primary_color']};
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{shared_chatbot['name']}</h1>
                    <p>{shared_chatbot['description']}</p>
                </header>
                
                <div class="chat-container">
                    <div class="chat-header">
                        <img src="/uploads/{appearance['logo']}" onerror="this.src='/static/default_logo.png'; this.onerror=null;" alt="Logo">
                        <h2>{appearance['title']}</h2>
                    </div>
                    <div class="chat-messages" id="chat-messages">
                        <div class="message bot">{shared_chatbot['welcome_message']}</div>
                    </div>
                    <div class="suggested-questions" id="suggested-questions"></div>
                    <div class="chat-input">
                        <input type="text" id="user-input" placeholder="{appearance['placeholder_text']}">
                        <button id="send-button">{appearance['button_text']}</button>
                    </div>
                </div>
            </div>
            
            <script>
                const chatMessages = document.getElementById('chat-messages');
                const userInput = document.getElementById('user-input');
                const sendButton = document.getElementById('send-button');
                const suggestedQuestions = document.getElementById('suggested-questions');
                
                let conversation = [];
                
                // Ajouter les messages suggérés
                const suggestedMessagesData = {json.dumps(suggested_messages)};
                
                function showSuggestedQuestions() {{
                    suggestedQuestions.innerHTML = '';
                    suggestedMessagesData.forEach(question => {{
                        const questionElement = document.createElement('div');
                        questionElement.classList.add('suggested-question');
                        questionElement.textContent = question;
                        questionElement.addEventListener('click', function() {{
                            userInput.value = question;
                            sendMessage();
                        }});
                        suggestedQuestions.appendChild(questionElement);
                    }});
                }}
                
                // Afficher les messages suggérés au chargement
                showSuggestedQuestions();
                
                function addMessage(text, sender) {{
                    const messageElement = document.createElement('div');
                    messageElement.classList.add('message');
                    messageElement.classList.add(sender);
                    messageElement.textContent = text;
                    chatMessages.appendChild(messageElement);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }}
                
                function sendMessage() {{
                    const text = userInput.value.trim();
                    if (!text) return;
                    
                    addMessage(text, 'user');
                    userInput.value = '';
                    
                    conversation.push({{ role: 'user', content: text }});
                    
                    // Masquer temporairement les questions suggérées
                    suggestedQuestions.style.display = 'none';
                    
                    fetch('/api/chat/{shared_chatbot["id"]}', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            token: '{shared_chatbot["share_token"]}',
                            message: text,
                            conversation: conversation
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        const reply = data.response;
                        addMessage(reply, 'bot');
                        conversation.push({{ role: 'assistant', content: reply }});
                        
                        // Réafficher les questions suggérées
                        suggestedQuestions.style.display = 'flex';
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        addMessage('Désolé, une erreur est survenue. Veuillez réessayer plus tard.', 'bot');
                        suggestedQuestions.style.display = 'flex';
                    }});
                }}
                
                sendButton.addEventListener('click', sendMessage);
                userInput.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') sendMessage();
                }});
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_homepage(self):
        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Créez Votre Chatbot IA</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .hero {
                    background-color: #0084ff;
                    color: white;
                    padding: 60px 20px;
                    text-align: center;
                }
                .hero h1 {
                    font-size: 2.5rem;
                    margin-bottom: 20px;
                }
                .hero p {
                    font-size: 1.2rem;
                    margin-bottom: 30px;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                }
                .btn {
                    display: inline-block;
                    background-color: white;
                    color: #0084ff;
                    padding: 12px 24px;
                    border-radius: 30px;
                    text-decoration: none;
                    font-weight: bold;
                    margin: 10px;
                    transition: all 0.3s ease;
                }
                .btn:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px 20px;
                }
                .features {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 30px;
                    margin-top: 40px;
                }
                .feature {
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    width: 300px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .feature h3 {
                    margin-top: 0;
                    color: #0084ff;
                }
                .footer {
                    background-color: #333;
                    color: white;
                    text-align: center;
                    padding: 20px;
                    margin-top: 40px;
                }
            </style>
        </head>
        <body>
            <div class="hero">
                <h1>Créez Votre Propre Chatbot IA</h1>
                <p>Transformez votre site web avec un chatbot IA intelligent alimenté par GPT-4. Intégrez facilement des connaissances à partir de votre site pour offrir une assistance personnalisée à vos visiteurs.</p>
                <a href="/register" class="btn">S'inscrire</a>
                <a href="/login" class="btn">Se connecter</a>
            </div>
            
            <div class="container">
                <h2 style="text-align: center;">Fonctionnalités</h2>
                <div class="features">
                    <div class="feature">
                        <h3>Création Facile</h3>
                        <p>Créez votre chatbot en quelques clics sans aucune compétence technique nécessaire.</p>
                    </div>
                    <div class="feature">
                        <h3>Alimenté par GPT-4</h3>
                        <p>Utilisez la puissance de GPT-4 pour des réponses naturelles et intelligentes.</p>
                    </div>
                    <div class="feature">
                        <h3>Web Crawler Sélectif</h3>
                        <p>Choisissez précisément quelles pages de votre site web doivent être utilisées pour enrichir les connaissances de votre chatbot.</p>
                    </div>
                    <div class="feature">
                        <h3>Widget Personnalisable</h3>
                        <p>Personnalisez l'apparence de votre chatbot avec vos couleurs, logo, et messages suggérés.</p>
                    </div>
                    <div class="feature">
                        <h3>Analyses Avancées</h3>
                        <p>Suivez les performances de votre chatbot avec des métriques détaillées et l'historique des conversations.</p>
                    </div>
                    <div class="feature">
                        <h3>Export de Données</h3>
                        <p>Exportez facilement l'historique des conversations pour analyse dans vos outils préférés.</p>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>&copy; 2023 Chatbot IA Creator. Tous droits réservés.</p>
            </div>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_login_page(self):
        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Connexion - Chatbot IA</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .login-container {
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    width: 350px;
                }
                h1 {
                    text-align: center;
                    color: #0084ff;
                    margin-top: 0;
                }
                form {
                    display: flex;
                    flex-direction: column;
                }
                label {
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input {
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                button {
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 12px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                }
                .register-link {
                    text-align: center;
                    margin-top: 20px;
                }
                .register-link a {
                    color: #0084ff;
                    text-decoration: none;
                }
                .error {
                    color: red;
                    margin-bottom: 15px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h1>Connexion</h1>
                <div id="error-message" class="error" style="display: none;"></div>
                <form id="login-form" method="post" action="/login">
                    <label for="username">Nom d'utilisateur</label>
                    <input type="text" id="username" name="username" required>
                    
                    <label for="password">Mot de passe</label>
                    <input type="password" id="password" name="password" required>
                    
                    <button type="submit">Se connecter</button>
                </form>
                <div class="register-link">
                    <p>Pas encore de compte? <a href="/register">Inscrivez-vous</a></p>
                </div>
            </div>
            
            <script>
                // Afficher les messages d'erreur éventuels
                const urlParams = new URLSearchParams(window.location.search);
                const errorMessage = urlParams.get('error');
                
                if (errorMessage) {
                    const errorElement = document.getElementById('error-message');
                    errorElement.textContent = decodeURIComponent(errorMessage);
                    errorElement.style.display = 'block';
                }
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_register_page(self):
        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Inscription - Chatbot IA</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .register-container {
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    width: 350px;
                }
                h1 {
                    text-align: center;
                    color: #0084ff;
                    margin-top: 0;
                }
                form {
                    display: flex;
                    flex-direction: column;
                }
                label {
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input {
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                button {
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 12px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                }
                .login-link {
                    text-align: center;
                    margin-top: 20px;
                }
                .login-link a {
                    color: #0084ff;
                    text-decoration: none;
                }
                .error {
                    color: red;
                    margin-bottom: 15px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="register-container">
                <h1>Inscription</h1>
                <div id="error-message" class="error" style="display: none;"></div>
                <form id="register-form" method="post" action="/register">
                    <label for="username">Nom d'utilisateur</label>
                    <input type="text" id="username" name="username" required>
                    
                    <label for="password">Mot de passe</label>
                    <input type="password" id="password" name="password" required>
                    
                    <label for="confirm-password">Confirmer le mot de passe</label>
                    <input type="password" id="confirm-password" name="confirm-password" required>
                    
                    <button type="submit">S'inscrire</button>
                </form>
                <div class="login-link">
                    <p>Déjà un compte? <a href="/login">Connectez-vous</a></p>
                </div>
            </div>
            
            <script>
                // Vérifier que les mots de passe correspondent
                document.getElementById('register-form').addEventListener('submit', function(e) {
                    const password = document.getElementById('password').value;
                    const confirmPassword = document.getElementById('confirm-password').value;
                    
                    if (password !== confirmPassword) {
                        e.preventDefault();
                        const errorElement = document.getElementById('error-message');
                        errorElement.textContent = 'Les mots de passe ne correspondent pas.';
                        errorElement.style.display = 'block';
                    }
                });
                
                // Afficher les messages d'erreur éventuels
                const urlParams = new URLSearchParams(window.location.search);
                const errorMessage = urlParams.get('error');
                
                if (errorMessage) {
                    const errorElement = document.getElementById('error-message');
                    errorElement.textContent = decodeURIComponent(errorMessage);
                    errorElement.style.display = 'block';
                }
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_dashboard(self, user_id):
        user_chatbots = get_user_chatbots(user_id)
        
        # Format the chatbots for display
        chatbots_html = ""
        for chatbot_id, chatbot in user_chatbots.items():
            knowledge_count = len(chatbot["knowledge_sources"])
            
            # Calculer le nombre de conversations
            conversation_count = len(conversations.get(chatbot_id, []))
            
            chatbots_html += f"""
            <div class="chatbot-card">
                <div class="chatbot-icon" style="background-color: {chatbot['appearance']['primary_color']};">
                    <img src="/uploads/{chatbot['appearance']['logo']}" onerror="this.src='/static/default_logo.png'; this.onerror=null;" alt="Logo">
                </div>
                <div class="chatbot-info">
                    <h3>{chatbot['name']}</h3>
                    <p>{chatbot['description']}</p>
                    <div class="chatbot-stats">
                        <div class="stat">
                            <span class="stat-value">{knowledge_count}</span>
                            <span class="stat-label">Sources</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value">{conversation_count}</span>
                            <span class="stat-label">Conversations</span>
                        </div>
                    </div>
                </div>
                <div class="chatbot-actions">
                    <a href="/chatbot/{chatbot_id}" class="btn-primary">Gérer</a>
                    <a href="/share/{chatbot['share_token']}" class="btn-secondary" target="_blank">Aperçu</a>
                </div>
            </div>
            """
        
        if not chatbots_html:
            chatbots_html = """
            <div class="empty-state">
                <img src="/static/bot-icon.png" alt="Robot" style="width: 120px; margin-bottom: 20px;">
                <h3>Vous n'avez pas encore créé de chatbot</h3>
                <p>Créez votre premier chatbot pour commencer à offrir une assistance automatisée à vos visiteurs.</p>
                <a href="/create-chatbot" class="btn-primary">Créer votre premier chatbot</a>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tableau de bord - Chatbot IA</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .navbar {{
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .navbar h1 {{
                    margin: 0;
                    font-size: 1.5rem;
                }}
                .navbar-actions a {{
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .dashboard-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 30px;
                }}
                .btn-primary {{
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    font-weight: 500;
                    transition: background-color 0.2s;
                }}
                .btn-primary:hover {{
                    background-color: #0073e6;
                }}
                .btn-secondary {{
                    background-color: #f0f0f0;
                    color: #333;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    font-weight: 500;
                    transition: background-color 0.2s;
                }}
                .btn-secondary:hover {{
                    background-color: #e0e0e0;
                }}
                .chatbots-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                    gap: 20px;
                }}
                .chatbot-card {{
                    background-color: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    display: flex;
                    flex-direction: column;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .chatbot-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
                }}
                .chatbot-icon {{
                    width: 100%;
                    height: 100px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .chatbot-icon img {{
                    width: 60px;
                    height: 60px;
                    object-fit: contain;
                }}
                .chatbot-info {{
                    padding: 20px;
                    flex-grow: 1;
                }}
                .chatbot-info h3 {{
                    margin-top: 0;
                    color: #333;
                }}
                .chatbot-info p {{
                    color: #666;
                    margin-bottom: 15px;
                }}
                .chatbot-stats {{
                    display: flex;
                    gap: 20px;
                    margin-top: 15px;
                }}
                .stat {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .stat-value {{
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: #0084ff;
                }}
                .stat-label {{
                    font-size: 0.8rem;
                    color: #666;
                }}
                .chatbot-actions {{
                    padding: 15px 20px;
                    display: flex;
                    gap: 10px;
                    border-top: 1px solid #eee;
                }}
                .empty-state {{
                    text-align: center;
                    padding: 60px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .empty-state h3 {{
                    margin-top: 0;
                    color: #333;
                }}
                .empty-state p {{
                    color: #666;
                    margin-bottom: 25px;
                    max-width: 500px;
                }}
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <div class="dashboard-header">
                    <h2>Mes Chatbots</h2>
                    <a href="/create-chatbot" class="btn-primary">Créer un chatbot</a>
                </div>
                
                <div class="chatbots-grid">
                    {chatbots_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_create_chatbot_page(self, user_id):
        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Créer un Chatbot - Chatbot IA</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .navbar {
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .navbar h1 {
                    margin: 0;
                    font-size: 1.5rem;
                }
                .navbar-actions a {
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .create-form {
                    background-color: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h2 {
                    margin-top: 0;
                    color: #0084ff;
                    margin-bottom: 20px;
                }
                form {
                    display: flex;
                    flex-direction: column;
                }
                .form-group {
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                }
                input, textarea {
                    width: 100%;
                    padding: 12px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                    box-sizing: border-box;
                    font-family: inherit;
                }
                textarea {
                    min-height: 120px;
                    resize: vertical;
                }
                button {
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 12px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    font-weight: 500;
                    transition: background-color 0.2s;
                }
                button:hover {
                    background-color: #0073e6;
                }
                .error {
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-weight: 500;
                }
                .form-info {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    color: #666;
                }
                .back-link {
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #0084ff;
                    text-decoration: none;
                }
                .back-link:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <a href="/dashboard" class="back-link">← Retour au tableau de bord</a>
                
                <div class="create-form">
                    <h2>Créer un nouveau Chatbot</h2>
                    
                    <div class="form-info">
                        Commencez par définir les informations de base de votre chatbot. Vous pourrez ensuite ajouter des connaissances et personnaliser son apparence.
                    </div>
                    
                    <div id="error-message" class="error" style="display: none;"></div>
                    
                    <form id="create-form" method="post" action="/create-chatbot">
                        <div class="form-group">
                            <label for="name">Nom du Chatbot</label>
                            <input type="text" id="name" name="name" required placeholder="Ex: Assistant Commercial">
                        </div>
                        
                        <div class="form-group">
                            <label for="description">Description</label>
                            <textarea id="description" name="description" required placeholder="Décrivez le rôle et la fonction de votre chatbot..."></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="welcome-message">Message de Bienvenue</label>
                            <textarea id="welcome-message" name="welcome_message" placeholder="Message qui s'affichera au début de chaque conversation">Bonjour ! Je suis votre assistant. Comment puis-je vous aider aujourd'hui ?</textarea>
                        </div>
                        
                        <button type="submit">Créer mon Chatbot</button>
                    </form>
                </div>
            </div>
            
            <script>
                // Afficher les messages d'erreur éventuels
                const urlParams = new URLSearchParams(window.location.search);
                const errorMessage = urlParams.get('error');
                
                if (errorMessage) {
                    const errorElement = document.getElementById('error-message');
                    errorElement.textContent = decodeURIComponent(errorMessage);
                    errorElement.style.display = 'block';
                }
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_chatbot_page(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        chatbot = chatbots[chatbot_id]
        
        # Préparer la liste des sources de connaissances
        knowledge_html = ""
        for source in chatbot["knowledge_sources"]:
            knowledge_html += f"""
            <div class="knowledge-item">
                <div class="knowledge-info">
                    <h4>Site Web: {source['url']}</h4>
                    <p>Pages: {source['pages_count']} | Chunks: {source['chunks_count']}</p>
                    <p>Ajouté le: {source['added_at'].split('T')[0]}</p>
                </div>
            </div>
            """
        
        if not knowledge_html:
            knowledge_html = """
            <div class="empty-state">
                <p>Aucune source de connaissances n'a été ajoutée à ce chatbot.</p>
            </div>
            """
        
        # Calculer quelques statistiques de base
        conversation_count = len(conversations.get(chatbot_id, []))
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{chatbot['name']} - Gestion du Chatbot</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .navbar {{
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .navbar h1 {{
                    margin: 0;
                    font-size: 1.5rem;
                }}
                .navbar-actions a {{
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .chatbot-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .back-link {{
                    color: #0084ff;
                    text-decoration: none;
                    margin-bottom: 20px;
                    display: inline-block;
                }}
                .chatbot-header-actions {{
                    display: flex;
                    gap: 10px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .card-stats {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    flex: 1;
                    min-width: 200px;
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2rem;
                    font-weight: bold;
                    color: #0084ff;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    color: #666;
                }}
                .links-section {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .link-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .link-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
                }}
                .link-card a {{
                    color: #333;
                    text-decoration: none;
                    display: block;
                }}
                .link-card i {{
                    font-size: 2rem;
                    color: #0084ff;
                    margin-bottom: 10px;
                }}
                .link-card h3 {{
                    margin-top: 0;
                    margin-bottom: 5px;
                }}
                .link-card p {{
                    color: #666;
                    margin-top: 5px;
                }}
                .card h3 {{
                    color: #0084ff;
                    margin-top: 0;
                }}
                .btn-primary {{
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    font-weight: 500;
                }}
                .btn-secondary {{
                    background-color: #f0f0f0;
                    color: #333;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                }}
                .knowledge-item {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                }}
                .empty-state {{
                    text-align: center;
                    padding: 30px;
                    border: 1px dashed #ddd;
                    border-radius: 5px;
                }}
                .add-knowledge-form {{
                    margin-top: 20px;
                }}
                .add-knowledge-form label {{
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                }}
                .add-knowledge-form input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    box-sizing: border-box;
                }}
                .loader {{
                    border: 4px solid #f3f3f3;
                    border-radius: 50%;
                    border-top: 4px solid #0084ff;
                    width: 20px;
                    height: 20px;
                    animation: spin 1s linear infinite;
                    display: inline-block;
                    margin-right: 10px;
                    vertical-align: middle;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .error {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-weight: 500;
                }}
                /* Icônes pour les raccourcis */
                .icon {{
                    font-size: 2rem;
                    margin-bottom: 10px;
                }}
                .icon-knowledge::before {{ content: "📚"; }}
                .icon-appearance::before {{ content: "🎨"; }}
                .icon-conversations::before {{ content: "💬"; }}
                .icon-metrics::before {{ content: "📊"; }}
                .icon-share::before {{ content: "🔗"; }}
                .icon-widget::before {{ content: "🧩"; }}
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <a href="/dashboard" class="back-link">← Retour au tableau de bord</a>
                
                <div class="chatbot-header">
                    <h2>{chatbot['name']}</h2>
                    <div class="chatbot-header-actions">
                        <a href="/share/{chatbot['share_token']}" class="btn-secondary" target="_blank">Aperçu</a>
                    </div>
                </div>
                
                <div class="card-stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(chatbot["knowledge_sources"])}</div>
                        <div class="stat-label">Sources de connaissances</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{conversation_count}</div>
                        <div class="stat-label">Conversations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{chatbot['created_at'].split('T')[0]}</div>
                        <div class="stat-label">Date de création</div>
                    </div>
                </div>
                
                <div class="links-section">
                    <div class="link-card">
                        <a href="#knowledge-section">
                            <div class="icon icon-knowledge"></div>
                            <h3>Connaissances</h3>
                            <p>Gérer les sources d'informations</p>
                        </a>
                    </div>
                    <div class="link-card">
                        <a href="/appearance/{chatbot_id}">
                            <div class="icon icon-appearance"></div>
                            <h3>Apparence</h3>
                            <p>Personnaliser le design</p>
                        </a>
                    </div>
                    <div class="link-card">
                        <a href="/conversations/{chatbot_id}">
                            <div class="icon icon-conversations"></div>
                            <h3>Conversations</h3>
                            <p>Historique des échanges</p>
                        </a>
                    </div>
                    <div class="link-card">
                        <a href="/metrics/{chatbot_id}">
                            <div class="icon icon-metrics"></div>
                            <h3>Métriques</h3>
                            <p>Analyses et statistiques</p>
                        </a>
                    </div>
                    <div class="link-card">
                        <a href="#share-section">
                            <div class="icon icon-share"></div>
                            <h3>Partage</h3>
                            <p>Obtenir le lien de partage</p>
                        </a>
                    </div>
                    <div class="link-card">
                        <a href="#widget-section">
                            <div class="icon icon-widget"></div>
                            <h3>Widget</h3>
                            <p>Code d'intégration</p>
                        </a>
                    </div>
                </div>
                
                <div class="card" id="knowledge-section">
                    <h3>Sources de Connaissances</h3>
                    <div class="knowledge-list">
                        {knowledge_html}
                    </div>
                    
                    <div class="add-knowledge-form">
                        <h4>Ajouter une nouvelle source</h4>
                        <div id="knowledge-error" class="error" style="display: none;"></div>
                        <form id="discover-urls-form">
                            <label for="website-url">URL du site web à explorer</label>
                            <input type="url" id="website-url" name="website-url" placeholder="https://exemple.com" required>
                            
                            <button type="submit" id="discover-urls-btn" class="btn-primary">Découvrir les pages</button>
                        </form>
                        
                        <div id="urls-selection" style="display: none; margin-top: 20px;">
                            <h4>Sélectionnez les pages à extraire</h4>
                            <p>Cochez les pages dont vous souhaitez extraire le contenu pour alimenter votre chatbot.</p>
                            
                            <div id="url-list" style="margin-top: 15px; max-height: 300px; overflow-y: auto;">
                                <!-- Les URLs découvertes seront ajoutées ici -->
                            </div>
                            
                            <div style="margin-top: 15px; display: flex; gap: 10px;">
                                <button id="select-all" class="btn-secondary">Tout sélectionner</button>
                                <button id="deselect-all" class="btn-secondary">Tout désélectionner</button>
                            </div>
                            
                            <button id="extract-content-btn" class="btn-primary" style="margin-top: 15px;">Extraire le contenu sélectionné</button>
                        </div>
                    </div>
                </div>
                
                <div class="card" id="share-section">
                    <h3>Partager votre Chatbot</h3>
                    <p>Partagez ce lien pour permettre à n'importe qui d'utiliser votre chatbot:</p>
                    
                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; word-break: break-all;">
                        http://{HOST}:{PORT}/share/{chatbot['share_token']}
                    </div>
                    
                    <button id="copy-share-link" class="btn-primary">Copier le lien</button>
                </div>
                
                <div class="card" id="widget-section">
                    <h3>Widget pour votre site web</h3>
                    <p>Copiez ce code et collez-le dans votre site web pour intégrer le chatbot:</p>
                    
                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; margin-top: 15px; max-height: 300px; overflow-y: auto;">
                        <pre><code id="widget-code">{generate_widget_code(chatbot_id).replace('<', '&lt;').replace('>', '&gt;')}</code></pre>
                    </div>
                    
                    <button id="copy-widget-code" class="btn-primary" style="margin-top: 15px;">Copier le code</button>
                </div>
            </div>
            
            <script>
                // Copier le lien de partage
                document.getElementById('copy-share-link').addEventListener('click', () => {{
                    const shareLink = "http://{HOST}:{PORT}/share/{chatbot['share_token']}";
                    navigator.clipboard.writeText(shareLink).then(() => {{
                        alert('Lien copié !');
                    }});
                }});
                
                // Copier le code du widget
                document.getElementById('copy-widget-code').addEventListener('click', () => {{
                    const widgetCode = document.getElementById('widget-code').textContent;
                    navigator.clipboard.writeText(widgetCode).then(() => {{
                        alert('Code copié !');
                    }});
                }});
                
                // Découverte des URLs
                document.getElementById('discover-urls-form').addEventListener('submit', function(e) {{
                    e.preventDefault();
                    
                    const websiteUrl = document.getElementById('website-url').value;
                    const discoverUrlsBtn = document.getElementById('discover-urls-btn');
                    const knowledgeError = document.getElementById('knowledge-error');
                    const urlsSelection = document.getElementById('urls-selection');
                    const urlList = document.getElementById('url-list');
                    
                    // Afficher le loader
                    const originalBtnText = discoverUrlsBtn.innerHTML;
                    discoverUrlsBtn.innerHTML = '<span class="loader"></span> Découverte en cours...';
                    discoverUrlsBtn.disabled = true;
                    knowledgeError.style.display = 'none';
                    
                    fetch('/api/discover-urls', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            chatbot_id: '{chatbot_id}',
                            website_url: websiteUrl
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        discoverUrlsBtn.innerHTML = originalBtnText;
                        discoverUrlsBtn.disabled = false;
                        
                        if (data.success) {{
                            // Afficher la section de sélection d'URLs
                            urlsSelection.style.display = 'block';
                            
                            // Remplir la liste des URLs
                            urlList.innerHTML = '';
                            data.urls.forEach(url => {{
                                const urlItem = document.createElement('div');
                                urlItem.style.margin = '10px 0';
                                urlItem.innerHTML = `
                                    <label style="display: flex; align-items: flex-start;">
                                        <input type="checkbox" name="selected_urls" value="\${url}" checked style="margin-top: 5px;">
                                        <div style="margin-left: 10px;">
                                            <div style="font-weight: 500;">\${url}</div>
                                        </div>
                                    </label>
                                `;
                                urlList.appendChild(urlItem);
                            }});
                        }} else {{
                            knowledgeError.textContent = "Erreur lors de la découverte des URLs.";
                            knowledgeError.style.display = 'block';
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        discoverUrlsBtn.innerHTML = originalBtnText;
                        discoverUrlsBtn.disabled = false;
                        knowledgeError.textContent = 'Une erreur est survenue. Veuillez réessayer.';
                        knowledgeError.style.display = 'block';
                    }});
                }});
                
                // Gestion de la sélection des URLs
                document.getElementById('select-all').addEventListener('click', function() {{
                    const checkboxes = document.querySelectorAll('input[name="selected_urls"]');
                    checkboxes.forEach(checkbox => checkbox.checked = true);
                }});
                
                document.getElementById('deselect-all').addEventListener('click', function() {{
                    const checkboxes = document.querySelectorAll('input[name="selected_urls"]');
                    checkboxes.forEach(checkbox => checkbox.checked = false);
                }});
                
                // Extraction du contenu
                document.getElementById('extract-content-btn').addEventListener('click', function() {{
                    const selectedUrls = Array.from(document.querySelectorAll('input[name="selected_urls"]:checked')).map(cb => cb.value);
                    const websiteUrl = document.getElementById('website-url').value;
                    const extractBtn = this;
                    const knowledgeError = document.getElementById('knowledge-error');
                    
                    if (selectedUrls.length === 0) {{
                        knowledgeError.textContent = "Veuillez sélectionner au moins une URL.";
                        knowledgeError.style.display = 'block';
                        return;
                    }}
                    
                    // Afficher le loader
                    const originalBtnText = extractBtn.innerHTML;
                    extractBtn.innerHTML = '<span class="loader"></span> Extraction en cours...';
                    extractBtn.disabled = true;
                    knowledgeError.style.display = 'none';
                    
                    fetch('/api/extract-content', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            chatbot_id: '{chatbot_id}',
                            website_url: websiteUrl,
                            selected_urls: selectedUrls
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        extractBtn.innerHTML = originalBtnText;
                        extractBtn.disabled = false;
                        
                        if (data.success) {{
                            alert('Contenu extrait avec succès ! La page va être rechargée.');
                            window.location.reload();
                        }} else {{
                            knowledgeError.textContent = data.message || "Erreur lors de l'extraction du contenu.";
                            knowledgeError.style.display = 'block';
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        extractBtn.innerHTML = originalBtnText;
                        extractBtn.disabled = false;
                        knowledgeError.textContent = 'Une erreur est survenue. Veuillez réessayer.';
                        knowledgeError.style.display = 'block';
                    }});
                }});
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_conversations_page(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        chatbot = chatbots[chatbot_id]
        chatbot_conversations = conversations.get(chatbot_id, [])
        
        # Organiser les conversations par date
        conversations_by_date = {}
        for conv in chatbot_conversations:
            date = datetime.fromisoformat(conv["timestamp"]).strftime("%Y-%m-%d")
            if date not in conversations_by_date:
                conversations_by_date[date] = []
            conversations_by_date[date].append(conv)
        
        # Trier les dates en ordre décroissant
        sorted_dates = sorted(conversations_by_date.keys(), reverse=True)
        
        # Générer le HTML pour les conversations
        conversations_html = ""
        for date in sorted_dates:
            date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%d %B %Y")
            conversations_html += f'<h3 class="date-header">{date_formatted}</h3>'
            
            for conv in conversations_by_date[date]:
                time = datetime.fromisoformat(conv["timestamp"]).strftime("%H:%M")
                sentiment_class = f"sentiment-{conv['metadata']['sentiment']}"
                query_type = conv['metadata']['query_type'].replace('_', ' ').title()
                
                conversations_html += f"""
                <div class="conversation-item">
                    <div class="conversation-header">
                        <span class="conversation-time">{time}</span>
                        <div class="conversation-metadata">
                            <span class="conversation-sentiment {sentiment_class}">{conv['metadata']['sentiment'].title()}</span>
                            <span class="conversation-type">{query_type}</span>
                        </div>
                    </div>
                    <div class="conversation-content">
                        <div class="user-message">
                            <div class="message-label">Utilisateur</div>
                            <div class="message-text">{conv['user_input']}</div>
                        </div>
                        <div class="bot-message">
                            <div class="message-label">Chatbot</div>
                            <div class="message-text">{conv['ai_response']}</div>
                        </div>
                    </div>
                </div>
                """
        
        if not conversations_html:
            conversations_html = """
            <div class="empty-state">
                <img src="/static/chat-icon.png" alt="Chat" style="width: 100px; margin-bottom: 20px;">
                <h3>Aucune conversation pour le moment</h3>
                <p>Les conversations apparaîtront ici dès que des utilisateurs commenceront à discuter avec votre chatbot.</p>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Conversations - {chatbot['name']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .navbar {{
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .navbar h1 {{
                    margin: 0;
                    font-size: 1.5rem;
                }}
                .navbar-actions a {{
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .back-link {{
                    color: #0084ff;
                    text-decoration: none;
                    margin-bottom: 20px;
                    display: inline-block;
                }}
                .page-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .btn-primary {{
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    font-weight: 500;
                }}
                .btn-export {{
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                    margin-left: 10px;
                }}
                .date-header {{
                    color: #666;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                .conversation-item {{
                    border: 1px solid #eee;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    overflow: hidden;
                }}
                .conversation-header {{
                    padding: 10px 15px;
                    background-color: #f9f9f9;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .conversation-time {{
                    font-weight: 500;
                    color: #666;
                }}
                .conversation-metadata {{
                    display: flex;
                    gap: 10px;
                }}
                .conversation-sentiment {{
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    font-weight: 500;
                }}
                .sentiment-positive {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .sentiment-negative {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
                .sentiment-neutral {{
                    background-color: #e2e3e5;
                    color: #383d41;
                }}
                .conversation-type {{
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    background-color: #e2f0ff;
                    color: #0084ff;
                }}
                .conversation-content {{
                    padding: 15px;
                }}
                .user-message, .bot-message {{
                    margin-bottom: 15px;
                }}
                .message-label {{
                    font-weight: 500;
                    margin-bottom: 5px;
                    color: #666;
                }}
                .message-text {{
                    background-color: #f5f5f5;
                    padding: 10px 15px;
                    border-radius: 8px;
                    white-space: pre-wrap;
                }}
                .user-message .message-label {{
                    color: #0084ff;
                }}
                .bot-message .message-label {{
                    color: #666;
                }}
                .empty-state {{
                    text-align: center;
                    padding: 50px 20px;
                }}
                .empty-state h3 {{
                    color: #333;
                    margin-top: 0;
                }}
                .empty-state p {{
                    color: #666;
                    max-width: 500px;
                    margin: 10px auto;
                }}
                .export-options {{
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <a href="/chatbot/{chatbot_id}" class="back-link">← Retour au chatbot</a>
                
                <div class="page-header">
                    <h2>Historique des Conversations</h2>
                    <div>
                        <a href="/export/{chatbot_id}/csv" class="btn-primary btn-export" target="_blank">Exporter en CSV</a>
                        <a href="/export/{chatbot_id}/json" class="btn-primary btn-export" target="_blank">Exporter en JSON</a>
                    </div>
                </div>
                
                <div class="card">
                    {conversations_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_metrics_page(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        chatbot = chatbots[chatbot_id]
        metrics = get_chatbot_metrics(chatbot_id)
        charts = generate_metrics_charts(chatbot_id)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Métriques - {chatbot['name']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .navbar {{
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .navbar h1 {{
                    margin: 0;
                    font-size: 1.5rem;
                }}
                .navbar-actions a {{
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .back-link {{
                    color: #0084ff;
                    text-decoration: none;
                    margin-bottom: 20px;
                    display: inline-block;
                }}
                .metrics-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .metrics-overview {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #0084ff;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    color: #666;
                }}
                .chart-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .chart-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chart-title {{
                    color: #333;
                    margin-top: 0;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .chart-img {{
                    max-width: 100%;
                    height: auto;
                    margin: 0 auto;
                    display: block;
                }}
                .empty-state {{
                    text-align: center;
                    padding: 50px 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .empty-state h3 {{
                    color: #333;
                    margin-top: 0;
                }}
                .empty-state p {{
                    color: #666;
                    max-width: 500px;
                    margin: 10px auto;
                }}
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <a href="/chatbot/{chatbot_id}" class="back-link">← Retour au chatbot</a>
                
                <div class="metrics-header">
                    <h2>Métriques et Analyses</h2>
                </div>
        """
        
        # Afficher les métriques générales
        html += """
                <div class="metrics-overview">
        """
        
        # Conversations totales
        html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['total_conversations']}</div>
                        <div class="metric-label">Conversations Totales</div>
                    </div>
        """
        
        # Heure la plus active
        busiest_hour = "N/A" if metrics['busiest_hour'] is None else f"{metrics['busiest_hour']}h00"
        html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{busiest_hour}</div>
                        <div class="metric-label">Heure la Plus Active</div>
                    </div>
        """
        
        # Longueur moyenne des réponses
        html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['avg_response_length']}</div>
                        <div class="metric-label">Caractères/Réponse</div>
                    </div>
        """
        
        # Distribution des sentiments (simplifié)
        positive = metrics['sentiment_distribution'].get('positive', 0)
        neutral = metrics['sentiment_distribution'].get('neutral', 0)
        negative = metrics['sentiment_distribution'].get('negative', 0)
        total_with_sentiment = positive + neutral + negative
        
        if total_with_sentiment > 0:
            positive_pct = int(positive / total_with_sentiment * 100)
            html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{positive_pct}%</div>
                        <div class="metric-label">Sentiments Positifs</div>
                    </div>
            """
        
        html += """
                </div>
        """
        
        # Afficher les graphiques s'il y a des données
        if metrics['total_conversations'] > 0:
            html += """
                <div class="chart-container">
            """
            
            # Graphique des sentiments
            if 'sentiment_distribution' in charts:
                html += f"""
                    <div class="chart-card">
                        <h3 class="chart-title">Distribution des Sentiments</h3>
                        <img src="data:image/png;base64,{charts['sentiment_distribution']}" class="chart-img" alt="Distribution des Sentiments">
                    </div>
                """
            
            # Graphique des types de requêtes
            if 'query_types' in charts:
                html += f"""
                    <div class="chart-card">
                        <h3 class="chart-title">Types de Requêtes</h3>
                        <img src="data:image/png;base64,{charts['query_types']}" class="chart-img" alt="Types de Requêtes">
                    </div>
                """
            
            # Graphique des conversations dans le temps
            if 'conversations_over_time' in charts:
                html += f"""
                    <div class="chart-card">
                        <h3 class="chart-title">Conversations par Jour</h3>
                        <img src="data:image/png;base64,{charts['conversations_over_time']}" class="chart-img" alt="Conversations par Jour">
                    </div>
                """
            
            # Graphique par heure
            if 'hour_distribution' in charts:
                html += f"""
                    <div class="chart-card">
                        <h3 class="chart-title">Distribution par Heure</h3>
                        <img src="data:image/png;base64,{charts['hour_distribution']}" class="chart-img" alt="Distribution par Heure">
                    </div>
                """
            
            html += """
                </div>
            """
        else:
            html += """
                <div class="empty-state">
                    <h3>Aucune donnée disponible</h3>
                    <p>Les métriques et analyses seront disponibles dès que des utilisateurs commenceront à interagir avec votre chatbot.</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    def _serve_appearance_page(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        chatbot = chatbots[chatbot_id]
        appearance = chatbot["appearance"]
        suggested_messages = chatbot["suggested_messages"]
        
        # Convertir la liste de messages suggérés en texte pour le textarea
        suggested_messages_text = "\n".join(suggested_messages)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Apparence - {chatbot['name']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .navbar {{
                    background-color: #0084ff;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .navbar h1 {{
                    margin: 0;
                    font-size: 1.5rem;
                }}
                .navbar-actions a {{
                    color: white;
                    text-decoration: none;
                    margin-left: 20px;
                    font-weight: 500;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .back-link {{
                    color: #0084ff;
                    text-decoration: none;
                    margin-bottom: 20px;
                    display: inline-block;
                }}
                .page-header {{
                    margin-bottom: 20px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .form-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }}
                .form-group {{
                    margin-bottom: 20px;
                }}
                label {{
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                }}
                input, textarea, select {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-family: inherit;
                    font-size: 16px;
                    box-sizing: border-box;
                }}
                textarea {{
                    min-height: 100px;
                    resize: vertical;
                }}
                .color-preview {{
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    margin-left: 10px;
                    vertical-align: middle;
                }}
                .btn-primary {{
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 500;
                }}
                .preview-section {{
                    margin-top: 30px;
                }}
                .preview-container {{
                    max-width: 350px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .preview-header {{
                    padding: 15px;
                    display: flex;
                    align-items: center;
                }}
                .preview-header-logo {{
                    width: 30px;
                    height: 30px;
                    margin-right: 10px;
                    object-fit: contain;
                }}
                .preview-header-title {{
                    margin: 0;
                    font-size: 1.2rem;
                }}
                .preview-messages {{
                    height: 200px;
                    padding: 15px;
                    overflow-y: auto;
                    background-color: #f9f9f9;
                }}
                .preview-message {{
                    max-width: 80%;
                    padding: 10px;
                    margin-bottom: 10px;
                }}
                .preview-bot-message {{
                    border-radius: 10px;
                    margin-right: auto;
                }}
                .preview-user-message {{
                    border-radius: 10px;
                    margin-left: auto;
                    color: white;
                }}
                .preview-suggested {{
                    padding: 10px 15px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    border-top: 1px solid #eee;
                }}
                .preview-suggested-question {{
                    background-color: #f5f5f5;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                }}
                .preview-input {{
                    display: flex;
                    padding: 10px 15px;
                    border-top: 1px solid #eee;
                }}
                .preview-input-field {{
                    flex-grow: 1;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 20px;
                    font-size: 14px;
                }}
                .preview-input-button {{
                    margin-left: 10px;
                    border: none;
                    border-radius: 20px;
                    padding: 8px 15px;
                    font-size: 14px;
                    color: white;
                }}
                .logo-preview {{
                    width: 100px;
                    height: 100px;
                    object-fit: contain;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin-top: 10px;
                    background-color: #f9f9f9;
                }}
                .upload-note {{
                    font-size: 0.9rem;
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="navbar">
                <h1>Chatbot IA Creator</h1>
                <div class="navbar-actions">
                    <a href="/dashboard">Tableau de bord</a>
                    <a href="/logout">Déconnexion</a>
                </div>
            </div>
            
            <div class="container">
                <a href="/chatbot/{chatbot_id}" class="back-link">← Retour au chatbot</a>
                
                <div class="page-header">
                    <h2>Personnalisation de l'Apparence</h2>
                </div>
                
                <div class="card">
                    <form id="appearance-form" action="/appearance/{chatbot_id}" method="post">
                        <div class="form-grid">
                            <div>
                                <div class="form-group">
                                    <label for="primary-color">Couleur Principale</label>
                                    <div>
                                        <input type="color" id="primary-color" name="primary_color" value="{appearance['primary_color']}">
                                        <input type="text" id="primary-color-text" value="{appearance['primary_color']}" style="width: calc(100% - 50px); margin-left: 10px;">
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <label for="secondary-color">Couleur Secondaire</label>
                                    <div>
                                        <input type="color" id="secondary-color" name="secondary_color" value="{appearance['secondary_color']}">
                                        <input type="text" id="secondary-color-text" value="{appearance['secondary_color']}" style="width: calc(100% - 50px); margin-left: 10px;">
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <label for="bubble-style">Style des Bulles</label>
                                    <select id="bubble-style" name="bubble_style">
                                        <option value="rounded" {appearance['bubble_style'] == 'rounded' and 'selected' or ''}>Arrondi</option>
                                        <option value="square" {appearance['bubble_style'] == 'square' and 'selected' or ''}>Carré</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="font">Police d'Écriture</label>
                                    <select id="font" name="font">
                                        <option value="Arial, sans-serif" {appearance['font'] == 'Arial, sans-serif' and 'selected' or ''}>Arial</option>
                                        <option value="'Segoe UI', sans-serif" {appearance['font'] == "'Segoe UI', sans-serif" and 'selected' or ''}>Segoe UI</option>
                                        <option value="Roboto, sans-serif" {appearance['font'] == 'Roboto, sans-serif' and 'selected' or ''}>Roboto</option>
                                        <option value="'Open Sans', sans-serif" {appearance['font'] == "'Open Sans', sans-serif" and 'selected' or ''}>Open Sans</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div>
                                <div class="form-group">
                                    <label for="title">Titre du Chatbot</label>
                                    <input type="text" id="title" name="title" value="{appearance['title']}">
                                </div>
                                
                                <div class="form-group">
                                    <label for="button-text">Texte du Bouton</label>
                                    <input type="text" id="button-text" name="button_text" value="{appearance['button_text']}">
                                </div>
                                
                                <div class="form-group">
                                    <label for="placeholder-text">Texte de l'Input</label>
                                    <input type="text" id="placeholder-text" name="placeholder_text" value="{appearance['placeholder_text']}">
                                </div>
                                
                                <div class="form-group">
                                    <label for="logo">Logo du Chatbot</label>
                                    <div>
                                        <img src="/uploads/{appearance['logo']}" id="logo-preview" class="logo-preview" onerror="this.src='/static/default_logo.png'; this.onerror=null;" alt="Logo Preview">
                                    </div>
                                    <p class="upload-note">Pour changer le logo, utilisez le formulaire d'upload ci-dessous.</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="suggested-messages">Messages Suggérés (un par ligne)</label>
                            <textarea id="suggested-messages" name="suggested_messages">{suggested_messages_text}</textarea>
                        </div>
                        
                        <button type="submit" class="btn-primary">Enregistrer les Modifications</button>
                    </form>
                    
                    <div class="form-group" style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px;">
                        <label for="logo-upload">Changer le Logo</label>
                        <form id="logo-upload-form" action="/upload-logo/{chatbot_id}" method="post" enctype="multipart/form-data">
                            <input type="file" id="logo-upload" name="logo" accept="image/png,image/jpeg,image/gif,image/svg+xml">
                            <p class="upload-note">Formats acceptés: PNG, JPG, GIF, SVG. Taille recommandée: 100x100 pixels.</p>
                            <button type="submit" class="btn-primary" style="margin-top: 10px;">Uploader le Logo</button>
                        </form>
                    </div>
                </div>
                
                <div class="preview-section">
                    <h3>Aperçu du Chatbot</h3>
                    <div class="preview-container">
                        <div class="preview-header" id="preview-header">
                            <img id="preview-logo" src="/uploads/{appearance['logo']}" class="preview-header-logo" onerror="this.src='/static/default_logo.png'; this.onerror=null;" alt="Logo">
                            <h3 id="preview-title" class="preview-header-title">{appearance['title']}</h3>
                        </div>
                        <div class="preview-messages">
                            <div id="preview-bot-message" class="preview-message preview-bot-message">{chatbot['welcome_message']}</div>
                            <div id="preview-user-message" class="preview-message preview-user-message">Bonjour, j'ai une question.</div>
                        </div>
                        <div class="preview-suggested">
                            {' '.join([f'<div class="preview-suggested-question">{msg}</div>' for msg in suggested_messages[:3]])}
                        </div>
                        <div class="preview-input">
                            <input type="text" id="preview-input-field" class="preview-input-field" placeholder="{appearance['placeholder_text']}">
                            <button id="preview-input-button" class="preview-input-button">{appearance['button_text']}</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Mise à jour de l'aperçu en temps réel
                const primaryColor = document.getElementById('primary-color');
                const primaryColorText = document.getElementById('primary-color-text');
                const secondaryColor = document.getElementById('secondary-color');
                const secondaryColorText = document.getElementById('secondary-color-text');
                const bubbleStyle = document.getElementById('bubble-style');
                const font = document.getElementById('font');
                const title = document.getElementById('title');
                const buttonText = document.getElementById('button-text');
                const placeholderText = document.getElementById('placeholder-text');
                
                // Éléments de l'aperçu
                const previewHeader = document.getElementById('preview-header');
                const previewTitle = document.getElementById('preview-title');
                const previewBotMessage = document.getElementById('preview-bot-message');
                const previewUserMessage = document.getElementById('preview-user-message');
                const previewInputField = document.getElementById('preview-input-field');
                const previewInputButton = document.getElementById('preview-input-button');
                
                // Synchroniser les inputs de couleur texte et visuel
                primaryColor.addEventListener('input', function() {{
                    primaryColorText.value = this.value;
                    updatePreview();
                }});
                
                primaryColorText.addEventListener('input', function() {{
                    if (/^#[0-9A-F]{{6}}$/i.test(this.value)) {{
                        primaryColor.value = this.value;
                        updatePreview();
                    }}
                }});
                
                secondaryColor.addEventListener('input', function() {{
                    secondaryColorText.value = this.value;
                    updatePreview();
                }});
                
                secondaryColorText.addEventListener('input', function() {{
                    if (/^#[0-9A-F]{{6}}$/i.test(this.value)) {{
                        secondaryColor.value = this.value;
                        updatePreview();
                    }}
                }});
                
                // Mettre à jour l'aperçu lors des changements
                [bubbleStyle, font, title, buttonText, placeholderText].forEach(element => {{
                    element.addEventListener('input', updatePreview);
                }});
                
                function updatePreview() {{
                    // Mettre à jour les couleurs
                    previewHeader.style.backgroundColor = primaryColor.value;
                    previewBotMessage.style.backgroundColor = secondaryColor.value;
                    previewUserMessage.style.backgroundColor = primaryColor.value;
                    previewInputButton.style.backgroundColor = primaryColor.value;
                    
                    // Mettre à jour le style des bulles
                    const borderRadius = bubbleStyle.value === 'rounded' ? '10px' : '0';
                    previewBotMessage.style.borderRadius = borderRadius;
                    previewUserMessage.style.borderRadius = borderRadius;
                    
                    // Mettre à jour la police
                    document.body.style.fontFamily = font.value;
                    
                    // Mettre à jour les textes
                    previewTitle.textContent = title.value;
                    previewInputButton.textContent = buttonText.value;
                    previewInputField.placeholder = placeholderText.value;
                }}
                
                // Mise à jour initiale de l'aperçu
                updatePreview();
                
                // Preview du logo uploadé
                const logoUpload = document.getElementById('logo-upload');
                const logoPreview = document.getElementById('logo-preview');
                const previewLogo = document.getElementById('preview-logo');
                
                logoUpload.addEventListener('change', function() {{
                    const file = this.files[0];
                    if (file) {{
                        const reader = new FileReader();
                        reader.onload = function(e) {{
                            logoPreview.src = e.target.result;
                            previewLogo.src = e.target.result;
                        }};
                        reader.readAsDataURL(file);
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        self._set_headers()
        self.wfile.write(html.encode())
    
    # Handlers de formulaires
    def _handle_register(self):
        post_data = self._get_post_data()
        
        username = post_data.get("username", [""])[0]
        password = post_data.get("password", [""])[0]
        
        if not username or not password:
            self._send_redirect("/register?error=Tous%20les%20champs%20sont%20obligatoires")
            return
        
        success, message = register_user(username, password)
        
        if success:
            self._send_redirect("/login")
        else:
            self._send_redirect(f"/register?error={message}")
    
    def _handle_login(self):
        post_data = self._get_post_data()
        
        username = post_data.get("username", [""])[0]
        password = post_data.get("password", [""])[0]
        
        if not username or not password:
            self._send_redirect("/login?error=Tous%20les%20champs%20sont%20obligatoires")
            return
        
        success, message_or_session_id = login_user(username, password)
        
        if success:
            self.send_response(302)
            cookie = SimpleCookie()
            cookie["session_id"] = message_or_session_id
            cookie["session_id"]["path"] = "/"
            cookie["session_id"]["max-age"] = 86400  # 24 heures
            self.send_header("Set-Cookie", cookie["session_id"].OutputString())
            self.send_header("Location", "/dashboard")
            self.end_headers()
        else:
            self._send_redirect(f"/login?error={message_or_session_id}")
    
    def _handle_logout(self, session_id):
        if session_id in sessions:
            del sessions[session_id]
        
        self.send_response(302)
        cookie = SimpleCookie()
        cookie["session_id"] = ""
        cookie["session_id"]["path"] = "/"
        cookie["session_id"]["max-age"] = 0
        self.send_header("Set-Cookie", cookie["session_id"].OutputString())
        self.send_header("Location", "/")
        self.end_headers()
    
    def _handle_create_chatbot(self, user_id):
        post_data = self._get_post_data()
        
        name = post_data.get("name", [""])[0]
        description = post_data.get("description", [""])[0]
        welcome_message = post_data.get("welcome_message", ["Bonjour ! Je suis votre assistant. Comment puis-je vous aider aujourd'hui ?"])[0]
        
        if not name or not description:
            self._send_redirect("/create-chatbot?error=Tous%20les%20champs%20sont%20obligatoires")
            return
        
        chatbot = create_chatbot(user_id, name, description, welcome_message)
        
        self._send_redirect(f"/chatbot/{chatbot['id']}")
    
    def _handle_chatbot_action(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        # Traiter les différentes actions (aucune action spécifique à traiter pour l'instant)
        self._send_redirect(f"/chatbot/{chatbot_id}")
    
    def _handle_appearance_update(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        post_data = self._get_post_data()
        
        # Extraire les données du formulaire
        primary_color = post_data.get("primary_color", [chatbots[chatbot_id]["appearance"]["primary_color"]])[0]
        secondary_color = post_data.get("secondary_color", [chatbots[chatbot_id]["appearance"]["secondary_color"]])[0]
        bubble_style = post_data.get("bubble_style", [chatbots[chatbot_id]["appearance"]["bubble_style"]])[0]
        font = post_data.get("font", [chatbots[chatbot_id]["appearance"]["font"]])[0]
        title = post_data.get("title", [chatbots[chatbot_id]["appearance"]["title"]])[0]
        button_text = post_data.get("button_text", [chatbots[chatbot_id]["appearance"]["button_text"]])[0]
        placeholder_text = post_data.get("placeholder_text", [chatbots[chatbot_id]["appearance"]["placeholder_text"]])[0]
        suggested_messages_text = post_data.get("suggested_messages", [""])[0]
        
        # Convertir les messages suggérés en liste
        suggested_messages = [msg.strip() for msg in suggested_messages_text.split("\n") if msg.strip()]
        
        # Mettre à jour l'apparence
        appearance_data = {
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "bubble_style": bubble_style,
            "font": font,
            "title": title,
            "button_text": button_text,
            "placeholder_text": placeholder_text,
            "suggested_messages": suggested_messages
        }
        
        update_chatbot_appearance(chatbot_id, appearance_data)
        
        self._send_redirect(f"/appearance/{chatbot_id}")
    
    def _handle_logo_upload(self, user_id, chatbot_id):
        if chatbot_id not in chatbots or chatbots[chatbot_id]["user_id"] != user_id:
            self._send_redirect("/dashboard")
            return
        
        # Récupérer les données du fichier
        content_type = self.headers.get('Content-Type')
        
        if not content_type or not content_type.startswith('multipart/form-data'):
            self._send_redirect(f"/appearance/{chatbot_id}?error=Format%20de%20fichier%20non%20valide")
            return
        
        # Parser le multipart form-data
        boundary = content_type.split('=')[1].encode()
        remain_bytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remain_bytes -= len(line)
        
        if not boundary in line:
            self._send_redirect(f"/appearance/{chatbot_id}?error=Format%20non%20valide")
            return
        
        # Lire les headers du fichier
        line = self.rfile.readline()
        remain_bytes -= len(line)
        
        # Chercher le nom du fichier
        filename = None
        if b'Content-Disposition' in line:
            filename_search = re.search(b'filename="(.*)"', line)
            if filename_search:
                filename = filename_search.group(1).decode()
        
        if not filename:
            self._send_redirect(f"/appearance/{chatbot_id}?error=Aucun%20fichier%20sélectionné")
            return
        
        # Lire les lignes jusqu'au contenu du fichier
        while remain_bytes > 0:
            line = self.rfile.readline()
            remain_bytes -= len(line)
            if line == b'\r\n':
                break
        
        # Lire le contenu du fichier
        file_data = b''
        prev_line = None
        while remain_bytes > 0:
            line = self.rfile.readline()
            remain_bytes -= len(line)
            
            if boundary in line:
                # Fin du fichier, retirer les derniers \r\n
                if prev_line and prev_line.endswith(b'\r\n'):
                    file_data = file_data[:-2]
                break
            
            file_data += line
            prev_line = line
        
        # Vérifier que le fichier est bien un type d'image valide
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
            self._send_redirect(f"/appearance/{chatbot_id}?error=Format%20de%20fichier%20non%20supporté.%20Utilisez%20PNG,%20JPG,%20JPEG,%20GIF%20ou%20SVG.")
            return
        
        # Enregistrer le logo
        success, message = upload_logo(chatbot_id, file_data, filename)
        
        if success:
            self._send_redirect(f"/appearance/{chatbot_id}")
        else:
            self._send_redirect(f"/appearance/{chatbot_id}?error={message}")

# Démarrage du serveur
def run_server():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serveur démarré sur http://{HOST}:{PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
