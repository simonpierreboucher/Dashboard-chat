import os
import re
import uuid
import json
import hashlib
import requests
import threading
import tiktoken
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlparse
import openai
import markdown
import faiss
import base64

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

# Initialisation du tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Dossier pour stocker les données
os.makedirs("data", exist_ok=True)
os.makedirs("data/vector_stores", exist_ok=True)
os.makedirs("data/chatbots", exist_ok=True)
os.makedirs("data/users", exist_ok=True)

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

# Web Crawler
class WebCrawler:
    def __init__(self, start_url, max_depth=MAX_CRAWL_DEPTH, max_pages=MAX_PAGES):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
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
    
    def crawl(self, url, depth=0):
        if depth > self.max_depth or len(self.visited_urls) >= self.max_pages or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        print(f"Crawling: {url} (profondeur: {depth})")
        
        content = self.get_page_content(url)
        if content:
            self.pages_content.append({
                "url": url,
                "content": content,
                "depth": depth
            })
        
        if depth < self.max_depth:
            links = self.extract_links(url)
            for link in links:
                self.crawl(link, depth + 1)
    
    def start(self):
        self.crawl(self.start_url)
        return self.pages_content

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
        "knowledge_sources": []
    }
    
    chatbots[chatbot_id] = chatbot
    save_chatbots()
    
    return chatbot

def add_knowledge_to_chatbot(chatbot_id, website_url):
    if chatbot_id not in chatbots:
        return False, "Chatbot introuvable"
    
    # Crawl le site web
    crawler = WebCrawler(website_url)
    pages = crawler.start()
    
    if not pages:
        return False, "Aucune page n'a pu être extraite du site"
    
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
        "added_at": datetime.now().isoformat()
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

def chat_with_bot(chatbot_id, user_input, conversation_history=None):
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
    
    return response["choices"][0]["message"]["content"]

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

def generate_widget_code(chatbot_id):
    chatbot = chatbots.get(chatbot_id)
    if not chatbot:
        return None
    
    share_token = chatbot["share_token"]
    host = f"http://{HOST}:{PORT}"
    
    # Code HTML et JS pour le widget
    widget_code = f"""
<!-- {chatbot['name']} Chatbot Widget -->
<div id="ai-chatbot-widget">
  <div id="ai-chatbot-toggle">💬</div>
  <div id="ai-chatbot-container" style="display:none;">
    <div id="ai-chatbot-header">
      <h3>{chatbot['name']}</h3>
      <button id="ai-chatbot-close">×</button>
    </div>
    <div id="ai-chatbot-messages"></div>
    <div id="ai-chatbot-input-container">
      <input type="text" id="ai-chatbot-input" placeholder="Posez votre question...">
      <button id="ai-chatbot-send">Envoyer</button>
    </div>
  </div>
</div>

<style>
  #ai-chatbot-widget {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-family: Arial, sans-serif;
    z-index: 9999;
  }}
  
  #ai-chatbot-toggle {{
    width: 50px;
    height: 50px;
    background-color: #0084ff;
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
    background-color: #0084ff;
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
    border-radius: 10px;
    max-width: 80%;
  }}
  
  .ai-chatbot-user-message {{
    background-color: #e6f2ff;
    margin-left: auto;
  }}
  
  .ai-chatbot-bot-message {{
    background-color: #f0f0f0;
    margin-right: auto;
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
    background-color: #0084ff;
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
    const input = document.getElementById('ai-chatbot-input');
    const sendBtn = document.getElementById('ai-chatbot-send');
    
    const chatbotId = '{chatbot_id}';
    const shareToken = '{share_token}';
    const apiUrl = '{host}/api/chat/' + chatbotId;
    
    let conversation = [];
    
    // Afficher le message de bienvenue
    function showWelcomeMessage() {{
      addMessage('{chatbot["welcome_message"]}', 'bot');
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

# Sauvegarde des données
def save_users():
    with open("data/users.json", "w") as f:
        json.dump(users_db, f)

def save_chatbots():
    with open("data/chatbots.json", "w") as f:
        json.dump(chatbots, f)

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
        
        else:
            self._set_headers(status_code=404)
            self.wfile.write(b"404 Not Found")
    
    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        session_id, user_id = self._get_session()
        
        # Routes d'API
        if path.startswith("/api/"):
            return self._handle_api_post(path, user_id)
        
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
        
        else:
            self._set_headers("application/json", 404)
            self.wfile.write(json.dumps({"error": "API endpoint not found"}).encode())
    
    def _handle_api_post(self, path, user_id):
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
                response = chat_with_bot(chatbot_id, message, conversation)
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
                    font-family: Arial, sans-serif;
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
                    background-color: #0084ff;
                    color: white;
                    padding: 15px;
                    text-align: center;
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
                    border-radius: 10px;
                }}
                .bot {{
                    background-color: #f0f0f0;
                    align-self: flex-start;
                }}
                .user {{
                    background-color: #e6f2ff;
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
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 10px 15px;
                    cursor: pointer;
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
                        <h2>Discutez avec moi</h2>
                    </div>
                    <div class="chat-messages" id="chat-messages">
                        <div class="message bot">{shared_chatbot['welcome_message']}</div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="user-input" placeholder="Posez votre question...">
                        <button id="send-button">Envoyer</button>
                    </div>
                </div>
            </div>
            
            <script>
                const chatMessages = document.getElementById('chat-messages');
                const userInput = document.getElementById('user-input');
                const sendButton = document.getElementById('send-button');
                
                let conversation = [];
                
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
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        addMessage('Désolé, une erreur est survenue. Veuillez réessayer plus tard.', 'bot');
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
                        <h3>Web Crawler Intégré</h3>
                        <p>Importez automatiquement le contenu de votre site web pour enrichir les connaissances de votre chatbot.</p>
                    </div>
                    <div class="feature">
                        <h3>Widget Personnalisable</h3>
                        <p>Intégrez facilement votre chatbot sur votre site avec notre widget prêt à l'emploi.</p>
                    </div>
                    <div class="feature">
                        <h3>Lien de Partage</h3>
                        <p>Partagez votre chatbot avec un simple lien, sans nécessiter d'inscription.</p>
                    </div>
                    <div class="feature">
                        <h3>Base de Connaissances Évolutive</h3>
                        <p>Ajoutez facilement de nouvelles sources d'information pour améliorer votre chatbot.</p>
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
            chatbots_html += f"""
            <div class="chatbot-card">
                <h3>{chatbot['name']}</h3>
                <p>{chatbot['description']}</p>
                <p><strong>Sources de connaissances:</strong> {knowledge_count}</p>
                <div class="chatbot-actions">
                    <a href="/chatbot/{chatbot_id}" class="btn">Gérer</a>
                    <a href="/share/{chatbot['share_token']}" class="btn" target="_blank">Aperçu</a>
                </div>
            </div>
            """
        
        if not chatbots_html:
            chatbots_html = """
            <div class="empty-state">
                <p>Vous n'avez pas encore créé de chatbot.</p>
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
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
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
                }}
                .chatbots-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                }}
                .chatbot-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chatbot-actions {{
                    display: flex;
                    margin-top: 15px;
                }}
                .chatbot-actions .btn {{
                    background-color: #f0f0f0;
                    color: #333;
                    padding: 8px 15px;
                    border-radius: 5px;
                    text-decoration: none;
                    margin-right: 10px;
                }}
                .empty-state {{
                    text-align: center;
                    padding: 50px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
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
                }
                form {
                    display: flex;
                    flex-direction: column;
                }
                label {
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input, textarea {
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                textarea {
                    min-height: 100px;
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
                }
                .error {
                    color: red;
                    margin-bottom: 15px;
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
                <div class="create-form">
                    <h2>Créer un nouveau Chatbot</h2>
                    <div id="error-message" class="error" style="display: none;"></div>
                    <form id="create-form" method="post" action="/create-chatbot">
                        <label for="name">Nom du Chatbot</label>
                        <input type="text" id="name" name="name" required>
                        
                        <label for="description">Description</label>
                        <textarea id="description" name="description" required></textarea>
                        
                        <label for="welcome-message">Message de Bienvenue</label>
                        <textarea id="welcome-message" name="welcome_message" placeholder="Bonjour ! Je suis votre assistant. Comment puis-je vous aider aujourd'hui ?">Bonjour ! Je suis votre assistant. Comment puis-je vous aider aujourd'hui ?</textarea>
                        
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
        
        # Générer le code du widget
        widget_code = generate_widget_code(chatbot_id)
        widget_code_escaped = widget_code.replace("<", "&lt;").replace(">", "&gt;")
        
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{chatbot['name']} - Chatbot IA</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
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
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .chatbot-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
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
                    margin-right: 10px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .card h3 {{
                    color: #0084ff;
                    margin-top: 0;
                }}
                .knowledge-item {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                }}
                .knowledge-item h4 {{
                    margin-top: 0;
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
                label {{
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                    margin-bottom: 15px;
                }}
                .chat-test {{
                    display: flex;
                    flex-direction: column;
                    height: 400px;
                }}
                .chat-messages {{
                    flex-grow: 1;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-y: auto;
                    margin-bottom: 15px;
                }}
                .chat-input-container {{
                    display: flex;
                }}
                .chat-input {{
                    flex-grow: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }}
                .message {{
                    max-width: 80%;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 10px;
                }}
                .bot {{
                    background-color: #f0f0f0;
                    align-self: flex-start;
                }}
                .user {{
                    background-color: #e6f2ff;
                    align-self: flex-end;
                    margin-left: auto;
                }}
                .code-container {{
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    overflow-x: auto;
                    margin-top: 15px;
                }}
                code {{
                    font-family: monospace;
                    white-space: pre-wrap;
                }}
                .copy-btn {{
                    margin-top: 10px;
                    background-color: #0084ff;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                .tab-container {{
                    margin-top: 20px;
                }}
                .tabs {{
                    display: flex;
                    border-bottom: 1px solid #ddd;
                    margin-bottom: 20px;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    border: 1px solid transparent;
                }}
                .tab.active {{
                    border: 1px solid #ddd;
                    border-bottom: none;
                    border-radius: 5px 5px 0 0;
                    background-color: white;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .share-info {{
                    margin-bottom: 15px;
                }}
                .share-link {{
                    padding: 10px;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    word-break: break-all;
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
                <div class="chatbot-header">
                    <h2>{chatbot['name']}</h2>
                    <div>
                        <a href="/share/{chatbot['share_token']}" class="btn-secondary" target="_blank">Aperçu</a>
                        <a href="/dashboard" class="btn-secondary">Retour</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Informations</h3>
                    <p><strong>Description:</strong> {chatbot['description']}</p>
                    <p><strong>Message de bienvenue:</strong> {chatbot['welcome_message']}</p>
                    <p><strong>Créé le:</strong> {chatbot['created_at'].split('T')[0]}</p>
                </div>
                
                <div class="tab-container">
                    <div class="tabs">
                        <div class="tab active" data-tab="knowledge">Sources de Connaissances</div>
                        <div class="tab" data-tab="test">Tester le Chatbot</div>
                        <div class="tab" data-tab="share">Partager</div>
                        <div class="tab" data-tab="widget">Widget</div>
                    </div>
                    
                    <div class="tab-content active" id="knowledge-tab">
                        <div class="card">
                            <h3>Sources de Connaissances</h3>
                            <div class="knowledge-list">
                                {knowledge_html}
                            </div>
                            
                            <div class="add-knowledge-form">
                                <h4>Ajouter une nouvelle source</h4>
                                <div id="knowledge-error" class="error" style="display: none;"></div>
                                <form id="add-knowledge-form">
                                    <label for="website-url">URL du site web</label>
                                    <input type="url" id="website-url" name="website-url" placeholder="https://exemple.com" required>
                                    
                                    <button type="submit" id="add-knowledge-btn" class="btn-primary">Ajouter cette source</button>
                                </form>
                            </div>
                        </div>
                    </div>
                    
                    <div class="tab-content" id="test-tab">
                        <div class="card">
                            <h3>Tester le Chatbot</h3>
                            <div class="chat-test">
                                <div class="chat-messages" id="chat-messages">
                                    <div class="message bot">{chatbot['welcome_message']}</div>
                                </div>
                                <div class="chat-input-container">
                                    <input type="text" id="chat-input" class="chat-input" placeholder="Posez votre question...">
                                    <button id="chat-send" class="btn-primary" style="margin-left: 10px;">Envoyer</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="tab-content" id="share-tab">
                        <div class="card">
                            <h3>Partager votre Chatbot</h3>
                            <div class="share-info">
                                <p>Partagez ce lien pour permettre à n'importe qui d'utiliser votre chatbot:</p>
                            </div>
                            <div class="share-link">
                                http://{HOST}:{PORT}/share/{chatbot['share_token']}
                            </div>
                            <button id="copy-share-link" class="btn-primary">Copier le lien</button>
                        </div>
                    </div>
                    
                    <div class="tab-content" id="widget-tab">
                        <div class="card">
                            <h3>Widget pour votre site web</h3>
                            <p>Copiez ce code et collez-le dans votre site web pour intégrer le chatbot:</p>
                            <div class="code-container">
                                <code id="widget-code">{widget_code_escaped}</code>
                            </div>
                            <button id="copy-widget-code" class="copy-btn">Copier le code</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Gestion des onglets
                const tabs = document.querySelectorAll('.tab');
                const tabContents = document.querySelectorAll('.tab-content');
                
                tabs.forEach(tab => {{
                    tab.addEventListener('click', () => {{
                        const tabId = tab.getAttribute('data-tab');
                        
                        // Activer l'onglet
                        tabs.forEach(t => t.classList.remove('active'));
                        tab.classList.add('active');
                        
                        // Afficher le contenu de l'onglet
                        tabContents.forEach(content => {{
                            content.classList.remove('active');
                        }});
                        document.getElementById(tabId + '-tab').classList.add('active');
                    }});
                }});
                
                // Copier le lien de partage
                document.getElementById('copy-share-link').addEventListener('click', () => {{
                    const shareLink = document.querySelector('.share-link').textContent.trim();
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
                
                // Test du chatbot
                const chatMessages = document.getElementById('chat-messages');
                const chatInput = document.getElementById('chat-input');
                const chatSend = document.getElementById('chat-send');
                
                let conversation = [];
                
                function addMessage(text, sender) {{
                    const messageElement = document.createElement('div');
                    messageElement.classList.add('message');
                    messageElement.classList.add(sender);
                    messageElement.textContent = text;
                    chatMessages.appendChild(messageElement);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }}
                
                function sendMessage() {{
                    const text = chatInput.value.trim();
                    if (!text) return;
                    
                    addMessage(text, 'user');
                    chatInput.value = '';
                    
                    conversation.push({{ role: 'user', content: text }});
                    
                    fetch('/api/chat/{chatbot_id}', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            message: text,
                            conversation: conversation
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        const reply = data.response;
                        addMessage(reply, 'bot');
                        conversation.push({{ role: 'assistant', content: reply }});
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        addMessage('Désolé, une erreur est survenue. Veuillez réessayer plus tard.', 'bot');
                    }});
                }}
                
                chatSend.addEventListener('click', sendMessage);
                chatInput.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') sendMessage();
                }});
                
                // Ajout d'une source de connaissances
                const addKnowledgeForm = document.getElementById('add-knowledge-form');
                const addKnowledgeBtn = document.getElementById('add-knowledge-btn');
                const knowledgeError = document.getElementById('knowledge-error');
                
                addKnowledgeForm.addEventListener('submit', function(e) {{
                    e.preventDefault();
                    
                    const websiteUrl = document.getElementById('website-url').value;
                    
                    // Afficher le loader
                    const originalBtnText = addKnowledgeBtn.innerHTML;
                    addKnowledgeBtn.innerHTML = '<span class="loader"></span> Traitement en cours...';
                    addKnowledgeBtn.disabled = true;
                    
                    fetch('/api/add-knowledge', {{
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
                        addKnowledgeBtn.innerHTML = originalBtnText;
                        addKnowledgeBtn.disabled = false;
                        
                        if (data.success) {{
                            alert('Source de connaissances ajoutée avec succès !');
                            // Recharger la page pour afficher la nouvelle source
                            window.location.reload();
                        }} else {{
                            knowledgeError.textContent = data.message;
                            knowledgeError.style.display = 'block';
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        addKnowledgeBtn.innerHTML = originalBtnText;
                        addKnowledgeBtn.disabled = false;
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

# Démarrage du serveur
def run_server():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serveur démarré sur http://{HOST}:{PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
