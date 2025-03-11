import os
import tempfile
import chromadb
import ollama
import streamlit as st
import sqlite3
import datetime
import time
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Get the absolute path of the project root directory
# Adjust this if your script is not directly in the src/RAG directory
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(current_script_path)))

# Define the system prompt for LLM with multilingual support
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: 
- Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
- If the question is in Indonesian, answer in Indonesian. If the question is in English, answer in English. Match the language of your response to the language of the question.
"""

# Configuration parameters
AVAILABLE_LLM_MODELS = ["llama3", "deepseek-r1:8b"]
DEFAULT_LLM_MODEL = "deepseek-r1:8b"
AVAILABLE_EMBEDDING_MODELS = ["nomic-embed-text:latest"]
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"

# Use absolute paths for database files
VECTORDB_PATH = os.path.join(project_root, "db", "vector")
DB_PATH = os.path.join(project_root, "db", "chat-history", "chat_db.sqlite")

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100

# Debug information to help identify path issues
st.sidebar.info(f"Project root: {project_root}")
st.sidebar.info(f"Vector DB path: {VECTORDB_PATH}")
st.sidebar.info(f"SQLite DB path: {DB_PATH}")

# Ensure directories exist


def ensure_directories():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        os.makedirs(VECTORDB_PATH, exist_ok=True)
        st.sidebar.success("Database directories created successfully")
    except Exception as e:
        st.sidebar.error(f"Error creating directories: {e}")


# Call ensure_directories at startup
ensure_directories()

# Initialize SQLite database for chat history


def init_db():
    # Ensure directory exists
    ensure_directories()

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            thinking_content TEXT,
            query_results TEXT,
            relevant_text_ids TEXT,
            relevant_text TEXT
        )
        ''')

        c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
        return False

# Get session ID


def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return st.session_state.session_id

# Save message to DB


def save_message(role, content, thinking_content=None, query_results=None, relevant_text_ids=None, relevant_text=None):
    session_id = get_session_id()

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Convert non-string data to JSON string
        import json
        query_results_json = json.dumps(
            query_results) if query_results is not None else None
        relevant_text_ids_json = json.dumps(
            relevant_text_ids) if relevant_text_ids is not None else None

        c.execute("""
            INSERT INTO chats 
            (session_id, role, content, thinking_content, query_results, relevant_text_ids, relevant_text) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, thinking_content, query_results_json, relevant_text_ids_json, relevant_text))

        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving message: {e}")
        return False

# Save document to DB


def save_document(name, chunks):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO documents (name, chunks) VALUES (?, ?)",
                  (name, chunks))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving document: {e}")
        return False


# Delete document from DB

def delete_document(doc_name):
    """Deletes a document from the database and vector store."""
    try:
        # 1. Delete from SQLite DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM documents WHERE name = ?", (doc_name,))
        conn.commit()
        conn.close()

        # 2. Delete from vector store
        collection = RAGModel().get_vector_collection()
        if collection:
            # Print all IDs for debugging
            try:
                all_data = collection.get()
                all_ids = all_data["ids"]
                all_metadatas = all_data["metadatas"]

                # Find document IDs by looking at metadata
                doc_ids = []
                for i, metadata in enumerate(all_metadatas):
                    # Check if this metadata contains file information that matches our document
                    source = metadata.get('source', '')
                    if doc_name in source or doc_name == source:
                        doc_ids.append(all_ids[i])

                # If no matches by metadata, try prefix matching
                if not doc_ids:
                    normalize_doc_name = doc_name.translate(
                        str.maketrans({"-": "_", ".": "_", " ": "_"}))
                    doc_ids = [id for id in all_ids if id.startswith(
                        f"{normalize_doc_name}_")]

                # Delete the chunks
                if doc_ids:
                    collection.delete(ids=doc_ids)
                    st.info(f"Deleted {len(doc_ids)} chunks from vector store")
            except Exception as e:
                st.warning(f"Vector store operation issue: {e}")

        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False
# Get chat history from DB


def get_chat_history():
    session_id = get_session_id()

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """SELECT role, content, thinking_content, query_results, relevant_text_ids, relevant_text 
            FROM chats WHERE session_id = ? ORDER BY timestamp""",
            (session_id,)
        )

        import json
        chat_history = []
        for row in c.fetchall():
            role, content, thinking_content, query_results, relevant_text_ids, relevant_text = row

            # Parse JSON strings back to objects if they exist
            if query_results and query_results.strip():
                try:
                    query_results = json.loads(query_results)
                except json.JSONDecodeError:
                    query_results = None

            if relevant_text_ids and relevant_text_ids.strip():
                try:
                    relevant_text_ids = json.loads(relevant_text_ids)
                except json.JSONDecodeError:
                    relevant_text_ids = None

            chat_history.append({
                "role": role,
                "content": content,
                "thinking_content": thinking_content,
                "query_results": query_results,
                "relevant_text_ids": relevant_text_ids,
                "relevant_text": relevant_text
            })

        conn.close()
        return chat_history
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []

# Get recent chat sessions


def get_recent_chat_sessions(limit=5):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Get unique sessions with first question and last timestamp
        c.execute("""
            SELECT 
                c.session_id, 
                MIN(c.timestamp) as start_time,
                MAX(c.timestamp) as last_time,
                (SELECT content FROM chats WHERE session_id = c.session_id AND role = 'user' ORDER BY timestamp ASC LIMIT 1) as first_question
            FROM chats c
            GROUP BY c.session_id
            ORDER BY last_time DESC
            LIMIT ?
        """, (limit,))

        recent_sessions = c.fetchall()
        conn.close()
        return recent_sessions
    except sqlite3.Error as e:
        st.error(f"Database error retrieving recent sessions: {e}")
        return []

# Get all documents from DB


def get_documents():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, chunks FROM documents ORDER BY timestamp DESC")
        documents = [{"name": name, "chunks": chunks}
                     for name, chunks in c.fetchall()]
        conn.close()
        return documents
    except sqlite3.Error as e:
        st.error(f"Database error retrieving documents: {e}")
        return []

# Delete Session From Database


def delete_chat_session(session_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Error deleting session: {e}")
        return False

# RAG Model Class


class RAGModel:
    def __init__(self, llm_model_name=DEFAULT_LLM_MODEL, embedding_model_name=DEFAULT_EMBEDDING_MODEL, db_dir=VECTORDB_PATH, collection_name="rag_app", chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, k_retrieval=10):
        """Initialize RAG model with specified parameters"""
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.k_retrieval = k_retrieval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define system prompt
        self.system_prompt = system_prompt

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        # Initialize cross-encoder model
        self.encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2")

    def get_vector_collection(self):
        """Gets or creates a ChromaDB collection for vector storage."""
        try:
            ollama_ef = OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings", model_name=self.embedding_model_name)
            chroma_client = chromadb.PersistentClient(path=self.db_dir)
            return chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})
        except Exception as e:
            st.error(f"Error connecting to vector database: {e}")
            return None

    def process_document(self, uploaded_file: UploadedFile) -> list[Document]:
        """Processes an uploaded PDF file by converting it to text chunks."""
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file
        return self.text_splitter.split_documents(docs)

    def add_to_vector_collection(self, all_splits: list[Document], file_name: str):
        """Adds document splits to a vector collection for semantic search."""
        collection = self.get_vector_collection()
        if not collection:
            return 0

        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return len(documents)

    def query_collection(self, prompt: str, n_results: int = 10):
        """Queries the vector collection with a given prompt."""
        collection = self.get_vector_collection()
        if not collection:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        return collection.query(query_texts=[prompt], n_results=n_results)

    def re_rank_documents(self, query: str, documents: list[str], top_k: int = 3) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
        if not documents or len(documents) == 0:
            return "", []

        relevant_text = ""
        relevant_text_ids = []
        ranks = self.encoder_model.rank(
            query, documents, top_k=min(top_k, len(documents)))
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]]
            relevant_text_ids.append(rank["corpus_id"])
        return relevant_text, relevant_text_ids

    # def call_llm(self, context: str, prompt: str):
    #     """Calls the language model with context and prompt to generate a response."""
    #     try:
    #         response = ollama.chat(
    #             model=self.llm_model_name,
    #             stream=True,
    #             messages=[{"role": "system", "content": self.system_prompt}, {
    #                 "role": "user", "content": f"Context: {context}, Question: {prompt}"}]
    #         )
    #         for chunk in response:
    #             if chunk["done"] is False:
    #                 yield chunk["message"]["content"]
    #             else:
    #                 break
    #     except Exception as e:
    #         yield f"Error calling LLM: {str(e)}"

    # def call_llm(self, context: str, prompt: str):
    #     """Calls the language model with context and prompt to generate a response using ngrok URL."""
    #     try:
    #         import requests
    #         import json

    #         # Use the same ngrok URL that's being used for embeddings
    #         ngrok_url = "https://3458-103-78-115-240.ngrok-free.app/api/chat"

    #         # Prepare the request payload
    #         payload = {
    #             "model": self.llm_model_name,
    #             "messages": [
    #                 {"role": "system", "content": self.system_prompt},
    #                 {"role": "user", "content": f"Context: {context}, Question: {prompt}"}
    #             ],
    #             "stream": True
    #         }

    #         # Make a streaming request to the ngrok endpoint
    #         response = requests.post(ngrok_url, json=payload, stream=True)

    #         if response.status_code != 200:
    #             yield f"Error: Received status code {response.status_code} from API"
    #             return

    #         # Process the streaming response
    #         for line in response.iter_lines():
    #             if line:
    #                 try:
    #                     # Remove 'data: ' prefix if it exists (common in SSE streams)
    #                     if line.startswith(b'data: '):
    #                         line = line[6:]

    #                     # Skip empty lines or heartbeats
    #                     if not line or line == b':' or line == b'data: [DONE]':
    #                         continue

    #                     # Parse the JSON
    #                     chunk = json.loads(line)

    #                     # Extract the content based on the expected response format
    #                     # This might need adjustment based on your API's exact response format
    #                     if "choices" in chunk and len(chunk["choices"]) > 0:
    #                         content = chunk["choices"][0].get(
    #                             "delta", {}).get("content", "")
    #                         if content:
    #                             yield content
    #                     elif "message" in chunk and "content" in chunk["message"]:
    #                         # Similar to your original format
    #                         yield chunk["message"]["content"]

    #                 except json.JSONDecodeError:
    #                     # Skip non-JSON lines
    #                     continue
    #                 except Exception as e:
    #                     yield f"Error parsing response: {str(e)}"

    #         # Add a fallback in case the stream closes without proper indication
    #         yield ""

    #     except Exception as e:
    #         yield f"Error calling LLM via ngrok: {str(e)}"

    def call_llm(self, context: str, prompt: str):
        """Calls the language model with Digital Ocean GenAI Agent API to generate a response."""
        try:
            import requests
            import json

            # Digital Ocean GenAI Agent API endpoint
            api_url = "https://agent-9d0a55ab65f61611182c-p7e2w.ondigitalocean.app/api/v1/chat/completions"

            # Your authorization token
            headers = {
                "Authorization": "Bearer eEtfrIkIm1-CK-HtfIwgdiGtcmAAfKsJ",
                "Content-Type": "application/json"
            }

            # Prepare the request payload according to Digital Ocean API docs
            payload = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context: {context}, Question: {prompt}"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True,
                "include_retrieval_info": True  # Include info about retrieved documents
            }

            # Make a streaming request to the Digital Ocean endpoint
            response = requests.post(
                api_url, headers=headers, json=payload, stream=True)

            if response.status_code != 200:
                yield f"Error: Received status code {response.status_code} from API"
                return

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Remove 'data: ' prefix if it exists (common in SSE streams)
                        if line.startswith(b'data: '):
                            line = line[6:]

                        # Skip empty lines or heartbeats
                        if not line or line == b':' or line == b'data: [DONE]':
                            continue

                        # Parse the JSON
                        chunk = json.loads(line)

                        # Extract the content based on Digital Ocean API response format
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    yield content
                            elif "message" in choice and "content" in choice["message"]:
                                yield choice["message"]["content"]

                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
                    except Exception as e:
                        yield f"Error parsing response: {str(e)}"

            # Add a fallback in case the stream closes without proper indication
            yield ""

        except Exception as e:
            yield f"Error calling Digital Ocean GenAI API: {str(e)}"

# Sidebar Component


def sidebar_component():
    """Sidebar component for RAG model."""

    # Check current view from session state
    current_view = st.session_state.get("rag_view", "documents")

    # Display Recent Chats section only when in Chat view
    if current_view == "chat":
        # Recent Chat History Section
        st.sidebar.title("Recent Chats")

        # Get recent chat sessions
        recent_sessions = get_recent_chat_sessions(5)  # Show last 5 sessions

        if recent_sessions:
            # Handle confirmation for deletion
            if "delete_confirm" in st.session_state and st.session_state.delete_confirm:
                session_to_delete = st.session_state.delete_confirm
                # Find the question for this session
                session_info = next(
                    (s for s in recent_sessions if s[0] == session_to_delete), None)
                if session_info:
                    question = session_info[3]
                    preview = question if len(
                        question) <= 30 else question[:27] + "..."
                    st.sidebar.warning(f"Delete chat: \"{preview}\"?")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.button("✓ Yes", key="confirm_yes"):
                            success = delete_chat_session(session_to_delete)
                            if success:
                                st.sidebar.success("Chat deleted!")
                                if st.session_state.session_id == session_to_delete:
                                    # Current session was deleted, create a new one
                                    st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            # Clear confirmation state
                            st.session_state.delete_confirm = None
                            st.rerun()
                    with col2:
                        if st.button("✗ No", key="confirm_no"):
                            # Clear confirmation state
                            st.session_state.delete_confirm = None
                            st.rerun()
                st.sidebar.divider()

            # List all sessions with delete buttons
            for session_id, start_time, last_time, first_question in recent_sessions:
                # Format timestamps
                try:
                    last_time_dt = datetime.datetime.fromisoformat(last_time)
                    formatted_time = last_time_dt.strftime("%d %b, %H:%M")
                except (ValueError, TypeError):
                    formatted_time = "Unknown time"

                # Handle None or empty first question
                if not first_question:
                    first_question = "New conversation"

                # Truncate first question if too long
                preview = first_question if len(
                    first_question) <= 40 else first_question[:37] + "..."

                # Create two columns for session button and delete button
                col1, col2 = st.sidebar.columns([5, 1])

                # Session button in first column
                with col1:
                    if st.button(
                        f"🗨️ {preview}\n📅 {formatted_time}",
                        key=f"history_{session_id}",
                        use_container_width=True
                    ):
                        # Switch to this session and reload
                        st.session_state.session_id = session_id
                        st.rerun()

                # Delete button in second column
                with col2:
                    st.write("")  # Add some spacing
                    if st.button("🗑️", key=f"delete_{session_id}"):
                        st.session_state.delete_confirm = session_id
                        st.rerun()
        else:
            st.sidebar.info("No recent chats found")

        # Add a button to start a new chat session
        if st.sidebar.button("Start New Chat Session", type="primary", use_container_width=True):
            # Generate a new session ID
            st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            st.rerun()

        st.sidebar.divider()

    # RAG configuration section (shown in both views)
    st.sidebar.title("RAG Configuration")

    # Model selection section - AS EXPANDER
    with st.sidebar.expander("Model Selection", expanded=True):
        # LLM model selection using available models
        llm_models = AVAILABLE_LLM_MODELS
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = DEFAULT_LLM_MODEL

        selected_model = st.selectbox(
            "Language Model",
            llm_models,
            index=llm_models.index(
                st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
        )

        # Update model if changed
        if selected_model != st.session_state.llm_model:
            st.session_state.llm_model = selected_model
            if "rag_model" in st.session_state:
                st.session_state.rag_model.llm_model_name = selected_model
                st.success(f"Model updated to {selected_model}")

        # Embedding model selection
        embedding_models = AVAILABLE_EMBEDDING_MODELS
        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL

        selected_embedding = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=embedding_models.index(
                st.session_state.embedding_model) if st.session_state.embedding_model in embedding_models else 0
        )

        # Update embedding model if changed
        if selected_embedding != st.session_state.embedding_model:
            st.session_state.embedding_model = selected_embedding
            if "rag_model" in st.session_state:
                try:
                    st.session_state.rag_model = RAGModel(
                        llm_model_name=st.session_state.llm_model,
                        embedding_model_name=selected_embedding
                    )
                    st.success(
                        f"Embedding model updated to {selected_embedding}")
                except Exception as e:
                    st.error(f"Error updating embedding model: {str(e)}")

    # Model Settings - AS EXPANDER
    with st.sidebar.expander("Model Settings", expanded=False):
        # Chunking parameters
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Size of text chunks for processing"
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=DEFAULT_CHUNK_OVERLAP,
            step=10,
            help="Overlap between chunks"
        )

        # Parameter-parameter lain
        st.slider(
            "Context Length",
            min_value=1024,
            max_value=8192,
            value=4096,
            step=1024,
            help="Maximum context length"
        )

        top_k = st.slider(
            "Top K Retrieval",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Number of documents to retrieve"
        )

        # Apply chunking parameters if changed
        if "chunk_size" not in st.session_state or "chunk_overlap" not in st.session_state:
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
        elif st.session_state.chunk_size != chunk_size or st.session_state.chunk_overlap != chunk_overlap:
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            if "rag_model" in st.session_state:
                st.session_state.rag_model.chunk_size = chunk_size
                st.session_state.rag_model.chunk_overlap = chunk_overlap
                st.session_state.rag_model.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success("Chunking parameters updated")

    # Advanced Options - AS EXPANDER
    with st.sidebar.expander("Advanced Options", expanded=False):
        st.toggle("Keep Model in Memory", value=True)
        st.toggle("Use Re-ranking", value=True)

# RAG Chat Component


def document_management_component():
    """Component for document management."""
    # Initialize database
    init_db_result = init_db()
    if not init_db_result:
        st.error("Failed to initialize database. Check sidebar for details.")

    # Initialize RAG model
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = RAGModel()

    # Document Management Section
    st.header("Document Management")

    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        process_button = st.button(
            "Process Documents", type="primary", use_container_width=True)

    # Process documents
    if uploaded_files and process_button:
        with st.status("Processing documents...") as status:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
                normalize_uploaded_file_name = uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"}))
                all_splits = st.session_state.rag_model.process_document(
                    uploaded_file)
                st.write(f"Created {len(all_splits)} text chunks")

                chunks_added = st.session_state.rag_model.add_to_vector_collection(
                    all_splits, normalize_uploaded_file_name)

                # Save document to DB
                save_document(uploaded_file.name, len(all_splits))

                st.success(
                    f"Added {chunks_added} chunks from {uploaded_file.name}")

            status.update(
                label=f"Documents processed successfully!", state="complete")

    # Display processed documents
    documents = get_documents()
    if documents:
        st.subheader("Indexed Documents")

        # Create columns for the header
        col1, col2, col3, col4 = st.columns([0.1, 0.6, 0.2, 0.1])
        col1.write("**#**")
        col2.write("**Document**")
        col3.write("**Chunks**")
        col4.write("**Action**")

        st.divider()

        # List documents with delete buttons
        for i, doc in enumerate(documents):
            col1, col2, col3, col4 = st.columns([0.1, 0.6, 0.2, 0.1])
            col1.write(f"{i+1}")
            col2.write(doc['name'])
            col3.write(doc['chunks'])
            if col4.button("🗑️", key=f"delete_doc_{i}", help=f"Delete {doc['name']}"):
                if delete_document(doc['name']):
                    st.success(f"Deleted {doc['name']}")
                    time.sleep(1)  # Give user time to see the message
                    st.rerun()

        st.divider()

        # Add option to delete all documents
        if st.button("Reset Vector Database", type="tertiary"):
            try:
                # Delete the vector store directory and recreate it
                import shutil
                shutil.rmtree(VECTORDB_PATH)
                os.makedirs(VECTORDB_PATH, exist_ok=True)

                # Clear the documents table
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("DELETE FROM documents")
                conn.commit()
                conn.close()

                st.success("Vector database has been reset")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting vector database: {e}")
    else:
        st.info("No documents indexed. Upload and process documents to start.")


def chat_component():
    """Component for chat interface."""
    # Initialize database and RAG model
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = RAGModel()

    # Initialize session state for query results
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "relevant_text" not in st.session_state:
        st.session_state.relevant_text = None
    if "relevant_text_ids" not in st.session_state:
        st.session_state.relevant_text_ids = None

    # Chat Section
    st.header("Chat with your Documents")
    st.divider()

    # Display chat history
    chat_history = get_chat_history()

    if chat_history:
        for message in chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    # First show thinking process if available
                    if message.get("thinking_content"):
                        with st.expander("AI Thinking Process", expanded=True):
                            st.markdown(message["thinking_content"])

                    # Then display the main content
                    st.write(message["content"])

                    # Other expanders come after the content
                    if message.get("query_results"):
                        with st.expander("See retrieved documents"):
                            st.write(message["query_results"])

                    if message.get("relevant_text_ids"):
                        with st.expander("See most relevant document ids"):
                            st.write(message["relevant_text_ids"])

                    if message.get("relevant_text"):
                        with st.expander("See relevant text"):
                            st.write(message["relevant_text"])
    else:
        documents = get_documents()
        if documents:
            st.info(
                "Ask a question about your documents using the input box below")
        else:
            st.warning(
                "Please upload and process documents in the Document Management section first")

    # Use st.chat_input for a cleaner chat interface
    prompt = st.chat_input("Ask a question about your documents...")

    # Process prompt when submitted
    documents = get_documents()
    if prompt and documents:
        # Save user message to DB
        save_message("user", prompt)

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # First, show a status while retrieving documents
        with st.status("Retrieving relevant documents...") as status:
            try:
                # Get results from vector DB
                results = st.session_state.rag_model.query_collection(
                    prompt)
                st.session_state.query_results = results

                if results and len(results.get("documents", [[]])[0]) > 0:
                    context = results.get("documents")[0]
                    relevant_text, relevant_text_ids = st.session_state.rag_model.re_rank_documents(
                        prompt, context)

                    # Save to session state
                    st.session_state.relevant_text = relevant_text
                    st.session_state.relevant_text_ids = relevant_text_ids

                    status.update(
                        label="Documents retrieved successfully!", state="complete")
                else:
                    st.error(
                        "No relevant documents found. Please try a different question or upload more documents.")
                    status.update(
                        label="No relevant documents found", state="error")
                    return
            except Exception as e:
                st.error(f"Error retrieving documents: {str(e)}")
                status.update(
                    label=f"Error: {str(e)}", state="error")
                return

        # Generate assistant response
        with st.chat_message("assistant"):
            # First, create and auto-expand the thinking process expander
            with st.expander("AI Thinking Process", expanded=True):
                thinking_placeholder = st.empty()

            # Next, create placeholder for the response below the thinking process
            response_placeholder = st.empty()

            # Variables to store response and thinking process
            full_response = ""
            thinking_content = ""
            in_thinking_section = False

            # Define thinking starter phrases - these are typical phrases that models use when starting to think
            thinking_starters = ["Okay", "Let me",
                                 "I need to", "First", "Based on", "Looking at"]

            # Collect initial content to detect if it's a thinking process
            initial_content = ""
            initial_content_collected = False

            # Stream response from LLM
            for chunk in st.session_state.rag_model.call_llm(context=relevant_text, prompt=prompt):
                # Handle explicit thinking tags
                if "<think>" in chunk:
                    parts = chunk.split("<think>")
                    if len(parts) > 0:
                        full_response += parts[0]
                    if len(parts) > 1:
                        thinking_content += parts[1]
                    in_thinking_section = True
                    # We've found explicit tags, so stop checking for implicit thinking
                    initial_content_collected = True

                elif "</think>" in chunk and in_thinking_section:
                    parts = chunk.split("</think>")
                    thinking_content += parts[0]
                    if len(parts) > 1:
                        full_response += parts[1]
                    in_thinking_section = False

                elif in_thinking_section:
                    thinking_content += chunk

                else:
                    # Check for implicit thinking starters if we haven't collected much content yet
                    if not initial_content_collected and len(initial_content) < 100:
                        initial_content += chunk

                        # Check if we have enough content to determine if it's starting with thinking phrases
                        if len(initial_content) > 20 or "." in initial_content:
                            initial_content_collected = True

                            # Check if the initial content starts with any thinking starter phrases
                            if any(initial_content.strip().startswith(starter) for starter in thinking_starters):
                                # This looks like a thinking process without explicit tags
                                first_period = initial_content.find(".")
                                if first_period != -1 and first_period < 50:
                                    # If there's a period early in the text, treat everything up to
                                    # the first paragraph break as thinking
                                    first_paragraph_break = initial_content.find(
                                        "\n\n")
                                    if first_paragraph_break != -1:
                                        thinking_content = initial_content[:first_paragraph_break]
                                        full_response = initial_content[first_paragraph_break:]
                                    else:
                                        # No paragraph break yet, put it all in thinking for now
                                        thinking_content = initial_content
                                else:
                                    # No early period, just add it all to thinking for now
                                    thinking_content = initial_content
                            else:
                                # Not a thinking starter, treat as normal response
                                full_response = initial_content
                    else:
                        full_response += chunk

                # Update placeholders with latest content
                if in_thinking_section or not initial_content_collected:
                    thinking_placeholder.markdown(thinking_content + "▌")
                else:
                    thinking_placeholder.markdown(thinking_content)

                response_placeholder.markdown(
                    full_response + ("" if in_thinking_section or not initial_content_collected else "▌"))

                # Add slight delay for typing effect
                time.sleep(0.01)

            # Final update to remove cursor
            thinking_placeholder.markdown(thinking_content)
            response_placeholder.markdown(full_response)

            # Debug expanders after the answer
            with st.expander("See retrieved documents"):
                st.write(st.session_state.query_results)

            with st.expander("See most relevant document ids"):
                st.write(st.session_state.relevant_text_ids)

            with st.expander("See relevant text"):
                st.write(st.session_state.relevant_text)

            # Save complete response with debug info to DB
            save_message(
                role="assistant",
                content=full_response,
                thinking_content=thinking_content,
                query_results=st.session_state.query_results,
                relevant_text_ids=st.session_state.relevant_text_ids,
                relevant_text=st.session_state.relevant_text
            )


def rag():
    """Main function to run the RAG tool."""
    # First do sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.divider()

        # Set default view if not already set
        if "rag_view" not in st.session_state:
            st.session_state.rag_view = "documents"

        # Navigation buttons
        if st.button("📄 Document Management",
                     type="primary" if st.session_state.rag_view == "documents" else "secondary",
                     use_container_width=True):
            st.session_state.rag_view = "documents"
            st.rerun()

        if st.button("💬 Chat with Documents",
                     type="primary" if st.session_state.rag_view == "chat" else "secondary",
                     use_container_width=True):
            st.session_state.rag_view = "chat"
            st.rerun()

        st.divider()

    # Then call the sidebar component with all the settings
    sidebar_component()

    # Display the selected view
    if st.session_state.rag_view == "documents":
        document_management_component()
    else:
        chat_component()
