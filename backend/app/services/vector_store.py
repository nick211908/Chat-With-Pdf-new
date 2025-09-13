from pymongo import MongoClient
# --- THE FIX: Import the class with the correct name ---
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.docstore.document import Document
from app.core.config import settings
from loguru import logger
import numpy as np

class MongoVectorStore:
    """
    Manages storing and retrieving document embeddings from MongoDB using
    the Hugging Face Inference API for embeddings.
    """
    def __init__(self):
        try:
            self.client = MongoClient(settings.MONGO_CONNECTION_STRING)
            self.db = self.client.get_database("chat_with_pdf_db")
            self.collection = self.db.get_collection("document_embeddings_hf")

            # --- CRITICAL CHANGE: Use the corrected class name here ---
            logger.info("Initializing Hugging Face Inference API client...")
            self.embedding_model = HuggingFaceEndpointEmbeddings(
                huggingfacehub_api_token=settings.HUGGINGFACE_API_TOKEN,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.success("Successfully connected to MongoDB and initialized Hugging Face embedding client.")
        except Exception as e:
            logger.error(f"Failed to initialize MongoVectorStore: {e}")
            raise

    # The rest of the file remains exactly the same.
    # The add_documents and get_retriever methods will work perfectly.

    def add_documents(self, documents: list[Document], user_id: str, document_id: str):
        """
        Embeds document chunks via API call and adds them to the vector store.
        """
        try:
            texts_to_embed = [doc.page_content for doc in documents]
            if not texts_to_embed:
                logger.warning("No text found in documents to embed.")
                return

            embeddings = self.embedding_model.embed_documents(texts_to_embed)

            docs_to_insert = []
            for i, doc in enumerate(documents):
                doc.metadata["user_id"] = user_id
                doc.metadata["document_id"] = document_id
                docs_to_insert.append({
                    "text": doc.page_content,
                    "embedding": embeddings[i],
                    "metadata": doc.metadata
                })

            if docs_to_insert:
                self.collection.insert_many(docs_to_insert)

            logger.success(f"All {len(documents)} chunks have been successfully embedded and stored.")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def get_retriever(self, user_id: str, document_id: str, k: int = 5):
        """
        Returns a retriever that performs vector similarity search.
        """
        def vector_similarity_retriever(query: str):
            try:
                query_embedding = self.embedding_model.embed_query(query)
                
                results = list(self.collection.find({
                    "metadata.user_id": user_id,
                    "metadata.document_id": document_id
                }))
                
                if not results:
                    logger.warning(f"No documents found for user {user_id} and document {document_id}")
                    return []
                
                scored_docs = []
                for result in results:
                    doc_embedding = result['embedding']
                    similarity_score = np.dot(query_embedding, doc_embedding)
                    
                    metadata = result['metadata'].copy()
                    metadata['similarity_score'] = float(similarity_score)
                    
                    scored_docs.append({
                        'document': Document(
                            page_content=result['text'], 
                            metadata=metadata
                        ),
                        'score': similarity_score
                    })
                
                scored_docs.sort(key=lambda x: x['score'], reverse=True)
                top_docs = [item['document'] for item in scored_docs[:k]]
                
                logger.info(f"Retrieved {len(top_docs)} similar documents for query.")
                return top_docs
                
            except Exception as e:
                logger.error(f"Error in vector similarity search: {e}")
                return []
        
        return vector_similarity_retriever

vector_store = MongoVectorStore()

def get_vector_store() -> MongoVectorStore:
    return vector_store
