import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.vector_store import vector_store
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

class RAGPipeline:
    """
    A class to encapsulate the RAG pipeline logic.
    """
    def __init__(self):
        logger.info("Initializing RAG pipeline...")
        self.vector_store = vector_store
        self.llm = self._init_llm()

    def _init_llm(self):
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.GOOGLE_GEMINI_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.exception("Failed to initialize GoogleGenerativeAI. Check API key and dependencies.")
            raise

    def _create_rag_chain(self, retriever):
        prompt_template = """
        You are a helpful assistant specialized in answering questions based on the provided document context.
        Your answers should be concise, accurate, and directly based on the information given.
        Do not make up information. If the answer is not in the context, say "I couldn't find the answer in the provided document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
        {"context": lambda docs: format_docs(retriever(docs)), "question": RunnablePassthrough()}
        | prompt
        | self.llm
        | StrOutputParser()
        )
        return rag_chain

    def invoke(self, query: str, document_id: str, user_id: str):
        """
        Invokes the RAG pipeline for a given query and document.
        """
        logger.info(f"Invoking RAG pipeline for user {user_id} and doc {document_id}")
        retriever = self.vector_store.get_retriever(document_id=document_id)
        rag_chain = self._create_rag_chain(retriever)
        return rag_chain.invoke(query)