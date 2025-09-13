from fastapi import APIRouter, Depends, HTTPException
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.core.security import get_current_user
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import MongoVectorStore, get_vector_store
from loguru import logger

router = APIRouter()
rag_pipeline = RAGPipeline()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_doc(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    vector_store: MongoVectorStore = Depends(get_vector_store)
):
    """
    Handles chat requests for a specific document.
    """
    user_id = current_user.get("sub")
    logger.info(f"Received chat request from user {user_id} for doc {request.document_id}")

    try:
        retriever = vector_store.get_retriever(
            user_id=user_id, document_id=request.document_id
        )

        chain = rag_pipeline._create_rag_chain(retriever)
        result = await chain.ainvoke(request.question)

        logger.info(f"Successfully generated response for user {user_id}")
        return ChatResponse(
            question=request.question,
            answer=result,
            document_id=request.document_id,
        )
    except Exception as e:
        logger.error(f"Error processing chat request for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your chat request.")
