import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from app.models.response import UploadResponse
from app.core.security import get_current_user
from app.services.pdf_loader import PDFLoader
from app.services.vector_store import MongoVectorStore
from app.services.vector_store import get_vector_store
from loguru import logger

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile,
    current_user: dict = Depends(get_current_user),
    vector_store: MongoVectorStore = Depends(get_vector_store)
):
    """
    Handles PDF file uploads.
    """
    user_id = current_user.get("sub")
    logger.info(f"Received upload request from user: {user_id}")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    try:
        document_id = str(uuid.uuid4())
        logger.info(f"Generated document_id: '{document_id}'")

        pdf_loader = PDFLoader(upload_file=file)
        documents = await pdf_loader.load_and_chunk()
        logger.info(f"Processed {len(documents)} document chunks")

        vector_store.add_documents(
            documents=documents,
            user_id=user_id,
            document_id=document_id
        )

        logger.success(f"Successfully processed and stored document {document_id} for user {user_id}")
        
        return UploadResponse(
            message="File uploaded and processed successfully.",
            filename=file.filename,
            document_id=document_id
        )
    except Exception as e:
        logger.error(f"Error during file upload for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
