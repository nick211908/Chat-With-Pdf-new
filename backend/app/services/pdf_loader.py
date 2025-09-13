import tempfile
import os
from fastapi import UploadFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

class PDFLoader:
    """
    Handles loading and chunking of a PDF file.
    """
    def __init__(self, upload_file: UploadFile):
        self.upload_file = upload_file

    async def load_and_chunk(self) -> list:
        """
        Temporarily saves the uploaded PDF, loads it using PyMuPDFLoader,
        and splits it into smaller text chunks.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                await self.upload_file.seek(0)
                content = await self.upload_file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            logger.info(f"PDF temporarily saved to {tmp_path}")

            # Load the document using the temporary file path
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunked_documents = text_splitter.split_documents(documents)
            
            logger.info(f"Successfully loaded and split PDF into {len(chunked_documents)} chunks.")
            return chunked_documents

        except Exception as e:
            logger.error(f"Failed to load and chunk PDF: {e}")
            return []
        finally:
            # Clean up the temporary file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")

