from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import logging
import json
import os
from datetime import datetime
from models import ChatHistory, User
from app import db
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EnhancedLaborLawSystem:
    """Enhanced labor law system with RAG, history, and chat functionality"""
    
    def __init__(self):
        logger.info("Initializing EnhancedLaborLawSystem")
        try:
            self.llm = ChatOpenAI(temperature=0)
            logger.info("LLM initialized successfully")
            
            self.embeddings = OpenAIEmbeddings()
            logger.info("Embeddings initialized successfully")
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            logger.info("Memory initialized successfully")
            
            self.vectorstore = self._initialize_vectorstore()
            logger.info("Vectorstore initialized successfully")
            
            self.qa_chain = self._initialize_qa_chain()
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.exception(f"Error initializing EnhancedLaborLawSystem: {e}")
            raise
        
    def _initialize_vectorstore(self) -> FAISS:
        try:
            logger.info("Loading labor law chunks from file")
            with open('attached_assets/labor_law_chunks.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"Loaded {len(chunks)} chunks from file")
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [{'article': chunk.get('article'), 'chunk_id': chunk.get('chunk_id')} for chunk in chunks]
            
            vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            logger.info("Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            logger.exception(f"Error initializing vectorstore: {str(e)}")
            # Return a dummy vectorstore for testing
            logger.warning("Returning dummy vectorstore due to initialization error")
            return FAISS.from_texts(["Test document"], self.embeddings)

    def _initialize_qa_chain(self) -> ConversationalRetrievalChain:
        try:
            # Create a custom prompt template
            template = """You are a helpful assistant for Jordanian labor law. Use the following pieces of context to answer the user's question in Arabic.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Always provide detailed answers based on the legal text provided.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Human: {question}
            
            Assistant: Let me help you with that based on Jordanian labor law."""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # Increased from 3 to 5 for more context
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
        except Exception as e:
            print(f"Error initializing QA chain: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {query}")
            result = self.qa_chain({"question": query})
            logger.info(f"Query processed successfully: {result}")
            
            return {
                "answer": result["answer"],
                "sources": [
                    {
                        "article": doc.metadata.get("article"),
                        "text": doc.page_content
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            logger.exception(f"Error processing query: {str(e)}")
            return {
                "answer": "عذراً، حدث خطأ في معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "sources": []
            }
    
    def _save_to_history(self, user_id: int, query: str, answer: str):
        """Save conversation to chat history"""
        try:
            history = ChatHistory(
                user_id=user_id,
                query=query,
                answer=answer,
                timestamp=datetime.utcnow()
            )
            db.session.add(history)
            db.session.commit()
        except Exception as e:
            logger.exception(f"Error saving to chat history: {e}")
    
    def get_chat_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat history for a user"""
        try:
            history = ChatHistory.query.filter_by(user_id=user_id)\
                .order_by(ChatHistory.timestamp.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "query": h.query,
                    "answer": h.answer,
                    "timestamp": h.timestamp.isoformat()
                }
                for h in history
            ]
        except Exception as e:
            logger.exception(f"Error retrieving chat history: {e}")
            return []

def initialize_enhanced_system() -> EnhancedLaborLawSystem:
    """Initialize the enhanced labor law system"""
    return EnhancedLaborLawSystem() 