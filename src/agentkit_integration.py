"""
AgentKit Integration Module
Integrates RAG system with LangChain agents for autonomous operation.
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.tools import BaseTool, tool
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import AgentAction, AgentFinish
except ImportError:
    print("Warning: LangChain not installed. Install with: pip install langchain langchain-openai")

from rag_pipeline import RAGPipeline


class RAGRetrievalTool(BaseTool):
    """Tool for retrieving information from RAG system."""
    
    name = "rag_retrieval"
    description = """Useful for retrieving information from the knowledge base.
    Input should be a question or search query.
    Returns relevant information from the documents."""
    
    rag_pipeline: RAGPipeline = None
    
    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline
    
    def _run(self, query: str) -> str:
        """Execute the tool."""
        try:
            result = self.rag_pipeline.query(
                question=query,
                top_k=3,
                return_sources=False
            )
            return result['answer']
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return f"Error retrieving information: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented)."""
        return self._run(query)


class DocumentSearchTool(BaseTool):
    """Tool for searching documents in the knowledge base."""
    
    name = "document_search"
    description = """Searches for relevant document chunks in the knowledge base.
    Input should be a search query.
    Returns the most relevant document excerpts with sources."""
    
    rag_pipeline: RAGPipeline = None
    
    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline
    
    def _run(self, query: str) -> str:
        """Execute the tool."""
        try:
            result = self.rag_pipeline.query(
                question=query,
                top_k=5,
                return_sources=True
            )
            
            # Format response with sources
            response = f"Found {result['num_sources']} relevant documents:\n\n"
            if result.get('sources'):
                for i, source in enumerate(result['sources'], 1):
                    response += f"{i}. {source['text']}\n"
                    response += f"   Source: {source['metadata'].get('source', 'Unknown')}\n\n"
            
            return response
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            return f"Error searching documents: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented)."""
        return self._run(query)


class RAGAgent:
    """
    Autonomous RAG agent with reasoning capabilities.
    Uses LangChain agents to orchestrate RAG operations.
    """
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize RAG Agent.
        
        Args:
            rag_pipeline: RAG pipeline instance (created if not provided)
            model: LLM model name
            temperature: Generation temperature
            max_iterations: Maximum agent iterations
            verbose: Enable verbose logging
        """
        # Initialize or use provided RAG pipeline
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True
        )
        
        logger.info("RAG Agent initialized successfully")
    
    def _create_tools(self) -> List[BaseTool]:
        """Create agent tools."""
        return [
            RAGRetrievalTool(rag_pipeline=self.rag_pipeline),
            DocumentSearchTool(rag_pipeline=self.rag_pipeline)
        ]
    
    def _create_agent(self):
        """Create the agent with prompt template."""
        # Define agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with access to a knowledge base through RAG (Retrieval Augmented Generation).
            
            Your capabilities:
            - Retrieve and synthesize information from documents
            - Search for specific information in the knowledge base
            - Answer questions based on retrieved context
            - Provide sources and citations when possible
            
            Guidelines:
            - Use the tools available to find information
            - Be accurate and cite your sources
            - If information is not available, say so clearly
            - Break down complex queries into simpler steps
            - Synthesize information from multiple sources when needed
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    def run(self, query: str) -> str:
        """
        Run the agent with a query.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        logger.info(f"Agent processing query: {query}")
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return result['output']
        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, message: str) -> str:
        """
        Chat with the agent (maintains conversation history).
        
        Args:
            message: User's message
            
        Returns:
            Agent's response
        """
        return self.run(message)
    
    def reset_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
        logger.info("Agent memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns:
            List of messages
        """
        return self.memory.chat_memory.messages


class MultiStepRAGAgent:
    """
    Advanced RAG agent with multi-step reasoning and planning.
    """
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        model: str = "gpt-4",
        verbose: bool = True
    ):
        """
        Initialize Multi-Step RAG Agent.
        
        Args:
            rag_pipeline: RAG pipeline instance
            model: LLM model (GPT-4 recommended for complex reasoning)
            verbose: Enable verbose logging
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.verbose = verbose
        
        logger.info("Multi-Step RAG Agent initialized")
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
        """
        prompt = f"""Break down the following complex question into simpler sub-questions that can be answered independently:

Question: {query}

Provide a numbered list of sub-questions:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse sub-questions
            lines = response.content.split('\n')
            sub_queries = [
                line.split('.', 1)[1].strip()
                for line in lines
                if line.strip() and any(c.isdigit() for c in line[:3])
            ]
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]
    
    def answer_with_planning(self, query: str) -> Dict[str, Any]:
        """
        Answer query using planning and multi-step reasoning.
        
        Args:
            query: User's query
            
        Returns:
            Comprehensive answer with reasoning steps
        """
        logger.info(f"Processing query with planning: {query}")
        
        # Step 1: Decompose query
        sub_queries = self.decompose_query(query)
        
        # Step 2: Answer each sub-query
        sub_answers = []
        for sub_query in sub_queries:
            result = self.rag_pipeline.query(sub_query, top_k=3)
            sub_answers.append({
                'question': sub_query,
                'answer': result['answer']
            })
        
        # Step 3: Synthesize final answer
        synthesis_prompt = f"""Based on the following sub-questions and answers, provide a comprehensive answer to the original question.

Original Question: {query}

Sub-questions and Answers:
"""
        
        for i, item in enumerate(sub_answers, 1):
            synthesis_prompt += f"\n{i}. Q: {item['question']}\n   A: {item['answer']}\n"
        
        synthesis_prompt += "\n\nProvide a coherent, comprehensive answer:"
        
        try:
            final_response = self.llm.invoke(synthesis_prompt)
            
            return {
                'query': query,
                'sub_queries': sub_queries,
                'sub_answers': sub_answers,
                'final_answer': final_response.content,
                'reasoning_steps': len(sub_queries)
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return {
                'query': query,
                'error': str(e)
            }


def main():
    """Demo/test function."""
    print(f"\n{'='*80}")
    print("RAG Agent Demo")
    print(f"{'='*80}\n")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Initialize agent
    agent = RAGAgent(
        rag_pipeline=rag_pipeline,
        verbose=True
    )
    
    print("Agent initialized successfully!\n")
    
    # Test queries
    test_queries = [
        "What information do you have about machine learning?",
        "Search for documents about artificial intelligence",
        "Explain the key concepts from the available documents"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        response = agent.run(query)
        print(f"Response: {response}\n")
    
    print("="*80)


if __name__ == "__main__":
    main()

