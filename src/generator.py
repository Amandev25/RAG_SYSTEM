"""
Generator Module
Handles answer generation using LLMs (OpenAI GPT models).
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI not installed. Install with: pip install openai")


class AnswerGenerator:
    """
    Generates answers using LLMs based on retrieved context.
    """
    
    # Default prompts
    DEFAULT_SYSTEM_PROMPT = """You are an AI assistant helping users find information from documents.
Use the provided context to answer questions accurately and concisely.
If you cannot answer from the context, say so clearly.
Always cite your sources when possible."""
    
    DEFAULT_QA_TEMPLATE = """Context:
{context}

Question: {question}

Based on the context provided above, please answer the question. If the answer cannot be found in the context, please state that clearly."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        qa_template: Optional[str] = None
    ):
        """
        Initialize Answer Generator.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            temperature: Generation temperature (0-2)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
            qa_template: Custom QA template
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Set prompts
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.qa_template = qa_template or self.DEFAULT_QA_TEMPLATE
        
        logger.info(f"AnswerGenerator initialized with model: {model}")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer based on question and context.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            include_sources: Whether to extract and include sources
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating answer for question: {question[:50]}...")
        
        # Format prompt
        user_prompt = self.qa_template.format(
            context=context,
            question=question
        )
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Build response
            result = {
                'answer': answer,
                'question': question,
                'model': self.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
            if include_sources:
                result['sources'] = self._extract_sources(context)
            
            logger.info(f"Generated answer ({response.usage.completion_tokens} tokens)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'question': question,
                'error': str(e)
            }
    
    def generate_streaming_answer(
        self,
        question: str,
        context: str
    ):
        """
        Generate an answer with streaming (for real-time display).
        
        Args:
            question: User's question
            context: Retrieved context
            
        Yields:
            Chunks of the generated answer
        """
        logger.info(f"Generating streaming answer for: {question[:50]}...")
        
        # Format prompt
        user_prompt = self.qa_template.format(
            context=context,
            question=question
        )
        
        try:
            # Call OpenAI API with streaming
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error: {str(e)}"
    
    def _extract_sources(self, context: str) -> List[Dict[str, str]]:
        """
        Extract source information from context.
        
        Args:
            context: Formatted context string
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        # Parse context for document markers
        lines = context.split('\n')
        current_source = None
        
        for line in lines:
            if line.startswith('[Document') and 'Source:' in line:
                # Extract source information
                try:
                    source_part = line.split('Source:')[1].strip().rstrip(')')
                    current_source = {
                        'source': source_part,
                        'document_marker': line.split(']')[0] + ']'
                    }
                    sources.append(current_source)
                except:
                    pass
        
        return sources
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        logger.info("Generating summary...")
        
        prompt = f"""Please provide a concise summary of the following text in about {max_length} words:

{text}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_length * 2  # Rough token estimate
            )
            
            summary = response.choices[0].message.content
            logger.info("Summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Context used
            
        Returns:
            Quality assessment
        """
        eval_prompt = f"""Evaluate the following answer to a question based on the provided context.

Question: {question}

Context: {context}

Answer: {answer}

Please evaluate:
1. Relevance: Does the answer address the question?
2. Accuracy: Is the answer consistent with the context?
3. Completeness: Does the answer fully address the question?

Provide a brief assessment and a score from 1-10."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of question-answering systems."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            evaluation = response.choices[0].message.content
            
            return {
                'evaluation': evaluation,
                'timestamp': logger.info("Answer quality evaluated")
            }
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {'error': str(e)}


def main():
    """Demo/test function."""
    print(f"\n{'='*80}")
    print("Answer Generator Demo")
    print(f"{'='*80}\n")
    
    # Sample context
    context = """[Document 1] (Source: AI_Textbook.pdf)
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

[Document 2] (Source: ML_Guide.pdf)
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."""
    
    # Initialize generator
    generator = AnswerGenerator(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Test questions
    questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "How does machine learning work?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)
        
        result = generator.generate_answer(question, context)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nTokens used: {result['usage']['total_tokens']}")
        
        if result.get('sources'):
            print("\nSources:")
            for source in result['sources']:
                print(f"  - {source['source']}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()

