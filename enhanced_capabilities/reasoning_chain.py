# Create a new file - enhanced_capabilities/reasoning_chain.py

from typing import List, Dict, Any
import json

class ReasoningChain:
    """Implementation of a multi-step reasoning chain for complex questions"""
    
    def __init__(self, retrieval_engine, prompt_engine):
        self.retrieval_engine = retrieval_engine
        self.prompt_engine = prompt_engine
    
    def solve(self, question: str, max_steps: int = 3) -> Dict[str, Any]:
        """Solve a complex question using multi-step reasoning"""
        # Start with initial reasoning step
        reasoning = []
        current_question = question
        
        for step in range(max_steps):
            # 1. Think about what information is needed
            thinking_prompt = f"To answer '{current_question}', what information do I need to find? Think step by step."
            
            try:
                info_needed = self.prompt_engine.generate_response(
                    query=thinking_prompt,
                    context=[],
                    options={"max_tokens": 200}
                )
            except Exception as e:
                print(f"Error in generating thinking step: {e}")
                info_needed = "I need to search for relevant information about this topic."
            
            # 2. Search for relevant information
            try:
                context_docs = self.retrieval_engine.retrieve_context(
                    query=current_question + " " + info_needed,
                    top_k=5
                )
            except Exception as e:
                print(f"Error in retrieving context: {e}")
                context_docs = []
            
            # 3. Reason with the information
            reasoning_prompt = f"""
            Question: {current_question}
            Information needed: {info_needed}
            
            Based on the information I have, I can reason as follows:
            """
            
            try:
                reasoning_step = self.prompt_engine.generate_response(
                    query=reasoning_prompt,
                    context=context_docs,
                    options={"max_tokens": 300}
                )
            except Exception as e:
                print(f"Error in reasoning step: {e}")
                reasoning_step = "Based on the available information, I can partially answer this question."
            
            reasoning.append({
                "step": step + 1,
                "info_needed": info_needed,
                "reasoning": reasoning_step
            })
            
            # 4. Determine if we need more information or can answer
            next_step_prompt = f"""
            Based on my reasoning so far:
            {reasoning_step}
            
            Do I have enough information to answer the original question: "{question}"?
            If yes, what is the answer?
            If no, what specific additional information do I need to find?
            """
            
            try:
                next_step = self.prompt_engine.generate_response(
                    query=next_step_prompt,
                    context=context_docs,
                    options={"max_tokens": 200}
                )
            except Exception as e:
                print(f"Error in next step determination: {e}")
                next_step = "Yes, I can now answer the question."
            
            if "yes" in next_step.lower() and "answer" in next_step.lower():
                # We have enough information to answer
                try:
                    final_answer = self.prompt_engine.generate_response(
                        query=f"Final answer to: {question}",
                        context=context_docs,
                        options={"max_tokens": 500}
                    )
                except Exception as e:
                    print(f"Error in final answer generation: {e}")
                    final_answer = "Based on my analysis, I can provide a partial answer to your question."
                
                return {
                    "answer": final_answer,
                    "reasoning_chain": reasoning,
                    "steps_taken": step + 1
                }
            
            # Extract the new question from next_step
            if "additional information" in next_step.lower():
                lines = next_step.split('\n')
                for line in lines:
                    if "?" in line:
                        current_question = line.strip()
                        break
        
        # If we reached max steps without a conclusive answer
        try:
            final_attempt = self.prompt_engine.generate_response(
                query=f"Based on my reasoning so far, my best answer to '{question}' is:",
                context=context_docs,
                options={"max_tokens": 500}
            )
        except Exception as e:
            print(f"Error in final attempt: {e}")
            final_attempt = "I don't have enough information to fully answer your question."
        
        return {
            "answer": final_attempt,
            "reasoning_chain": reasoning,
            "steps_taken": max_steps,
            "status": "incomplete"
        }