"""
userInput.py: Custom tool for gathering additional information in analysis system
"""
import time
from typing import Any, Optional, Dict
from langchain.tools.base import BaseTool
from pydantic import Field
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class InteractiveHumanInput(BaseTool):
    """Tool for gathering additional clinical information."""

    name: str = Field(default="AskClinician")
    description: str = Field(
        default="Use this tool to gather additional clinical information or clarification from the user."
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """
        Execute the tool to gather user input.
        
        Args:
            query (str): Question to ask the user
            run_manager (Optional[Any]): LangChain run manager
            
        Returns:
            str: User's response or status message
        """
        try:
            # Generate unique key for this query
            query_key = f"query_{hash(query)}"
            
            # Check session state for existing response
            if query_key in st.session_state:
                return st.session_state[query_key]

            # Create expandable container for input
            with st.expander("Additional Information Needed", expanded=True):
                st.markdown("""
                ### Follow-up Question
                To provide a more accurate analysis, please provide additional information.
                """)
                
                st.markdown(f"**Question:** {query}")
                
                # Input area
                user_input = st.text_area(
                    "Your Response:",
                    key=f"input_{query_key}",
                    height=100,
                    help="Please provide detailed information to help with the analysis"
                )
                
                # Submit button
                if st.button("Submit Response", key=f"submit_{query_key}"):
                    if user_input:
                        # Store in session state
                        st.session_state[query_key] = user_input
                        # Update conversation history if available
                        self._update_conversation_history(query, user_input)
                        logger.info(f"Received response for query: {query_key}")
                        return user_input
                    else:
                        st.warning("Please provide a response before submitting.")
                        return "No response provided"
            
            time.sleep(0.1)  # Small delay for UI
            return "Waiting for additional information..."
            
        except Exception as e:
            logger.error(f"Error in InteractiveHumanInput: {str(e)}")
            return f"Error gathering information: {str(e)}"

    def _update_conversation_history(self, question: str, answer: str):
        """
        Update conversation history in session state.
        
        Args:
            question (str): Asked question
            answer (str): User's response
        """
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        st.session_state.conversation_history.extend([
            {'role': 'assistant', 'content': question},
            {'role': 'user', 'content': answer}
        ])

    def get_conversation_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dict[str, Any]: Conversation context
        """
        if 'conversation_history' not in st.session_state:
            return {'history': [], 'last_query': None}
            
        history = st.session_state.conversation_history
        last_query = history[-1]['content'] if history else None
        
        return {
            'history': history,
            'last_query': last_query
        }

    async def _arun(self, query: str) -> str:
        """
        Async execution is not supported.
        
        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("Async execution is not supported")