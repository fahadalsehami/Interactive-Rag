"""
app.py: Main Streamlit application for Clinical Analysis using RDoC framework with RAG capabilities
"""
import sys
from pathlib import Path
import json
import streamlit as st
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.analyzers.rdoc_analyzer import RDoCAnalyzer
from src.utils.userInput import InteractiveHumanInput
from src.utils.model import get_kendra_config, validate_aws_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(
    page_title="Clinical Analysis System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .clinical-finding {
        padding: 1rem;
        border-left: 3px solid #2196F3;
        background: #000000;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .recommendation {
        padding: 1rem;
        border-left: 3px solid #4CAF50;
        background: #f8f9fa;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .warning {
        padding: 1rem;
        border-left: 3px solid #FFC107;
        background: #fff3e0;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'follow_up_questions' not in st.session_state:
    st.session_state.follow_up_questions = []

def check_directory_structure():
    """Verify and create necessary directories."""
    try:
        data_dir = project_root / "src" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to verify directory structure: {e}")
        return False

@st.cache_resource
def load_rdoc_matrix() -> Optional[Dict]:
    """Load RDoC matrix data."""
    try:
        if not check_directory_structure():
            raise Exception("Failed to verify directory structure")
            
        matrix_path = project_root / "src" / "data" / "rdoc_matrix.json"
        
        if not matrix_path.exists():
            raise FileNotFoundError(f"RDoC matrix file not found at {matrix_path}")
        
        with open(matrix_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded RDoC matrix from {matrix_path}")
            return data
            
    except Exception as e:
        logger.error(f"Error loading RDoC matrix: {str(e)}")
        st.error(f"Failed to load knowledge base: Please ensure rdoc_matrix.json exists in src/data directory")
        return None

def initialize_analyzer() -> Optional[RDoCAnalyzer]:
    """Initialize the RDoC analyzer with optional Kendra integration."""
    try:
        rdoc_matrix = load_rdoc_matrix()
        if not rdoc_matrix:
            return None
            
        # Check AWS credentials if using Kendra
        if validate_aws_credentials():
            kendra_config = get_kendra_config()
            if kendra_config:
                logger.info("Initializing analyzer with Kendra integration")
                return RDoCAnalyzer(
                    rdoc_matrix=rdoc_matrix,
                    kendra_index_id=kendra_config['index_id'],
                    region=kendra_config['region']
                )
        
        # Fallback to basic analyzer
        logger.info("Initializing basic analyzer without Kendra")
        return RDoCAnalyzer(rdoc_matrix=rdoc_matrix)
            
    except Exception as e:
        logger.error(f"Error initializing analyzer: {str(e)}")
        st.error(f"Failed to initialize analyzer: {str(e)}")
        return None

def display_conversation_history():
    """Display the conversation history."""
    if not st.session_state.conversation_history:
        return
        
    st.subheader("Conversation History")
    for entry in st.session_state.conversation_history:
        if entry['role'] == 'user':
            st.info(f"👤 User: {entry['content']}")
        elif entry['role'] == 'assistant':
            st.success(f"🤖 Assistant: {entry['content']}")

def handle_follow_up(analyzer: RDoCAnalyzer):
    """Handle follow-up questions and responses."""
    ask_human_tool = InteractiveHumanInput()
    
    if st.session_state.follow_up_questions:
        with st.expander("Additional Information Needed", expanded=True):
            response = ask_human_tool.run(st.session_state.follow_up_questions[-1])
            if response and response != "Waiting for additional information...":
                # Update conversation history
                st.session_state.conversation_history.append({
                    'role': 'user',
                    'content': response
                })
                
                # Get complete context
                complete_context = "\n".join([
                    entry['content'] 
                    for entry in st.session_state.conversation_history
                ])
                
                # Process updated context
                with st.spinner("Analyzing updated information..."):
                    analysis = analyzer.analyze_symptoms([complete_context])
                    st.session_state.current_analysis = analysis
                    process_analysis_results(analysis, analyzer)

def process_analysis_results(analysis: Dict[str, Any], analyzer: RDoCAnalyzer):
    """Process and display analysis results."""
    try:
        if not analysis:
            st.warning("No analysis results available.")
            return
            
        # Generate recommendations
        recommendations = analyzer.generate_clinical_recommendations(analysis)
        
        # Display results by domain
        st.subheader("Analysis Results")
        
        for domain, findings in analysis.items():
            if findings:
                with st.expander(f"{domain}", expanded=True):
                    for finding in findings:
                        st.markdown(
                            f"""<div class="clinical-finding">
                            <strong>🎯 Construct:</strong> {finding['construct']}<br><br>
                            <strong>🧪 Units of Analysis:</strong><br>
                            {format_units(finding.get('units', {}))}<br><br>
                            <strong>📋 Recommended Tests:</strong><br>
                            {', '.join(finding.get('tests', ['No specific tests recommended']))}<br>
                            </div>""",
                            unsafe_allow_html=True
                        )
        
        # Display recommendations
        if not recommendations.empty:
            st.subheader("Clinical Recommendations")
            st.dataframe(
                recommendations,
                use_container_width=True,
                column_config={
                    "Domain": st.column_config.TextColumn("Domain", width="medium"),
                    "Construct": st.column_config.TextColumn("Construct", width="medium"),
                    "Units_of_Analysis": st.column_config.TextColumn("Units of Analysis", width="large"),
                    "Recommended_Tests": st.column_config.TextColumn("Recommended Tests", width="large"),
                    "Relevance": st.column_config.TextColumn("Relevance", width="small")
                }
            )
            
    except Exception as e:
        logger.error(f"Error processing analysis results: {str(e)}")
        st.error("Error displaying analysis results")

def format_units(units: Dict[str, list]) -> str:
    """Format units for display."""
    formatted_units = []
    for category, items in units.items():
        if items:
            formatted_units.append(f"• {category.title()}: {', '.join(items)}")
    return "<br>".join(formatted_units) if formatted_units else "No specific units identified"

def main():
    st.title("🧬 Clinical Analysis System")
    
    # Initialize analyzer
    analyzer = initialize_analyzer()
    if not analyzer:
        return

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system analyzes clinical presentations using the RDoC framework
        with Retrieval-Augmented Generation (RAG) capabilities.
        """)
        
        st.header("Features")
        st.markdown("""
        - 🧠 RDoC framework analysis
        - 💬 Interactive follow-up questions
        - 🔍 AWS Kendra integration (if configured)
        - 📊 Comprehensive recommendations
        """)
        
        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # Main interface
    with st.container():
        if not st.session_state.current_analysis:
            clinical_input = st.text_area(
                "Enter Clinical Presentation",
                placeholder="Describe the clinical presentation in detail...",
                height=150,
                key="clinical_input"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔍 Analyze", type="primary", use_container_width=True):
                    if clinical_input:
                        with st.spinner("Analyzing presentation..."):
                            # Update conversation history
                            st.session_state.conversation_history.append({
                                'role': 'user',
                                'content': clinical_input
                            })
                            
                            # Perform analysis
                            analysis = analyzer.analyze_symptoms([clinical_input])
                            st.session_state.current_analysis = analysis
                            
                            # Process results
                            process_analysis_results(analysis, analyzer)
                    else:
                        st.warning("Please enter a clinical presentation.")

        # Display conversation and handle follow-ups
        display_conversation_history()
        handle_follow_up(analyzer)
        
        # Display current analysis
        if st.session_state.current_analysis:
            process_analysis_results(st.session_state.current_analysis, analyzer)
            
            if st.button("🔄 Start New Analysis", type="primary"):
                st.session_state.current_analysis = None
                st.session_state.follow_up_questions = []
                st.rerun()

if __name__ == "__main__":
    main()