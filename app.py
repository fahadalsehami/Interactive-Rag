"""
app.py: Streamlit app for clinical analysis with the RDoC framework and retrieval-augmented generation
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.analyzers.rdoc_analyzer import RDoCAnalyzer
from src.utils.model import (
    aws_region,
    generate_text,
    get_kendra_config,
    get_knowledge_base_config,
    model_id,
    validate_aws_credentials,
)
from src.utils.prompts import SYSTEM_PROMPT, create_analysis_prompt, create_followup_prompt

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
        background: rgba(33, 150, 243, 0.08);
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Copy settings from .streamlit/secrets.toml into the environment, without overriding variables already set
if (project_root / ".streamlit" / "secrets.toml").exists():
    for setting, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(setting, value)

# Initialize session state
for key, default in (
    ('conversation_history', []),
    ('current_analysis', None),
    ('grounded_analysis', None),
    ('follow_up_questions', []),
):
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def load_rdoc_matrix() -> Optional[Dict]:
    """Load RDoC matrix data."""
    matrix_path = project_root / "src" / "data" / "rdoc_matrix.json"
    try:
        with open(matrix_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded RDoC matrix from {matrix_path}")
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error loading RDoC matrix: {str(e)}")
        return None


@st.cache_resource
def connect_services() -> Dict[str, Any]:
    """Check AWS once per server process. Without credentials the app runs on the RDoC matrix alone."""
    aws_ready = validate_aws_credentials()
    knowledge_base = get_knowledge_base_config() if aws_ready else None
    kendra = get_kendra_config() if aws_ready and not knowledge_base else None
    return {'aws': aws_ready, 'knowledge_base': knowledge_base, 'kendra': kendra}


@st.cache_resource
def initialize_analyzer() -> Optional[RDoCAnalyzer]:
    """Initialize the RDoC analyzer, with retrieval when a knowledge base or Kendra index is configured."""
    rdoc_matrix = load_rdoc_matrix()
    if not rdoc_matrix:
        return None

    services = connect_services()
    if services['knowledge_base']:
        logger.info("Initializing analyzer with a Bedrock Knowledge Base")
        return RDoCAnalyzer(
            rdoc_matrix=rdoc_matrix,
            knowledge_base_id=services['knowledge_base']['knowledge_base_id'],
            region=services['knowledge_base']['region']
        )
    if services['kendra']:
        logger.info("Initializing analyzer with Kendra integration")
        return RDoCAnalyzer(
            rdoc_matrix=rdoc_matrix,
            kendra_index_id=services['kendra']['index_id'],
            region=services['kendra']['region']
        )

    logger.info("Initializing basic analyzer without retrieval")
    return RDoCAnalyzer(rdoc_matrix=rdoc_matrix)


def user_text() -> str:
    """Everything the clinician has written so far: the presentation and any follow-up answers."""
    return "\n".join(
        entry['content'] for entry in st.session_state.conversation_history if entry['role'] == 'user'
    )


def conversation_text() -> str:
    """The full exchange, including follow-up questions, for the language model."""
    return "\n".join(
        f"{'Follow-up question' if entry['role'] == 'assistant' else 'Clinician'}: {entry['content']}"
        for entry in st.session_state.conversation_history
    )


def matched_constructs(analysis: Dict[str, Any]) -> List[str]:
    """Names of the constructs the RDoC matrix matched, in order."""
    return [
        finding['construct'].replace('_', ' ')
        for findings in analysis.values()
        for finding in findings
    ]


def describe_matches(analysis: Dict[str, Any]) -> str:
    """RDoC matches as plain text for the prompt."""
    lines = []
    for domain, findings in analysis.items():
        for finding in findings:
            tests = ', '.join(finding.get('tests', []))
            lines.append(
                f"- {domain.replace('_', ' ')} / {finding['construct'].replace('_', ' ')}: "
                f"{format_units(finding.get('units', {}), separator='; ')}. Tests: {tests}"
            )
    return "\n".join(lines) if lines else "- No construct in the RDoC matrix matched the presentation directly."


def describe_passages(passages: List[Dict[str, str]]) -> str:
    """Retrieved passages as numbered text for the prompt."""
    if not passages:
        return "No passages were retrieved."
    return "\n\n".join(
        f"[{number}] {passage['title']}\n{passage['excerpt']}"
        for number, passage in enumerate(passages, start=1)
    )


def run_grounded_analysis(analyzer: RDoCAnalyzer, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retrieve passages and ask Claude on Bedrock for an analysis and one follow-up question."""
    if not connect_services()['aws']:
        return None

    constructs = matched_constructs(analysis)
    query = " ".join(["RDoC"] + constructs + [user_text()])
    passages = analyzer.retrieve(query)

    context = f"{conversation_text()}\n\nRetrieved passages:\n{describe_passages(passages)}"
    analysis_prompt = create_analysis_prompt().format(
        context=context,
        domain_info=describe_matches(analysis),
        question="Which RDoC constructs and units of analysis fit this presentation, and what should be assessed next?"
    )
    followup_prompt = create_followup_prompt().format(
        initial_info=conversation_text(),
        focus_areas=", ".join(constructs) if constructs else "the RDoC domains most likely to be involved",
        specific_query="Write exactly one follow-up question for the clinician, in one sentence, with nothing before or after it."
    )

    try:
        text = generate_text(analysis_prompt, system=SYSTEM_PROMPT)
        follow_up = generate_text(followup_prompt, system=SYSTEM_PROMPT, max_tokens=200)
    except ClientError as e:
        code = e.response['Error'].get('Code', 'error')
        logger.error(f"Bedrock error: {e}")
        st.warning(
            f"Claude on Amazon Bedrock returned {code}. Check that model access for `{model_id()}` "
            f"is enabled in {aws_region()}. The results below come from the RDoC matrix alone."
        )
        return None
    except BotoCoreError as e:
        logger.error(f"Bedrock error: {e}")
        st.warning("Amazon Bedrock could not be reached. The results below come from the RDoC matrix alone.")
        return None

    return {'text': text, 'sources': passages, 'follow_up': follow_up}


def run_analysis(analyzer: RDoCAnalyzer):
    """Analyze everything the clinician has written and store the results in the session."""
    analysis = analyzer.analyze_symptoms([user_text()])
    grounded = run_grounded_analysis(analyzer, analysis)
    st.session_state.current_analysis = analysis
    st.session_state.grounded_analysis = grounded
    st.session_state.follow_up_questions = [grounded['follow_up']] if grounded and grounded['follow_up'] else []


def display_conversation_history():
    """Display the conversation history."""
    if not st.session_state.conversation_history:
        return

    st.subheader("Conversation")
    for entry in st.session_state.conversation_history:
        if entry['role'] == 'user':
            st.info(f"👤 Clinician: {entry['content']}")
        elif entry['role'] == 'assistant':
            st.success(f"🤖 Follow-up question: {entry['content']}")


def display_grounded_analysis(grounded: Dict[str, Any]):
    """Display Claude's analysis and the passages it was given."""
    st.subheader("Grounded analysis")
    st.markdown(grounded['text'])
    if grounded['sources']:
        st.markdown("**Retrieved passages**")
        for number, passage in enumerate(grounded['sources'], start=1):
            title = passage['title'] or passage['uri'] or f"Passage {number}"
            st.markdown(f"[{number}] [{title}]({passage['uri']})" if passage['uri'] else f"[{number}] {title}")
    else:
        st.caption("No passages were retrieved, so this analysis uses the RDoC matrix and the conversation only.")


def handle_follow_up(analyzer: RDoCAnalyzer):
    """Ask the latest follow-up question and re-run the analysis with the answer."""
    if not st.session_state.follow_up_questions:
        return

    question = st.session_state.follow_up_questions[-1]
    with st.form("follow_up", clear_on_submit=True):
        st.markdown(f"**Follow-up question:** {question}")
        answer = st.text_area("Your answer", height=100)
        submitted = st.form_submit_button("Add to the analysis")

    if submitted:
        if not answer.strip():
            st.warning("Write an answer before adding it to the analysis.")
            return
        st.session_state.conversation_history.extend([
            {'role': 'assistant', 'content': question},
            {'role': 'user', 'content': answer.strip()}
        ])
        with st.spinner("Updating the analysis..."):
            run_analysis(analyzer)
        st.rerun()


def process_analysis_results(analysis: Dict[str, Any], analyzer: RDoCAnalyzer):
    """Process and display analysis results."""
    try:
        if not analysis:
            st.warning("No analysis results available.")
            return

        st.subheader("Analysis Results")

        if not any(analysis.values()):
            st.info(
                "No construct in the RDoC matrix matched this description. "
                "Try naming symptoms or behaviors such as anxiety, avoidance, motivation or attention."
            )
            return

        # Display results by domain
        for domain, findings in analysis.items():
            if findings:
                with st.expander(domain.replace('_', ' '), expanded=True):
                    for finding in findings:
                        st.markdown(
                            f"""<div class="clinical-finding">
                            <strong>🎯 Construct:</strong> {finding['construct'].replace('_', ' ')}<br><br>
                            <strong>🧪 Units of Analysis:</strong><br>
                            {format_units(finding.get('units', {}))}<br><br>
                            <strong>📋 Recommended Tests:</strong><br>
                            {', '.join(finding.get('tests') or ['No specific tests recommended'])}<br>
                            </div>""",
                            unsafe_allow_html=True
                        )

        # Display recommendations
        recommendations = analyzer.generate_clinical_recommendations(analysis)
        if not recommendations.empty:
            st.subheader("Clinical Recommendations")
            st.dataframe(
                recommendations,
                width="stretch",
                hide_index=True,
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


def format_units(units: Dict[str, list], separator: str = "<br>") -> str:
    """Format units for display."""
    formatted_units = []
    for category, items in units.items():
        if items:
            formatted_units.append(f"• {category.title()}: {', '.join(items)}")
    return separator.join(formatted_units) if formatted_units else "No specific units identified"


def main():
    st.title("🧬 Clinical Analysis System")

    # Initialize analyzer
    analyzer = initialize_analyzer()
    if not analyzer:
        st.error("Failed to load the knowledge base: make sure src/data/rdoc_matrix.json exists and is valid JSON.")
        return
    services = connect_services()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        Maps a clinical presentation to NIMH Research Domain Criteria (RDoC) constructs.
        With AWS configured, a knowledge base retrieves supporting passages and Claude on
        Amazon Bedrock writes a grounded analysis and asks a follow-up question.
        """)

        st.header("Status")
        st.markdown(f"""
        - 🧠 RDoC matrix: loaded
        - 🔍 Retrieval: {analyzer.retrieval_source or 'not configured'}
        - 💬 Claude on Bedrock: {'ready' if services['aws'] else 'not configured'}
        """)

        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.clear()
            st.rerun()

        st.caption("Research prototype for clinical decision support. It does not diagnose.")

    # Main interface
    if not st.session_state.current_analysis:
        clinical_input = st.text_area(
            "Enter Clinical Presentation",
            placeholder="Describe the presentation: symptoms, onset, duration and impact on daily life...",
            height=150,
            key="clinical_input"
        )

        if st.button("🔍 Analyze", type="primary"):
            if clinical_input.strip():
                st.session_state.conversation_history = [{'role': 'user', 'content': clinical_input.strip()}]
                with st.spinner("Analyzing presentation..."):
                    run_analysis(analyzer)
                st.rerun()
            else:
                st.warning("Please enter a clinical presentation.")
        return

    # Display the conversation, the results and the follow-up question
    display_conversation_history()
    process_analysis_results(st.session_state.current_analysis, analyzer)
    if st.session_state.grounded_analysis:
        display_grounded_analysis(st.session_state.grounded_analysis)
    handle_follow_up(analyzer)

    if st.button("🔄 Start New Analysis", type="primary"):
        st.session_state.conversation_history = []
        st.session_state.current_analysis = None
        st.session_state.grounded_analysis = None
        st.session_state.follow_up_questions = []
        st.rerun()


if __name__ == "__main__":
    main()
