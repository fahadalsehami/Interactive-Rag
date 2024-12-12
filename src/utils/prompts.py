"""
prompt.py: System prompts for the Clinical Analysis System
"""
from langchain.prompts import PromptTemplate

def create_analysis_prompt() -> PromptTemplate:
    """
    Create the main analysis prompt.
    
    Returns:
        PromptTemplate: Main analysis prompt
    """
    template = """Analyze the following clinical information using the RDoC framework:

Clinical Information:
{context}

Based on the provided information and retrieved knowledge, analyze in this sequence:

1. Initial Assessment:
   - Primary symptoms and presentations
   - Associated clinical domains
   - Required additional information

2. Domain Analysis:
   For each relevant domain:
   {domain_info}

3. Units of Analysis:
   a) Molecular Level:
      - Relevant biomarkers
      - Neurotransmitter systems
   
   b) Circuit Level:
      - Key neural circuits
      - Brain regions involved
   
   c) Behavioral Level:
      - Observable patterns
      - Functional impacts
   
   d) Assessment Measures:
      - Recommended tests
      - Clinical scales

4. Clinical Recommendations:
   - Suggested assessments
   - Priority areas
   - Clinical considerations

Question: {question}

Please provide structured findings and recommendations."""

    return PromptTemplate(
        template=template,
        input_variables=["context", "domain_info", "question"]
    )

def create_followup_prompt() -> PromptTemplate:
    """
    Create prompt for gathering additional information.
    
    Returns:
        PromptTemplate: Follow-up prompt
    """
    template = """Based on the clinical presentation:

{initial_info}

What additional information would be helpful? Consider:

1. Symptom characteristics
   - Onset and duration
   - Severity and frequency
   - Pattern of occurrence

2. Functional Impact
   - Daily activities
   - Social functioning
   - Occupational impact

3. History
   - Previous treatments
   - Medical history
   - Family history

4. Current Status
   - Recent changes
   - Current medications
   - Support systems

Focus areas:
{focus_areas}

Generate a specific follow-up question to gather the most important missing information.

Question: {specific_query}"""

    return PromptTemplate(
        template=template,
        input_variables=["initial_info", "focus_areas", "specific_query"]
    )

def create_summary_prompt() -> PromptTemplate:
    """
    Create prompt for summarizing findings.
    
    Returns:
        PromptTemplate: Summary prompt
    """
    template = """Summarize the clinical analysis findings:

Analysis Results:
{analysis_results}

Provide a concise summary including:
1. Key domains affected
2. Primary mechanisms identified
3. Recommended assessments
4. Clinical priorities

Summary:"""

    return PromptTemplate(
        template=template,
        input_variables=["analysis_results"]
    )

def create_integration_prompt() -> PromptTemplate:
    """
    Create prompt for integrating multiple information sources.
    
    Returns:
        PromptTemplate: Integration prompt
    """
    template = """Integrate the following information sources:

Initial Presentation:
{initial_presentation}

Follow-up Information:
{followup_info}

Knowledge Base Results:
{knowledge_base_results}

Provide an integrated analysis that:
1. Synthesizes all information sources
2. Identifies key patterns and relationships
3. Highlights critical areas for attention
4. Suggests comprehensive next steps

Integrated Analysis:"""

    return PromptTemplate(
        template=template,
        input_variables=["initial_presentation", "followup_info", "knowledge_base_results"]
    )