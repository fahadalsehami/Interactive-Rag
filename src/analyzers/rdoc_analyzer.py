"""
rdoc_analyzer.py: RDoC Matrix Analysis Module for Clinical Assessment
"""
from typing import Dict, List, Any, Optional
import re
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Symptom keywords that point to RDoC domains or constructs
KEYWORD_TARGETS = {
    'depression': ['negative valence', 'reward'],
    'anxiety': ['negative valence', 'acute threat'],
    'adhd': ['attention', 'working memory'],
    'headache': ['negative valence', 'cognitive'],
}


def _mentions(text: str, term: str) -> bool:
    """True if term appears in text as a whole word or phrase (case-insensitive)."""
    return re.search(r'\b' + re.escape(term.lower()) + r'\b', text) is not None


class RDoCAnalyzer:
    def __init__(
        self,
        rdoc_matrix: Dict[str, Any],
        kendra_index_id: Optional[str] = None,
        region: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ):
        """Initialize RDoC Analyzer with matrix data and an optional retrieval source."""
        self.matrix = rdoc_matrix

        # Initialize a Bedrock Knowledge Base or a Kendra index if configured
        self.knowledge_base_client = None
        self.knowledge_base_id = knowledge_base_id
        self.kendra_client = None
        self.kendra_index_id = kendra_index_id
        try:
            if knowledge_base_id and region:
                self.knowledge_base_client = boto3.client('bedrock-agent-runtime', region_name=region)
                logger.info("Bedrock Knowledge Base client initialized successfully")
            elif kendra_index_id and region:
                self.kendra_client = boto3.client('kendra', region_name=region)
                logger.info("Kendra client initialized successfully")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to initialize the retrieval client: {str(e)}")

    @property
    def retrieval_source(self) -> Optional[str]:
        """Name of the configured retrieval source, or None."""
        if self.knowledge_base_client:
            return "Amazon Bedrock Knowledge Base"
        if self.kendra_client:
            return "Amazon Kendra"
        return None

    def analyze_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Analyze symptoms and map to RDoC domains."""
        try:
            analysis = {
                domain: [] for domain in self.matrix.keys()
            }

            for symptom in symptoms:
                # Process each domain
                for domain, constructs in self.matrix.items():
                    for match in self._match_symptom_to_domain(symptom, domain, constructs):
                        if match['construct'] not in {m['construct'] for m in analysis[domain]}:
                            analysis[domain].append(match)

            return analysis

        except Exception as e:
            logger.error(f"Error in symptom analysis: {str(e)}")
            return {}

    def _match_symptom_to_domain(self, symptom: str, domain: str, constructs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match symptoms to specific RDoC constructs."""
        matches = []
        for construct_name, data in constructs.items():
            if self._is_relevant(symptom, domain, construct_name, data):
                matches.append({
                    'construct': construct_name,
                    'units': self._get_relevant_units(data),
                    'tests': self._get_recommended_tests(data),
                    'relevance': 'Direct Match'
                })
        return matches

    def _is_relevant(self, symptom: str, domain: str, construct: str, data: Dict[str, Any]) -> bool:
        """Determine if a construct is relevant to a symptom."""
        symptom_lower = symptom.lower()
        construct_lower = construct.lower().replace('_', ' ')
        target = f"{domain} {construct}".lower().replace('_', ' ')

        # Direct construct match
        if _mentions(symptom_lower, construct_lower):
            return True

        # Check molecular markers
        if any(_mentions(symptom_lower, mol) for mol in data.get('molecules', [])):
            return True

        # Check behaviors
        if any(_mentions(symptom_lower, beh) for beh in data.get('behavior', [])):
            return True

        # Check symptom keywords against the domain and construct names
        return any(
            _mentions(symptom_lower, keyword) and any(term in target for term in terms)
            for keyword, terms in KEYWORD_TARGETS.items()
        )

    def _get_relevant_units(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract relevant units of analysis."""
        units = {}
        for key in ['molecules', 'cells', 'circuits', 'behavior']:
            if key in data:
                units[key] = data[key]
        return units

    def _get_recommended_tests(self, data: Dict[str, Any]) -> List[str]:
        """Get recommended tests and paradigms."""
        tests = []
        if 'paradigms' in data:
            tests.extend(data['paradigms'])
        if 'self_report' in data:
            tests.extend(data['self_report'])
        return tests

    def generate_clinical_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Generate clinical recommendations based on analysis."""
        recommendations = []

        try:
            for domain, findings in analysis.items():
                if findings:
                    for finding in findings:
                        recommendations.append({
                            'Domain': domain.replace('_', ' '),
                            'Construct': finding['construct'].replace('_', ' '),
                            'Units_of_Analysis': self._format_units(finding.get('units', {})),
                            'Recommended_Tests': ', '.join(finding.get('tests', [])),
                            'Relevance': finding.get('relevance', 'Direct Match')
                        })

            return pd.DataFrame(recommendations)

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame()

    def _format_units(self, units: Dict[str, List[str]]) -> str:
        """Format units for display."""
        return '; '.join(f"{k}: {', '.join(v)}" for k, v in units.items() if v)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve passages for retrieval-augmented generation from the configured source.

        Returns a list of passages with `title`, `excerpt` and `uri`, or an empty list when no
        source is configured or the call fails.
        """
        if self.knowledge_base_client:
            return self._retrieve_from_knowledge_base(query)
        if self.kendra_client:
            return self._retrieve_from_kendra(query, top_k)
        return []

    def _retrieve_from_knowledge_base(self, query: str) -> List[Dict[str, str]]:
        """Retrieve passages from an Amazon Bedrock Knowledge Base (its default number of results)."""
        try:
            response = self.knowledge_base_client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query[:1000]}
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Knowledge Base retrieval error: {str(e)}")
            return []

        passages = []
        for result in response.get('retrievalResults', []):
            location = result.get('location', {})
            uri = (
                location.get('webLocation', {}).get('url')
                or location.get('s3Location', {}).get('uri')
                or ''
            )
            passages.append({
                'title': uri.rstrip('/').rsplit('/', 1)[-1].replace('-', ' ') if uri else '',
                'excerpt': result.get('content', {}).get('text', ''),
                'uri': uri
            })
        return passages

    def _retrieve_from_kendra(self, query: str, top_k: int) -> List[Dict[str, str]]:
        """Retrieve passages from an Amazon Kendra index. Kendra reads the first 30 words of the query."""
        try:
            response = self.kendra_client.retrieve(
                IndexId=self.kendra_index_id,
                QueryText=query[:1000],
                PageSize=top_k
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Kendra retrieval error: {str(e)}")
            return []

        return [
            {
                'title': item.get('DocumentTitle', ''),
                'excerpt': item.get('Content', ''),
                'uri': item.get('DocumentURI', '')
            }
            for item in response.get('ResultItems', [])
        ]
