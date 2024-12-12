"""
rdoc_analyzer.py: RDoC Matrix Analysis Module for Clinical Assessment
"""
from typing import Dict, List, Any, Optional
import boto3
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RDoCAnalyzer:
    def __init__(self, rdoc_matrix: Dict[str, Any], kendra_index_id: Optional[str] = None, region: Optional[str] = None):
        """Initialize RDoC Analyzer with matrix data and optional Kendra config."""
        self.matrix = rdoc_matrix
        
        # Initialize Kendra if configured
        self.kendra_client = None
        self.kendra_index_id = kendra_index_id
        if kendra_index_id and region:
            try:
                self.kendra_client = boto3.client('kendra', region_name=region)
                logger.info("Kendra client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Kendra client: {str(e)}")

    def analyze_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Analyze symptoms and map to RDoC domains."""
        try:
            analysis = {
                domain: [] for domain in self.matrix.keys()
            }
            
            for symptom in symptoms:
                # Process each domain
                for domain, constructs in self.matrix.items():
                    matches = self._match_symptom_to_domain(symptom, constructs)
                    if matches:
                        analysis[domain].extend(matches)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in symptom analysis: {str(e)}")
            return {}

    def _match_symptom_to_domain(self, symptom: str, constructs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match symptoms to specific RDoC constructs."""
        matches = []
        for construct_name, data in constructs.items():
            if self._is_relevant(symptom, construct_name, data):
                matches.append({
                    'construct': construct_name,
                    'units': self._get_relevant_units(data),
                    'tests': self._get_recommended_tests(data),
                    'relevance': 'Direct Match'
                })
        return matches

    def _is_relevant(self, symptom: str, construct: str, data: Dict[str, Any]) -> bool:
        """Determine if a construct is relevant to a symptom."""
        symptom_lower = symptom.lower()
        construct_lower = construct.lower().replace('_', ' ')
        
        # Direct construct match
        if construct_lower in symptom_lower:
            return True
            
        # Check molecular markers
        if 'molecules' in data and any(mol.lower() in symptom_lower for mol in data['molecules']):
            return True
            
        # Check behaviors
        if 'behavior' in data and any(beh.lower() in symptom_lower for beh in data['behavior']):
            return True
            
        # Check domains
        domains = {
            'depression': ['negative_valence', 'reward'],
            'anxiety': ['negative_valence', 'acute_threat'],
            'adhd': ['attention', 'working_memory'],
            'headache': ['negative_valence', 'cognitive'],
        }
        
        for keyword, related_domains in domains.items():
            if keyword in symptom_lower:
                return any(domain in construct_lower for domain in related_domains)
        
        return False

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
                            'Domain': domain,
                            'Construct': finding['construct'],
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

    def _search_kendra(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Kendra index if available."""
        if not self.kendra_client or not self.kendra_index_id:
            return None
            
        try:
            response = self.kendra_client.query(
                IndexId=self.kendra_index_id,
                QueryText=query
            )
            
            results = []
            for item in response.get('ResultItems', []):
                if item['Type'] == 'DOCUMENT':
                    results.append({
                        'title': item.get('DocumentTitle', {}).get('Text', ''),
                        'excerpt': item.get('DocumentExcerpt', {}).get('Text', ''),
                        'confidence': item.get('ScoreAttributes', {}).get('ScoreConfidence', '')
                    })
            
            return {'kendra_results': results} if results else None
            
        except Exception as e:
            logger.error(f"Kendra search error: {str(e)}")
            return None