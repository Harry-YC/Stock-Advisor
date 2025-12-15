"""
Priors Manager for Preclinical/Translational Literature Review
Loads and formats canonical drug development frameworks as Bayesian priors
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import logging
import math

logger = logging.getLogger(__name__)


class PriorsManager:
    """
    Manages canonical drug development frameworks (priors) for expert panel discussions.

    Loads frameworks from YAML config and formats them for inclusion in
    expert consultation prompts.

    Usage:
        pm = PriorsManager()
        priors_text = pm.format_priors_for_context(
            scenario="target_validation",
            persona="Bioscience Lead"
        )
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize PriorsManager

        Args:
            config_path: Path to priors config YAML. If None, uses default location.
        """
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "priors" / "preclinical_priors.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Detect schema version
        self.schema_version = self.config.get('schema_version', '1.0')

        # Extract sections (supports v1 and v2 schemas)
        if self.schema_version == '2.0':
            self.tier1_frameworks = self._extract_tier1()
            self.tier2_scenarios = self.config.get('tier_2_scenario_specific', {})
            self.tier3_frameworks = self._extract_tier3()
            self.scenarios = {}
            self.frameworks = {}
        else:
            self.scenarios = self.config.get('priors_library', {}).get('scenarios', {})
            self.frameworks = self.config.get('priors_library', {}).get('frameworks', {})
            self.tier1_frameworks = {}
            self.tier2_scenarios = {}
            self.tier3_frameworks = {}

        self.persona_defaults = self.config.get('persona_framework_defaults', {})
        self.persona_defaults = self.config.get('persona_framework_defaults', {})
        self.metadata = self.config.get('metadata', {})
        
        # Cache for framework embeddings
        self._framework_embeddings = {}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve_relevant_frameworks(
        self,
        query: str,
        embedder: Any,
        top_k: int = 3,
        threshold: float = 0.4
    ) -> List[str]:
        """
        Retrieve frameworks relevant to a query using semantic search.
        
        Args:
            query: The research question or topic
            embedder: Embedding model (must have embed_query and embed_documents)
            top_k: Max number of frameworks to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of framework IDs
        """
        if not query or not embedder:
            return []
            
        try:
            # 1. Embed query
            query_embedding = embedder.embed_query(query)
            
            # 2. Ensure frameworks are embedded (lazy load)
            all_ids = self.get_all_framework_ids()
            ids_to_embed = [fid for fid in all_ids if fid not in self._framework_embeddings]
            
            if ids_to_embed:
                texts = []
                for fid in ids_to_embed:
                    # Use summary and title for embedding
                    fw = self.get_framework_details(fid)
                    if fw:
                        summary = fw.get('summary', '')
                        title = fw.get('citation', {}).get('title', '')
                        text = f"{title}. {summary}"
                        texts.append(text)
                    else:
                        texts.append("")
                
                if texts:
                    embeddings = embedder.embed_documents(texts)
                    for fid, emb in zip(ids_to_embed, embeddings):
                        self._framework_embeddings[fid] = emb
            
            # 3. Calculate similarities
            scores = []
            for fid in all_ids:
                if fid in self._framework_embeddings:
                    score = self._cosine_similarity(query_embedding, self._framework_embeddings[fid])
                    if score >= threshold:
                        scores.append((fid, score))
            
            # 4. Sort and return
            scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Semantic priors search: found {len(scores)} matches, top: {scores[:1]}")
            
            return [fid for fid, score in scores[:top_k]]
            
        except Exception as e:
            logger.error(f"Semantic retrieval of priors failed: {e}")
            return []

    def _load_config(self) -> Dict:
        """Load and validate YAML configuration"""
        if not self.config_path.exists():
            # Return empty config if file doesn't exist (graceful degradation)
            return {
                'schema_version': '2.0',
                'metadata': {'version': '0.0.0', 'description': 'No priors loaded'},
                'tier_1_essential': {'frameworks': {}},
                'tier_2_scenario_specific': {},
                'tier_3_supplemental': {'frameworks': {}},
                'persona_framework_defaults': {}
            }

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                raise ValueError("Priors config is empty")

            schema_version = config.get('schema_version', '1.0')

            if schema_version == '2.0':
                if 'tier_1_essential' not in config:
                    raise ValueError("V2 config missing 'tier_1_essential' section")
            else:
                if 'priors_library' not in config:
                    raise ValueError("Config missing 'priors_library' section")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in priors config: {e}")

    def _extract_tier1(self) -> Dict:
        """Extract Tier 1 frameworks as dict {framework_id: framework_data}"""
        tier1_data = self.config.get('tier_1_essential', {})
        frameworks_dict = tier1_data.get('frameworks', {})
        return frameworks_dict if isinstance(frameworks_dict, dict) else {}

    def _extract_tier3(self) -> Dict:
        """Extract Tier 3 frameworks as dict {framework_id: framework_data}"""
        tier3_data = self.config.get('tier_3_supplemental', {})
        frameworks_dict = tier3_data.get('frameworks', {})
        return frameworks_dict if isinstance(frameworks_dict, dict) else {}

    def get_frameworks_for_scenario(self, scenario: str) -> List[str]:
        """
        Get framework IDs for a drug development scenario

        Args:
            scenario: Scenario name (e.g., "target_validation", "pk_pd", "biomarker")

        Returns:
            List of framework IDs
        """
        if self.schema_version == '2.0':
            # V2: Look in tier2_scenarios
            scenario_data = self.tier2_scenarios.get(scenario, [])
            if isinstance(scenario_data, list):
                return [fw.get('id', '') for fw in scenario_data if isinstance(fw, dict)]
            return []
        else:
            # V1: Look in scenarios
            if scenario not in self.scenarios:
                return []
            return self.scenarios[scenario].get('frameworks', [])

    def get_frameworks_for_persona(self, persona: str) -> List[str]:
        """
        Get relevant framework IDs for a persona

        Args:
            persona: Persona name (e.g., "Bioscience Lead", "DMPK Scientist")

        Returns:
            List of framework IDs
        """
        if persona not in self.persona_defaults:
            return []

        persona_config = self.persona_defaults[persona]
        frameworks = []

        # Collect tier 1 frameworks for this persona
        tier1_fws = persona_config.get('tier_1_frameworks', [])
        frameworks.extend(tier1_fws)

        # Collect tier 2 scenario frameworks for this persona
        tier2_scenarios = persona_config.get('tier_2_scenarios', [])
        for scenario in tier2_scenarios:
            frameworks.extend(self.get_frameworks_for_scenario(scenario))

        return frameworks

    def get_framework_details(self, framework_id: str) -> Optional[Dict]:
        """
        Get full details for a framework

        Args:
            framework_id: Framework ID

        Returns:
            Framework dict or None if not found
        """
        if self.schema_version == '2.0':
            # Search Tier 1
            if framework_id in self.tier1_frameworks:
                return self.tier1_frameworks[framework_id]

            # Search Tier 2 (all scenarios)
            for scenario_data in self.tier2_scenarios.values():
                if isinstance(scenario_data, list):
                    for fw in scenario_data:
                        if fw.get('id') == framework_id:
                            return fw

            # Search Tier 3
            if framework_id in self.tier3_frameworks:
                return self.tier3_frameworks[framework_id]

            return None
        else:
            return self.frameworks.get(framework_id)

    def get_all_framework_ids(self) -> List[str]:
        """Get all available framework IDs"""
        ids = []

        if self.schema_version == '2.0':
            ids.extend(self.tier1_frameworks.keys())
            for scenario_data in self.tier2_scenarios.values():
                if isinstance(scenario_data, list):
                    for fw in scenario_data:
                        if isinstance(fw, dict) and 'id' in fw:
                            ids.append(fw['id'])
            ids.extend(self.tier3_frameworks.keys())
        else:
            ids.extend(self.frameworks.keys())

        return ids

    def format_single_framework(self, framework_id: str, include_full_summary: bool = True) -> str:
        """
        Format a single framework as text

        Args:
            framework_id: Framework ID
            include_full_summary: If True, include full summary; if False, only key points

        Returns:
            Formatted text for this framework
        """
        fw = self.get_framework_details(framework_id)
        if not fw:
            return f"[Framework {framework_id} not found]"

        citation = fw.get('citation', {})

        authors = citation.get('authors', 'Unknown')
        year = citation.get('year', 'N/A')
        title = citation.get('title', 'No title')
        journal = citation.get('journal', '')
        pmid = citation.get('pmid', '')
        doi = citation.get('doi', '')

        output = []
        output.append(f"**{authors} ({year})**")
        output.append(f"*{title}*")

        source_parts = []
        if journal:
            source_parts.append(journal)
        if pmid:
            source_parts.append(f"PMID: {pmid}")
        if doi and not pmid:
            source_parts.append(f"DOI: {doi}")
        if source_parts:
            output.append(" | ".join(source_parts))

        output.append("")

        if include_full_summary:
            summary = fw.get('summary', '')
            if summary:
                output.append("**Summary:**")
                output.append(summary)
                output.append("")

        key_recs = fw.get('key_recommendations', fw.get('key_points', []))
        if key_recs:
            output.append("**Key Points:**")
            for rec in key_recs:
                output.append(f"  - {rec}")
            output.append("")

        strength = fw.get('strength', fw.get('evidence_strength', ''))
        domain = fw.get('domain', '')
        if strength:
            output.append(f"*Strength: {strength}*")
        if domain:
            output.append(f"*Domain: {domain}*")

        return "\n".join(output)

    def format_priors_for_context(
        self,
        scenario: Optional[str] = None,
        persona: Optional[str] = None,
        max_frameworks: int = 3,
        compressed: bool = False,
        query: Optional[str] = None,
        embedder: Any = None
    ) -> str:
        """
        Format canonical priors as text for evidence context

        Args:
            scenario: Drug development scenario (e.g., "target_validation", "pk_pd")
            persona: Persona name (e.g., "Bioscience Lead", "DMPK Scientist")
            max_frameworks: Maximum number of frameworks to include
            compressed: If True, only include key points (save tokens)
            query: Improving relevance via semantic search
            embedder: Embedder instance for semantic search

        Returns:
            Formatted text ready to insert into prompt
        """
        framework_ids = self._select_frameworks(scenario, persona, max_frameworks, query, embedder)

        if not framework_ids:
            return ""

        output = []
        output.append("=" * 70)
        output.append("CANONICAL DRUG DEVELOPMENT FRAMEWORKS")
        output.append("=" * 70)
        output.append("")
        output.append("**Baseline Knowledge:**")
        output.append("Before reviewing new literature, consider these established frameworks")
        output.append("from regulatory guidance and drug development best practices:")
        output.append("")

        for i, fw_id in enumerate(framework_ids, 1):
            output.append(f"--- Framework {i}: {fw_id.replace('_', ' ').title()} ---")
            output.append("")
            output.append(self.format_single_framework(fw_id, include_full_summary=not compressed))
            output.append("")

        output.append("=" * 70)
        output.append("HOW TO USE THESE FRAMEWORKS:")
        output.append("=" * 70)
        output.append("")
        output.append("1. **TREAT AS STRONG PRIORS**: These represent regulatory guidance and")
        output.append("   established drug development practices. Reference them appropriately.")
        output.append("")
        output.append("2. **COMPARE NEW EVIDENCE**: In your analysis, check if new literature:")
        output.append("   - CONFIRMS these frameworks (note alignment)")
        output.append("   - ADDS NOVEL DATA not covered by frameworks")
        output.append("   - CONTRADICTS guidance (flag and explain implications)")
        output.append("")
        output.append("3. **CITE APPROPRIATELY**:")
        output.append("   - Reference frameworks using: (Framework: Author Year)")
        output.append("   - Reference literature using: (PMID: XXXXXX)")
        output.append("")
        output.append("=" * 70)
        output.append("")

        return "\n".join(output)

    def _select_frameworks(
        self,
        scenario: Optional[str],
        persona: Optional[str],
        max_frameworks: int,
        query: Optional[str] = None,
        embedder: Any = None
    ) -> List[str]:
        """
        Select which frameworks to include based on scenario, persona, and query.

        Priority order:
        1. Semantic matches (if high confidence)
        2. Frameworks relevant to BOTH scenario AND persona
        3. Frameworks relevant to scenario only
        4. Frameworks relevant to persona only
        5. Tier 1 (essential) frameworks

        Returns:
            List of framework IDs (prioritized)
        """
        scenario_fws = set(self.get_frameworks_for_scenario(scenario)) if scenario else set()
        persona_fws = set(self.get_frameworks_for_persona(persona)) if persona else set()
        
        # Priority 0: Semantic Matches (The "Intuition" Layer)
        semantic_fws = []
        if query and embedder:
            semantic_fws = self.retrieve_relevant_frameworks(query, embedder, top_k=max_frameworks)

        # Priority 1: Overlap
        overlap = list(scenario_fws & persona_fws)

        # Priority 2: Scenario-specific
        scenario_only = list(scenario_fws - persona_fws)

        # Priority 3: Persona-specific
        persona_only = list(persona_fws - scenario_fws)

        # Combine: Semantic first, then Overlap, then rest
        # Use set to avoid duplicates while preserving order
        seen = set()
        selected = []
        
        for fid in semantic_fws + overlap + scenario_only + persona_only:
            if fid not in seen:
                selected.append(fid)
                seen.add(fid)

        # Truncate
        selected = selected[:max_frameworks]

        # If still empty, use tier 1 frameworks
        if not selected:
            if self.schema_version == '2.0':
                selected = list(self.tier1_frameworks.keys())[:max_frameworks]
            elif self.frameworks:
                selected = list(self.frameworks.keys())[:max_frameworks]

        return selected

    def get_active_frameworks_summary(
        self,
        scenario: Optional[str] = None,
        persona: Optional[str] = None
    ) -> str:
        """
        Get short summary of active frameworks for display in UI

        Returns:
            Comma-separated list of framework names
        """
        framework_ids = self._select_frameworks(scenario, persona, max_frameworks=5)

        if not framework_ids:
            return "None"

        names = []
        for fw_id in framework_ids:
            fw = self.get_framework_details(fw_id)
            if fw and 'citation' in fw:
                authors = fw['citation'].get('authors', '').split(',')[0]
                year = fw['citation'].get('year', '')
                names.append(f"{authors} {year}")
            else:
                names.append(fw_id.replace('_', ' ').title())

        return ", ".join(names) if names else "None"

    def validate_framework(self, framework_dict: Dict) -> List[str]:
        """
        Validate a framework dictionary has required fields

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        required_fields = ['citation', 'summary']
        for field in required_fields:
            if field not in framework_dict:
                errors.append(f"Missing required field: {field}")

        if 'citation' in framework_dict:
            cit = framework_dict['citation']
            required_cit_fields = ['authors', 'title', 'year']
            for field in required_cit_fields:
                if field not in cit:
                    errors.append(f"Missing citation field: {field}")

        if 'key_recommendations' in framework_dict:
            if not isinstance(framework_dict['key_recommendations'], list):
                errors.append("key_recommendations must be a list")

        return errors


if __name__ == "__main__":
    pm = PriorsManager()

    print("=" * 70)
    print("PRIORS MANAGER TEST")
    print("=" * 70)
    print()

    print(f"Config path: {pm.config_path}")
    print(f"Config exists: {pm.config_path.exists()}")
    print(f"Schema version: {pm.schema_version}")
    print(f"Tier 1 frameworks: {len(pm.tier1_frameworks)}")
    print(f"Tier 2 scenarios: {list(pm.tier2_scenarios.keys())}")
    print()

    all_ids = pm.get_all_framework_ids()
    print(f"All framework IDs: {all_ids}")
    print()

    priors_text = pm.format_priors_for_context(
        scenario="target_validation",
        persona="Bioscience Lead",
        max_frameworks=2
    )
    print(f"Generated priors text: {len(priors_text)} chars")
    if priors_text:
        print(priors_text[:500] + "...")
