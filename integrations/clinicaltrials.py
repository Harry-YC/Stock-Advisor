"""
ClinicalTrials.gov API Client

Search and retrieve clinical trial information for competitive intelligence.
Uses the ClinicalTrials.gov API v2.

API Documentation: https://clinicaltrials.gov/data-api/api
"""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TrialPhase(Enum):
    """Clinical trial phases"""
    EARLY_PHASE_1 = "Early Phase 1"
    PHASE_1 = "Phase 1"
    PHASE_1_2 = "Phase 1/Phase 2"
    PHASE_2 = "Phase 2"
    PHASE_2_3 = "Phase 2/Phase 3"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NOT_APPLICABLE = "Not Applicable"


class TrialStatus(Enum):
    """Clinical trial recruitment status"""
    NOT_YET_RECRUITING = "Not yet recruiting"
    RECRUITING = "Recruiting"
    ENROLLING_BY_INVITATION = "Enrolling by invitation"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    COMPLETED = "Completed"
    WITHDRAWN = "Withdrawn"
    UNKNOWN = "Unknown status"


@dataclass
class ClinicalTrial:
    """Represents a clinical trial from ClinicalTrials.gov"""
    nct_id: str
    title: str
    status: str
    phase: str = ""
    sponsor: str = ""
    conditions: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    enrollment: Optional[int] = None
    study_type: str = ""
    brief_summary: str = ""
    primary_outcomes: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    url: str = ""

    @property
    def is_active(self) -> bool:
        """Check if trial is currently active"""
        active_statuses = [
            "Not yet recruiting",
            "Recruiting",
            "Enrolling by invitation",
            "Active, not recruiting"
        ]
        return self.status in active_statuses

    @property
    def phase_number(self) -> Optional[int]:
        """Extract numeric phase for sorting"""
        if "4" in self.phase:
            return 4
        elif "3" in self.phase:
            return 3
        elif "2" in self.phase:
            return 2
        elif "1" in self.phase:
            return 1
        return None


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API v2"""

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        """Initialize the client"""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json"
        })

    def search(
        self,
        query: str,
        max_results: int = 100,
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
        sponsor: Optional[str] = None,
        start_date_from: Optional[str] = None,
        start_date_to: Optional[str] = None
    ) -> List[ClinicalTrial]:
        """
        Search for clinical trials

        Args:
            query: Search query (compound name, target, indication, etc.)
            max_results: Maximum number of results to return
            status: Filter by status (e.g., ["Recruiting", "Active, not recruiting"])
            phase: Filter by phase (e.g., ["Phase 2", "Phase 3"])
            sponsor: Filter by sponsor name
            start_date_from: Filter by start date (YYYY-MM-DD)
            start_date_to: Filter by start date (YYYY-MM-DD)

        Returns:
            List of ClinicalTrial objects
        """
        trials = []
        page_token = None

        fields = [
            "NCTId",
            "BriefTitle",
            "OfficialTitle",
            "OverallStatus",
            "Phase",
            "LeadSponsorName",
            "Condition",
            "InterventionName",
            "StartDate",
            "PrimaryCompletionDate",
            "EnrollmentCount",
            "StudyType",
            "BriefSummary",
            "PrimaryOutcomeMeasure",
            "LocationCity",
            "LocationCountry"
        ]

        while len(trials) < max_results:
            params = {
                "query.term": query,
                "fields": ",".join(fields),
                "pageSize": min(100, max_results - len(trials)),
                "format": "json"
            }

            if page_token:
                params["pageToken"] = page_token

            # Build filter expressions
            filters = []
            if status:
                status_filter = " OR ".join([f'AREA[OverallStatus]"{s}"' for s in status])
                filters.append(f"({status_filter})")
            if phase:
                phase_filter = " OR ".join([f'AREA[Phase]"{p}"' for p in phase])
                filters.append(f"({phase_filter})")
            if sponsor:
                filters.append(f'AREA[LeadSponsorName]"{sponsor}"')
            if start_date_from:
                filters.append(f'AREA[StartDate]RANGE[{start_date_from},MAX]')
            if start_date_to:
                filters.append(f'AREA[StartDate]RANGE[MIN,{start_date_to}]')

            if filters:
                params["filter.advanced"] = " AND ".join(filters)

            try:
                response = self.session.get(
                    f"{self.BASE_URL}/studies",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"ClinicalTrials.gov API error: {e}")
                break

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                trial = self._parse_study(study)
                if trial:
                    trials.append(trial)

            # Check for next page
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return trials

    def get_trial(self, nct_id: str) -> Optional[ClinicalTrial]:
        """
        Get a specific trial by NCT ID

        Args:
            nct_id: The NCT identifier (e.g., "NCT04123456")

        Returns:
            ClinicalTrial object or None if not found
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/studies/{nct_id}",
                params={"format": "json"},
                timeout=30
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return self._parse_study(response.json())
        except requests.exceptions.RequestException:
            return None

    def search_by_compound(
        self,
        compound_name: str,
        include_synonyms: bool = True,
        active_only: bool = False
    ) -> List[ClinicalTrial]:
        """
        Search for trials involving a specific compound

        Args:
            compound_name: Drug/compound name
            include_synonyms: Include common variations in search
            active_only: Only return active trials

        Returns:
            List of ClinicalTrial objects
        """
        # Build query with potential synonyms
        query = compound_name
        if include_synonyms:
            # Add common variations
            variations = [
                compound_name,
                compound_name.upper(),
                compound_name.lower(),
                compound_name.replace("-", " "),
                compound_name.replace(" ", "-")
            ]
            query = " OR ".join(set(variations))

        status = None
        if active_only:
            status = [
                "Not yet recruiting",
                "Recruiting",
                "Enrolling by invitation",
                "Active, not recruiting"
            ]

        return self.search(query, status=status)

    def search_by_target(
        self,
        target_name: str,
        active_only: bool = False
    ) -> List[ClinicalTrial]:
        """
        Search for trials targeting a specific molecular target

        Args:
            target_name: Target name (e.g., "KRAS", "PD-1", "EGFR")
            active_only: Only return active trials

        Returns:
            List of ClinicalTrial objects
        """
        # Search in interventions and conditions
        query = f'"{target_name}"'

        status = None
        if active_only:
            status = [
                "Not yet recruiting",
                "Recruiting",
                "Enrolling by invitation",
                "Active, not recruiting"
            ]

        return self.search(query, status=status)

    def search_by_indication(
        self,
        indication: str,
        phase: Optional[List[str]] = None,
        active_only: bool = False
    ) -> List[ClinicalTrial]:
        """
        Search for trials for a specific indication/disease

        Args:
            indication: Disease/condition name
            phase: Filter by phase
            active_only: Only return active trials

        Returns:
            List of ClinicalTrial objects
        """
        status = None
        if active_only:
            status = [
                "Not yet recruiting",
                "Recruiting",
                "Enrolling by invitation",
                "Active, not recruiting"
            ]

        return self.search(indication, status=status, phase=phase)

    def get_sponsor_pipeline(
        self,
        sponsor_name: str,
        active_only: bool = True
    ) -> List[ClinicalTrial]:
        """
        Get all trials for a specific sponsor (competitive intelligence)

        Args:
            sponsor_name: Company/organization name
            active_only: Only return active trials

        Returns:
            List of ClinicalTrial objects sorted by phase
        """
        status = None
        if active_only:
            status = [
                "Not yet recruiting",
                "Recruiting",
                "Enrolling by invitation",
                "Active, not recruiting"
            ]

        trials = self.search("", sponsor=sponsor_name, status=status, max_results=500)

        # Sort by phase (descending) then by start date
        trials.sort(
            key=lambda t: (-(t.phase_number or 0), t.start_date or ""),
            reverse=True
        )

        return trials

    def _parse_study(self, study_data: Dict) -> Optional[ClinicalTrial]:
        """Parse API response into ClinicalTrial object"""
        try:
            protocol = study_data.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            design_module = protocol.get("designModule", {})
            description_module = protocol.get("descriptionModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            contacts_module = protocol.get("contactsLocationsModule", {})

            nct_id = id_module.get("nctId", "")
            if not nct_id:
                return None

            # Extract interventions
            interventions = []
            for arm in arms_module.get("interventions", []):
                name = arm.get("name", "")
                if name:
                    interventions.append(name)

            # Extract primary outcomes
            primary_outcomes = []
            for outcome in outcomes_module.get("primaryOutcomes", []):
                measure = outcome.get("measure", "")
                if measure:
                    primary_outcomes.append(measure)

            # Extract locations
            locations = []
            for loc in contacts_module.get("locations", [])[:5]:  # First 5 locations
                city = loc.get("city", "")
                country = loc.get("country", "")
                if city and country:
                    locations.append(f"{city}, {country}")
                elif country:
                    locations.append(country)

            # Get lead sponsor
            sponsor = ""
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            if lead_sponsor:
                sponsor = lead_sponsor.get("name", "")

            # Parse phases
            phases = design_module.get("phases", [])
            phase = ", ".join(phases) if phases else ""

            return ClinicalTrial(
                nct_id=nct_id,
                title=id_module.get("briefTitle", "") or id_module.get("officialTitle", ""),
                status=status_module.get("overallStatus", ""),
                phase=phase,
                sponsor=sponsor,
                conditions=conditions_module.get("conditions", []),
                interventions=interventions,
                start_date=status_module.get("startDateStruct", {}).get("date", ""),
                completion_date=status_module.get("primaryCompletionDateStruct", {}).get("date", ""),
                enrollment=design_module.get("enrollmentInfo", {}).get("count"),
                study_type=design_module.get("studyType", ""),
                brief_summary=description_module.get("briefSummary", ""),
                primary_outcomes=primary_outcomes,
                locations=locations,
                url=f"https://clinicaltrials.gov/study/{nct_id}"
            )
        except Exception as e:
            print(f"Error parsing study: {e}")
            return None


def get_competitive_landscape(
    target_or_indication: str,
    include_phases: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to get competitive landscape for a target or indication

    Args:
        target_or_indication: Target name or disease indication
        include_phases: Filter by phases (default: Phase 2 and 3)

    Returns:
        Dict with:
        - trials: List of ClinicalTrial objects
        - by_sponsor: Dict mapping sponsor to trial count
        - by_phase: Dict mapping phase to trial count
        - active_count: Number of active trials
    """
    client = ClinicalTrialsClient()

    phases = include_phases or ["Phase 2", "Phase 2/Phase 3", "Phase 3"]
    trials = client.search(target_or_indication, phase=phases, max_results=200)

    # Aggregate by sponsor
    by_sponsor: Dict[str, int] = {}
    for trial in trials:
        sponsor = trial.sponsor or "Unknown"
        by_sponsor[sponsor] = by_sponsor.get(sponsor, 0) + 1

    # Aggregate by phase
    by_phase: Dict[str, int] = {}
    for trial in trials:
        phase = trial.phase or "Unknown"
        by_phase[phase] = by_phase.get(phase, 0) + 1

    # Count active
    active_count = sum(1 for t in trials if t.is_active)

    # Sort sponsors by count
    by_sponsor = dict(sorted(by_sponsor.items(), key=lambda x: x[1], reverse=True))

    return {
        "trials": trials,
        "by_sponsor": by_sponsor,
        "by_phase": by_phase,
        "active_count": active_count,
        "total_count": len(trials)
    }
