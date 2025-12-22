"""
SEC EDGAR Integration

Provides access to SEC filings including:
- 10-K, 10-Q, 8-K filings
- Form 4 (insider transactions)
- 13F (institutional holdings)

SEC EDGAR API is free and doesn't require an API key.
"""

import logging
import requests
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """SEC filing record."""
    form_type: str  # 10-K, 10-Q, 8-K, 4, 13F, etc.
    filed_date: str
    accepted_datetime: str
    accession_number: str
    primary_document: str
    filing_url: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_type": self.form_type,
            "filed_date": self.filed_date,
            "accession_number": self.accession_number,
            "filing_url": self.filing_url,
            "description": self.description,
        }


@dataclass
class InsiderTransaction:
    """Form 4 insider transaction."""
    filer_name: str
    filer_title: str
    transaction_date: str
    transaction_type: str  # 'buy' or 'sell'
    shares: float
    price: float
    shares_owned_after: float
    form_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filer_name": self.filer_name,
            "filer_title": self.filer_title,
            "transaction_date": self.transaction_date,
            "transaction_type": self.transaction_type,
            "shares": self.shares,
            "price": self.price,
            "total_value": self.shares * self.price,
            "shares_owned_after": self.shares_owned_after,
        }


class SECEdgarClient:
    """
    Client for SEC EDGAR API.

    Note: SEC requires a User-Agent header with contact information.
    """

    BASE_URL = "https://data.sec.gov"
    FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    # Cache TTLs
    CIK_CACHE_TTL = 86400 * 30  # 30 days for CIK lookups
    FILINGS_CACHE_TTL = 3600  # 1 hour for filings

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or "StockAdvisor research@example.com"
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._cik_map: Dict[str, str] = {}

    def _get_cache(self, key: str, ttl: int) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() - entry["timestamp"] < timedelta(seconds=ttl):
                    return entry["data"]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set item in cache."""
        with self._lock:
            self._cache[key] = {
                "data": data,
                "timestamp": datetime.now()
            }

    def get_company_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CIK string (padded to 10 digits) or None
        """
        symbol = symbol.upper()

        # Check memory cache
        if symbol in self._cik_map:
            return self._cik_map[symbol]

        cache_key = f"cik:{symbol}"
        cached = self._get_cache(cache_key, self.CIK_CACHE_TTL)
        if cached:
            return cached

        try:
            # SEC provides a ticker-to-CIK mapping
            response = requests.get(
                f"{self.BASE_URL}/files/company_tickers.json",
                headers=self.headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            # Build lookup map
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                self._cik_map[ticker] = cik

            if symbol in self._cik_map:
                self._set_cache(cache_key, self._cik_map[symbol])
                return self._cik_map[symbol]

            logger.warning(f"CIK not found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"CIK lookup failed for {symbol}: {e}")
            return None

    def get_recent_filings(
        self,
        symbol: str,
        form_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SECFiling]:
        """
        Get recent SEC filings for a company.

        Args:
            symbol: Stock ticker symbol
            form_types: Filter by form types (e.g., ['10-K', '10-Q', '8-K'])
            limit: Maximum number of filings

        Returns:
            List of SECFiling objects
        """
        cik = self.get_company_cik(symbol)
        if not cik:
            return []

        cache_key = f"filings:{symbol}:{','.join(form_types or [])}"
        cached = self._get_cache(cache_key, self.FILINGS_CACHE_TTL)
        if cached:
            return cached[:limit]

        try:
            # Get company submissions
            response = requests.get(
                f"{self.BASE_URL}/submissions/CIK{cik}.json",
                headers=self.headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            recent = data.get("filings", {}).get("recent", {})
            if not recent:
                logger.warning(f"No filings found for {symbol}")
                return []

            filings = []
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            descriptions = recent.get("primaryDocDescription", [])

            for i in range(min(len(forms), 100)):  # Check up to 100 filings
                form_type = forms[i] if i < len(forms) else ""

                # Filter by form type if specified
                if form_types and form_type not in form_types:
                    continue

                accession = accessions[i].replace("-", "") if i < len(accessions) else ""
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""

                # Build filing URL
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"

                filings.append(SECFiling(
                    form_type=form_type,
                    filed_date=dates[i] if i < len(dates) else "",
                    accepted_datetime=recent.get("acceptanceDateTime", [""])[i] if i < len(recent.get("acceptanceDateTime", [])) else "",
                    accession_number=accessions[i] if i < len(accessions) else "",
                    primary_document=primary_doc,
                    filing_url=filing_url,
                    description=descriptions[i] if i < len(descriptions) else "",
                ))

                if len(filings) >= limit:
                    break

            self._set_cache(cache_key, filings)
            logger.info(f"Retrieved {len(filings)} filings for {symbol}")
            return filings

        except Exception as e:
            logger.error(f"Filings fetch failed for {symbol}: {e}")
            return []

    def get_insider_transactions(
        self,
        symbol: str,
        days: int = 90
    ) -> List[InsiderTransaction]:
        """
        Get recent insider transactions (Form 4 filings).

        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back

        Returns:
            List of InsiderTransaction objects
        """
        # Get Form 4 filings
        filings = self.get_recent_filings(symbol, form_types=["4"], limit=50)

        if not filings:
            return []

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        transactions = []

        for filing in filings:
            if filing.filed_date < cutoff_date:
                continue

            # Parse basic transaction info from filing
            # Note: Full parsing would require downloading and parsing XML
            # This provides basic info from the filing metadata
            transactions.append(InsiderTransaction(
                filer_name=filing.description or "Unknown",
                filer_title="",
                transaction_date=filing.filed_date,
                transaction_type="unknown",
                shares=0,
                price=0,
                shares_owned_after=0,
                form_url=filing.filing_url,
            ))

        logger.info(f"Found {len(transactions)} insider filings for {symbol} in last {days} days")
        return transactions

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic company information from SEC.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with company info
        """
        cik = self.get_company_cik(symbol)
        if not cik:
            return {}

        try:
            response = requests.get(
                f"{self.BASE_URL}/submissions/CIK{cik}.json",
                headers=self.headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "cik": cik,
                "name": data.get("name", ""),
                "sic": data.get("sic", ""),
                "sic_description": data.get("sicDescription", ""),
                "state": data.get("stateOfIncorporation", ""),
                "fiscal_year_end": data.get("fiscalYearEnd", ""),
                "exchanges": data.get("exchanges", []),
            }

        except Exception as e:
            logger.error(f"Company info fetch failed for {symbol}: {e}")
            return {}

    def format_filings_summary(self, symbol: str) -> str:
        """
        Generate a markdown summary of SEC filings.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Formatted markdown string
        """
        # Get different types of filings
        annual = self.get_recent_filings(symbol, form_types=["10-K"], limit=1)
        quarterly = self.get_recent_filings(symbol, form_types=["10-Q"], limit=2)
        current = self.get_recent_filings(symbol, form_types=["8-K"], limit=3)
        insider_filings = self.get_recent_filings(symbol, form_types=["4"], limit=5)

        lines = [
            f"## SEC Filings ({symbol})",
            "",
        ]

        if annual:
            lines.append(f"**Latest 10-K (Annual Report):** {annual[0].filed_date}")
        else:
            lines.append("**Latest 10-K:** Not found")

        if quarterly:
            lines.append(f"**Latest 10-Q (Quarterly Report):** {quarterly[0].filed_date}")

        if current:
            lines.extend([
                "",
                "**Recent 8-K Filings:**",
            ])
            for f in current:
                desc = f.description[:50] + "..." if len(f.description) > 50 else f.description
                lines.append(f"- {f.filed_date}: {desc or 'Current Report'}")

        if insider_filings:
            lines.extend([
                "",
                f"**Recent Insider Filings:** {len(insider_filings)} Form 4 filings in database",
            ])

        return "\n".join(lines)


# Convenience functions
def get_sec_filings(symbol: str, form_types: Optional[List[str]] = None) -> List[SECFiling]:
    """Quick SEC filings lookup."""
    client = SECEdgarClient()
    return client.get_recent_filings(symbol, form_types)


def get_sec_summary(symbol: str) -> str:
    """Quick SEC summary lookup."""
    client = SECEdgarClient()
    return client.format_filings_summary(symbol)
