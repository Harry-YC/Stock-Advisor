"""
Export Citations to Various Formats

Supports:
- RIS (Research Information Systems) - Universal format
- BibTeX - LaTeX/academic writing
- EndNote XML
- CSV - Data analysis

Compatible with: Zotero, Mendeley, EndNote, RefWorks
"""

from typing import List
from core.pubmed_client import Citation


class CitationExporter:
    """Export citations to various reference manager formats"""

    @staticmethod
    def to_ris(citations: List[Citation]) -> str:
        """
        Export to RIS format (universal format for all reference managers)

        RIS format is supported by:
        - Zotero
        - Mendeley
        - EndNote
        - RefWorks
        - Papers
        """
        ris_lines = []

        for citation in citations:
            # Type of reference (JOUR = Journal Article)
            ris_lines.append("TY  - JOUR")

            # Title
            if citation.title:
                ris_lines.append(f"TI  - {citation.title}")

            # Authors (each on separate line)
            if citation.authors:
                for author in citation.authors:
                    ris_lines.append(f"AU  - {author}")

            # Journal
            if citation.journal:
                ris_lines.append(f"JO  - {citation.journal}")
                ris_lines.append(f"T2  - {citation.journal}")  # Alternate journal field

            # Year
            if citation.year:
                ris_lines.append(f"PY  - {citation.year}")

            # Abstract
            if citation.abstract:
                ris_lines.append(f"AB  - {citation.abstract}")

            # DOI
            if citation.doi:
                doi_clean = citation.doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                ris_lines.append(f"DO  - {doi_clean}")

            # PubMed ID
            if citation.pmid:
                ris_lines.append(f"AN  - {citation.pmid}")
                ris_lines.append(f"UR  - https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/")

            # Database
            ris_lines.append("DB  - PubMed")

            # End of record
            ris_lines.append("ER  - ")
            ris_lines.append("")  # Blank line between records

        return "\n".join(ris_lines)

    @staticmethod
    def to_bibtex(citations: List[Citation]) -> str:
        """
        Export to BibTeX format (LaTeX/academic writing)

        BibTeX is the standard for:
        - LaTeX documents
        - Academic writing
        - Also supported by Zotero, Mendeley
        """
        bibtex_entries = []

        for idx, citation in enumerate(citations, 1):
            # Generate citation key: FirstAuthorYearTitle
            first_author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
            year = citation.year if citation.year else "NODATE"
            # Clean title for key
            title_words = citation.title.split()[:3] if citation.title else ["Untitled"]
            title_key = "".join([w.capitalize() for w in title_words if w.isalnum()])

            cite_key = f"{first_author}{year}{title_key}"

            # Start entry
            bibtex_entries.append(f"@article{{{cite_key},")

            # Title
            if citation.title:
                bibtex_entries.append(f"  title = {{{citation.title}}},")

            # Authors
            if citation.authors:
                authors_bibtex = " and ".join(citation.authors)
                bibtex_entries.append(f"  author = {{{authors_bibtex}}},")

            # Journal
            if citation.journal:
                bibtex_entries.append(f"  journal = {{{citation.journal}}},")

            # Year
            if citation.year:
                bibtex_entries.append(f"  year = {{{citation.year}}},")

            # Abstract
            if citation.abstract:
                # Escape special LaTeX characters
                abstract_clean = citation.abstract.replace('%', '\\%').replace('_', '\\_')
                bibtex_entries.append(f"  abstract = {{{abstract_clean}}},")

            # DOI
            if citation.doi:
                doi_clean = citation.doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                bibtex_entries.append(f"  doi = {{{doi_clean}}},")

            # PubMed ID
            if citation.pmid:
                bibtex_entries.append(f"  pmid = {{{citation.pmid}}},")
                bibtex_entries.append(f"  url = {{https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/}},")

            # Close entry
            bibtex_entries.append("}")
            bibtex_entries.append("")  # Blank line

        return "\n".join(bibtex_entries)

    @staticmethod
    def to_csv(citations: List[Citation]) -> str:
        """
        Export to CSV format (for data analysis, Excel)
        """
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'PMID', 'Title', 'Authors', 'Journal', 'Year',
            'DOI', 'Abstract', 'PubMed URL'
        ])

        # Data rows
        for citation in citations:
            writer.writerow([
                citation.pmid,
                citation.title,
                '; '.join(citation.authors) if citation.authors else '',
                citation.journal,
                citation.year,
                citation.doi,
                citation.abstract,
                f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/"
            ])

        return output.getvalue()

    @staticmethod
    def to_endnote_xml(citations: List[Citation]) -> str:
        """
        Export to EndNote XML format
        """
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<xml>')
        xml_lines.append('<records>')

        for citation in citations:
            xml_lines.append('  <record>')
            xml_lines.append('    <database name="PubMed" path="pubmed.nlm.nih.gov">PubMed</database>')
            xml_lines.append('    <ref-type name="Journal Article">17</ref-type>')

            # Title
            if citation.title:
                xml_lines.append(f'    <titles><title>{_escape_xml(citation.title)}</title></titles>')

            # Authors
            if citation.authors:
                xml_lines.append('    <contributors>')
                xml_lines.append('      <authors>')
                for author in citation.authors:
                    xml_lines.append(f'        <author>{_escape_xml(author)}</author>')
                xml_lines.append('      </authors>')
                xml_lines.append('    </contributors>')

            # Journal
            if citation.journal:
                xml_lines.append(f'    <periodical><full-title>{_escape_xml(citation.journal)}</full-title></periodical>')

            # Year
            if citation.year:
                xml_lines.append(f'    <dates><year>{citation.year}</year></dates>')

            # Abstract
            if citation.abstract:
                xml_lines.append(f'    <abstract>{_escape_xml(citation.abstract)}</abstract>')

            # DOI
            if citation.doi:
                doi_clean = citation.doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                xml_lines.append(f'    <electronic-resource-num>{_escape_xml(doi_clean)}</electronic-resource-num>')

            # PMID
            if citation.pmid:
                xml_lines.append(f'    <accession-num>{citation.pmid}</accession-num>')
                xml_lines.append(f'    <urls><related-urls><url>https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/</url></related-urls></urls>')

            xml_lines.append('  </record>')

        xml_lines.append('</records>')
        xml_lines.append('</xml>')

        return "\n".join(xml_lines)


def _escape_xml(text: str) -> str:
    """Escape special XML characters"""
    if not text:
        return ""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


# Export format metadata
EXPORT_FORMATS = {
    'RIS': {
        'name': 'RIS (Research Information Systems)',
        'extension': '.ris',
        'description': 'Universal format for Zotero, Mendeley, EndNote, RefWorks',
        'mime_type': 'application/x-research-info-systems'
    },
    'BibTeX': {
        'name': 'BibTeX',
        'extension': '.bib',
        'description': 'LaTeX/academic writing (also Zotero, Mendeley)',
        'mime_type': 'application/x-bibtex'
    },
    'CSV': {
        'name': 'CSV (Comma Separated Values)',
        'extension': '.csv',
        'description': 'Excel, data analysis',
        'mime_type': 'text/csv'
    },
    'EndNote': {
        'name': 'EndNote XML',
        'extension': '.xml',
        'description': 'EndNote reference manager',
        'mime_type': 'application/xml'
    }
}
