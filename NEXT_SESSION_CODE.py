# ============================================================================
# CODE FOR NEXT SESSION - COPY/PASTE READY
# ============================================================================

# =============================================================================
# SECTION 1: FIX TAB 1 INDENTATION (app_lr.py lines 413-472)
# =============================================================================
# Replace lines 413-472 with this properly indented content:

"""
        # Show examples
        with st.expander("💡 Query Examples", expanded=False):
            st.markdown(\"\"\"
**Natural Language** (will be auto-optimized):
- "In adults with gastric outlet obstruction, how do stents compare to surgery for quality of life?"
- "What are the adverse events of duodenal stents vs surgical gastrojejunostomy?"

**PubMed Syntax** (will be used directly):
- `("Gastric Outlet Obstruction"[MeSH]) AND (stent OR surgery)`
- `"machine learning"[Title] AND radiology[MeSH]`
- `(diabetes[tiab] AND treatment[tiab]) NOT type1[tiab]`

**Tips:**
- Use quotes for phrases: `"machine learning"`
- MeSH terms: `"Gastric Outlet Obstruction"[MeSH]`
- Boolean: `AND`, `OR`, `NOT`
- Field tags: `[Title]`, `[tiab]`, `[Author]`
            \"\"\")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_results = st.number_input(
                "Max Results",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="query_max_results"
            )

        with col2:
            date_from = st.text_input(
                "Date From (YYYY/MM/DD)",
                placeholder="2020/01/01",
                help="Optional: Filter by publication date",
                key="query_date_from"
            )

        with col3:
            # Default to today's date (updates daily)
            today = datetime.now().strftime("%Y/%m/%d")
            date_to = st.text_input(
                "Date To (YYYY/MM/DD)",
                value=today,
                help=f"Defaults to today ({today})",
                key="query_date_to"
            )

        # Publication type filter
        pub_types = st.multiselect(
            "Publication Types (optional)",
            [
                "Clinical Trial",
                "Randomized Controlled Trial",
                "Meta-Analysis",
                "Systematic Review",
                "Review",
                "Case Reports",
                "Observational Study"
            ],
            key="query_pub_types"
        )

        # Search button for query search
        query_search_clicked = st.button("🔎 Execute Query Search", type="primary", use_container_width=True, key="query_search_btn")
"""

# =============================================================================
# SECTION 2: ADD TAB 2 (Insert after Tab 1, before line 473)
# =============================================================================
# Add this complete Tab 2 code:

"""
    # =============================================================================
    # TAB 2: IDENTIFIER SEARCH (new functionality)
    # =============================================================================
    with tab2:
        st.markdown(\"\"\"
Enter PMIDs or DOIs to directly fetch citations from PubMed.
**Supports batch import** - separate multiple identifiers with commas or newlines.
        \"\"\")

        col1, col2 = st.columns(2)

        with col1:
            pmid_input = st.text_area(
                "PubMed IDs (PMIDs)",
                placeholder="12345678, 23456789, 34567890\\nor one per line",
                help="Enter one or more PMIDs (1-8 digits)",
                height=150,
                key="pmid_input"
            )

        with col2:
            doi_input = st.text_area(
                "Digital Object Identifiers (DOIs)",
                placeholder="10.1038/nature12373\\n10.1016/j.yfrne.2016.01.008",
                help="Enter one or more DOIs (with or without URL prefix)",
                height=150,
                key="doi_input"
            )

        # Show validation feedback
        if pmid_input or doi_input:
            st.caption("**Validation:**")
            col1, col2 = st.columns(2)

            if pmid_input:
                valid_pmids, pmid_errors = validate_identifiers(pmid_input, "pmid")
                with col1:
                    if valid_pmids:
                        st.success(f"✓ {len(valid_pmids)} valid PMID(s)")
                    if pmid_errors:
                        for error in pmid_errors[:3]:  # Show first 3 errors
                            st.error(f"✗ {error}")
                        if len(pmid_errors) > 3:
                            st.caption(f"...and {len(pmid_errors) - 3} more error(s)")

            if doi_input:
                valid_dois, doi_errors = validate_identifiers(doi_input, "doi")
                with col2:
                    if valid_dois:
                        st.success(f"✓ {len(valid_dois)} valid DOI(s)")
                    if doi_errors:
                        for error in doi_errors[:3]:  # Show first 3 errors
                            st.error(f"✗ {error}")
                        if len(doi_errors) > 3:
                            st.caption(f"...and {len(doi_errors) - 3} more error(s)")

        # Search button for identifier search
        identifier_search_clicked = st.button("🔎 Fetch Citations", type="primary", use_container_width=True, key="identifier_search_btn")
"""

# =============================================================================
# SECTION 3: ADD IDENTIFIER SEARCH EXECUTION (Insert before query search execution)
# =============================================================================
# Add this BEFORE the existing "if st.button..." at line 473:

"""
    # =============================================================================
    # IDENTIFIER SEARCH EXECUTION
    # =============================================================================
    if 'identifier_search_clicked' in locals() and identifier_search_clicked:
        # Validate and collect identifiers
        valid_pmids = []
        valid_dois = []

        if pmid_input:
            valid_pmids, _ = validate_identifiers(pmid_input, "pmid")
        if doi_input:
            valid_dois, _ = validate_identifiers(doi_input, "doi")

        if not valid_pmids and not valid_dois:
            st.error("⚠️ Please enter at least one valid PMID or DOI")
            st.stop()

        with st.spinner(f"🔍 Fetching {len(valid_pmids) + len(valid_dois)} citation(s)..."):
            try:
                # Initialize PubMed client
                client = PubMedClient(
                    email=settings.PUBMED_EMAIL,
                    api_key=settings.PUBMED_API_KEY
                )

                all_citations = []
                not_found = []

                # Fetch by PMID
                if valid_pmids:
                    pmid_citations, failed_batches = client.fetch_citations(valid_pmids)
                    if failed_batches:
                        st.warning(f"⚠️ {len(failed_batches)} batch(es) failed to fetch")
                    all_citations.extend(pmid_citations)
                    # Track which PMIDs weren't found
                    fetched_pmids = {c.pmid for c in pmid_citations}
                    not_found.extend([p for p in valid_pmids if p not in fetched_pmids])

                # Fetch by DOI
                if valid_dois:
                    doi_citations, doi_not_found = client.fetch_by_doi(valid_dois)
                    all_citations.extend(doi_citations)
                    not_found.extend(doi_not_found)

                if all_citations:
                    # Save citations to database
                    for c in all_citations:
                        citation_dict = {
                            "pmid": c.pmid,
                            "title": c.title,
                            "authors": c.authors,
                            "journal": c.journal,
                            "year": c.year,
                            "abstract": c.abstract,
                            "doi": c.doi,
                            "publication_types": None,
                            "keywords": None
                        }
                        citation_dao.upsert_citation(citation_dict)
                        citation_dao.add_citation_to_project(st.session_state.current_project_id, c.pmid)

                    # Save to search history
                    identifier_query = f"Identifiers: PMIDs={len(valid_pmids)}, DOIs={len(valid_dois)}"
                    search_dao.add_search(
                        project_id=st.session_state.current_project_id,
                        query=identifier_query,
                        filters={},
                        total_results=len(all_citations),
                        retrieved_count=len(all_citations)
                    )

                    # Store in session state (same format as query search for results display)
                    st.session_state.search_results = {
                        "query": identifier_query,
                        "optimized_query": identifier_query,
                        "used_fallback": False,
                        "total_count": len(all_citations),
                        "retrieved_count": len(all_citations),
                        "query_translation": "Direct identifier fetch",
                        "citations": all_citations,
                        "search_date": datetime.now().isoformat()
                    }

                    st.success(f"✓ Fetched {len(all_citations)} citation(s)")
                    if not_found:
                        st.warning(f"⚠️ {len(not_found)} identifier(s) not found: {', '.join(not_found[:5])}")
                    st.success(f"💾 Saved to database")
                    st.rerun()
                else:
                    st.error("❌ No citations found for the provided identifiers")

            except Exception as e:
                st.error(f"❌ Fetch failed: {str(e)}")
"""

# =============================================================================
# SECTION 4: MODIFY QUERY SEARCH BUTTON (Line 473)
# =============================================================================
# Change this line:
#     if st.button("🔎 Execute Search", type="primary", use_container_width=True):
# To:
#     if query_search_clicked:

# =============================================================================
# END OF CODE SECTIONS
# =============================================================================

# TEST IDENTIFIERS:
# PMIDs: 34906319, 33146295, 32876693
# DOI: 10.1038/s41586-021-03819-2
# DOI with URL: https://doi.org/10.1016/j.cell.2021.04.048
