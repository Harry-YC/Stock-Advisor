"""
Visualizations Module

Timeline, citation network, and table visualizations for search results.
"""

import streamlit as st
from datetime import datetime
from typing import List, Any, Dict

from core.citation_utils import get_attr

# Check for visualization dependencies
try:
    import plotly.express as px
    import pandas as pd
    from core.citation_network import (
        CitationNetworkBuilder,
        create_network_visualization,
        create_timeline_visualization
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def render_visualizations(results: Dict[str, Any]):
    """
    Render the visualizations section for search results.

    Args:
        results: Search results dictionary containing citations
    """
    if not VISUALIZATION_AVAILABLE:
        st.info("Visualization dependencies not installed (plotly, pandas)")
        return

    citations_list = results.get('citations', [])
    if not citations_list:
        st.info("No citations to visualize")
        return

    with st.expander("📊 Visualizations", expanded=False):
        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "📅 Timeline",
            "🔗 Citation Network",
            "📋 Table View"
        ])

        with viz_tab1:
            _render_timeline(citations_list)

        with viz_tab2:
            _render_citation_network(citations_list)

        with viz_tab3:
            _render_table_view(citations_list, results)


def _render_timeline(citations: List[Any]):
    """
    Render the publication timeline visualization.

    Args:
        citations: List of Citation objects or dicts
    """
    st.subheader("Publication Timeline")

    # Extract years from citations
    years = []
    for c in citations:
        year = get_attr(c, 'year')
        if year:
            try:
                years.append(int(year))
            except (ValueError, TypeError):
                pass

    if not years:
        st.info("No year data available for timeline")
        return

    # Create histogram
    fig = px.histogram(
        x=years,
        nbins=min(20, len(set(years))),
        title="Publications by Year",
        labels={'x': 'Year', 'y': 'Number of Papers'}
    )
    fig.update_layout(
        bargap=0.1,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Year Range", f"{min(years)} - {max(years)}")
    col2.metric("Median Year", int(sorted(years)[len(years)//2]))
    col3.metric("Most Recent", max(years))


def _render_citation_network(citations: List[Any]):
    """
    Render the citation network visualization.

    Args:
        citations: List of Citation objects or dicts
    """
    st.subheader("Citation Network")
    st.caption("Build a network showing citation relationships between papers")

    # Get PMIDs for network
    pmids = []
    for c in citations:
        pmid = get_attr(c, 'pmid')
        if pmid:
            pmids.append(pmid)

    if len(pmids) < 3:
        st.info("Need at least 3 papers to build a citation network")
        return

    col1, col2 = st.columns(2)

    with col1:
        network_mode = st.radio(
            "Network Mode",
            ["Minimal (corpus only)", "Extended (include citing papers)"],
            help="Minimal: Only show connections within your search results. "
                 "Extended: Include papers that cite your results."
        )

    with col2:
        max_papers_network = st.slider(
            "Max papers to analyze",
            10,
            min(100, len(pmids)),
            min(30, len(pmids))
        )

    if st.button("🔗 Build Citation Network", type="primary"):
        with st.spinner("Building citation network (this may take 30-60 seconds)..."):
            try:
                builder = CitationNetworkBuilder()

                if network_mode == "Minimal (corpus only)":
                    network = builder.build_network_minimal(pmids[:max_papers_network])
                else:
                    network = builder.build_network(
                        pmids[:max_papers_network],
                        max_citations_per_paper=20,
                        max_references_per_paper=20,
                        include_external=True
                    )

                if network.num_papers > 0:
                    # Display network stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Papers", network.num_papers)
                    col2.metric("Citations", network.num_citations)
                    col3.metric("Clusters", len(network.clusters))
                    col4.metric("Hub Papers", len(network.hub_papers))

                    # Create visualization
                    fig = create_network_visualization(
                        network,
                        color_by="cluster",
                        size_by="citation_count",
                        corpus_only=(network_mode == "Minimal (corpus only)")
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="network_viz")

                    # Show hub papers
                    if network.hub_papers:
                        st.markdown("**Hub Papers** (highly cited within network):")
                        for paper_id in network.hub_papers[:5]:
                            node = network.nodes.get(paper_id)
                            if node:
                                st.markdown(
                                    f"- [{node.title[:80]}...]"
                                    f"(https://pubmed.ncbi.nlm.nih.gov/{node.pmid}/) "
                                    f"({node.citation_count} citations)"
                                )
                else:
                    st.warning("Could not build network - no citation data found")

            except Exception as e:
                st.error(f"Failed to build network: {str(e)}")


def _render_table_view(citations: List[Any], results: Dict[str, Any]):
    """
    Render the sortable table view of results.

    Args:
        citations: List of Citation objects or dicts
        results: Full results dictionary for additional data
    """
    st.subheader("Results Table")
    st.caption("Sortable and filterable view of all results")

    # Build dataframe from citations
    table_data = []
    for idx, c in enumerate(citations, 1):
        pmid = get_attr(c, 'pmid', '')
        title = get_attr(c, 'title', 'No title')
        authors = get_attr(c, 'authors', [])
        journal = get_attr(c, 'journal', '')
        year = get_attr(c, 'year', '')

        # Get score if available
        score = None
        if results.get('scored_citations'):
            for sc in results['scored_citations']:
                if sc.citation.get('pmid') == pmid:
                    score = sc.final_score
                    break

        first_author = authors[0] if authors else 'Unknown'
        if isinstance(first_author, list):
            first_author = first_author[0] if first_author else 'Unknown'

        table_data.append({
            'Rank': idx,
            'Score': f"{score:.3f}" if score else "-",
            'Title': title[:100] + '...' if len(str(title)) > 100 else title,
            'First Author': first_author,
            'Journal': journal,
            'Year': year,
            'PMID': pmid
        })

    if not table_data:
        st.info("No data to display")
        return

    df = pd.DataFrame(table_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PMID": st.column_config.LinkColumn(
                "PMID",
                help="Click to view on PubMed",
                display_text=r"(\d+)",
                validate=r"^\d+$"
            ),
            "Score": st.column_config.TextColumn(
                "Score",
                help="Clinical Utility Score"
            ),
            "Title": st.column_config.TextColumn(
                "Title",
                width="large"
            ),
        }
    )

    # Export options
    _render_export_buttons(df, results)


def _render_export_buttons(df, results: Dict[str, Any]):
    """
    Render export buttons for table data.

    Args:
        df: DataFrame with table data
        results: Full results dictionary
    """
    csv = df.to_csv(index=False)

    col_csv, col_word = st.columns(2)

    with col_csv:
        st.download_button(
            "📥 Download CSV",
            csv,
            f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

    with col_word:
        try:
            from core.report_export import generate_literature_report

            citations_list = results.get('citations', [])
            report_buffer = generate_literature_report(
                citations=citations_list,
                query=results.get('query'),
                project_name=st.session_state.get('current_project_name'),
                include_abstracts=False
            )

            st.download_button(
                label="📄 Download Word",
                data=report_buffer,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="search_export_word_btn",
                use_container_width=True
            )
        except Exception as e:
            st.button(
                "📄 Download Word",
                disabled=True,
                help=f"Export unavailable: {e}",
                key="search_export_word_btn_disabled",
                use_container_width=True
            )
