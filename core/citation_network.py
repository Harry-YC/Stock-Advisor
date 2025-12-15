"""
Citation Network Analysis

Builds and analyzes citation networks from literature search results.
Uses NetworkX for graph operations and Plotly for visualization.

Features:
- Build citation graph from papers
- Identify hub papers (highly cited)
- Detect research clusters
- Calculate influence metrics (PageRank, betweenness centrality)
- Generate interactive visualizations
"""

import networkx as nx
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from integrations.semantic_scholar import SemanticScholarClient, SemanticPaper


@dataclass
class NetworkNode:
    """Represents a paper node in the citation network"""
    paper_id: str
    pmid: str
    title: str
    authors: List[str]
    year: Optional[int]
    citation_count: int = 0
    in_degree: int = 0  # Papers citing this
    out_degree: int = 0  # Papers this cites
    pagerank: float = 0.0
    cluster: int = -1
    is_in_corpus: bool = True  # Whether this paper is in our search results


@dataclass
class CitationNetwork:
    """Container for citation network data and analysis results"""
    graph: nx.DiGraph
    nodes: Dict[str, NetworkNode]
    clusters: Dict[int, List[str]] = field(default_factory=dict)
    hub_papers: List[str] = field(default_factory=list)
    bridge_papers: List[str] = field(default_factory=list)

    @property
    def num_papers(self) -> int:
        return len(self.nodes)

    @property
    def num_citations(self) -> int:
        return self.graph.number_of_edges()

    @property
    def corpus_papers(self) -> List[NetworkNode]:
        """Papers that are in the original search corpus"""
        return [n for n in self.nodes.values() if n.is_in_corpus]


class CitationNetworkBuilder:
    """Builds and analyzes citation networks"""

    def __init__(self, semantic_scholar_api_key: Optional[str] = None):
        """
        Initialize the citation network builder

        Args:
            semantic_scholar_api_key: Optional API key for Semantic Scholar
        """
        self.ss_client = SemanticScholarClient(semantic_scholar_api_key)

    def build_network(
        self,
        pmids: List[str],
        max_citations_per_paper: int = 50,
        max_references_per_paper: int = 50,
        include_external: bool = True
    ) -> CitationNetwork:
        """
        Build a citation network from a list of PMIDs

        Args:
            pmids: List of PubMed IDs from search results
            max_citations_per_paper: Max citing papers to fetch per paper
            max_references_per_paper: Max references to fetch per paper
            include_external: Include papers outside the corpus in the network

        Returns:
            CitationNetwork object with graph and analysis
        """
        graph = nx.DiGraph()
        nodes: Dict[str, NetworkNode] = {}
        corpus_ids: set = set()

        # Step 1: Fetch metadata for corpus papers
        corpus_papers = self.ss_client.batch_get_papers(pmids, id_type="pmid")

        # Add corpus papers to graph
        for pmid, paper in corpus_papers.items():
            if not paper.paper_id:
                continue

            corpus_ids.add(paper.paper_id)
            node = NetworkNode(
                paper_id=paper.paper_id,
                pmid=pmid,
                title=paper.title,
                authors=paper.authors[:3],  # First 3 authors
                year=paper.year,
                citation_count=paper.citation_count,
                is_in_corpus=True
            )
            nodes[paper.paper_id] = node
            graph.add_node(paper.paper_id)

        # Step 2: Fetch citations and references for corpus papers
        for paper_id in list(corpus_ids):
            # Get citations (papers that cite this paper)
            citations = self.ss_client.get_citations(paper_id, limit=max_citations_per_paper)
            for citing_paper in citations:
                if not citing_paper.paper_id:
                    continue

                # Add citing paper if not exists
                if citing_paper.paper_id not in nodes:
                    if include_external or citing_paper.paper_id in corpus_ids:
                        nodes[citing_paper.paper_id] = NetworkNode(
                            paper_id=citing_paper.paper_id,
                            pmid=citing_paper.pmid,
                            title=citing_paper.title,
                            authors=citing_paper.authors[:3],
                            year=citing_paper.year,
                            citation_count=citing_paper.citation_count,
                            is_in_corpus=citing_paper.paper_id in corpus_ids
                        )
                        graph.add_node(citing_paper.paper_id)

                # Add edge: citing_paper -> paper_id (citation direction)
                if citing_paper.paper_id in nodes:
                    graph.add_edge(citing_paper.paper_id, paper_id)

            # Get references (papers this paper cites)
            references = self.ss_client.get_references(paper_id, limit=max_references_per_paper)
            for ref_paper in references:
                if not ref_paper.paper_id:
                    continue

                # Add referenced paper if not exists
                if ref_paper.paper_id not in nodes:
                    if include_external or ref_paper.paper_id in corpus_ids:
                        nodes[ref_paper.paper_id] = NetworkNode(
                            paper_id=ref_paper.paper_id,
                            pmid=ref_paper.pmid,
                            title=ref_paper.title,
                            authors=ref_paper.authors[:3],
                            year=ref_paper.year,
                            citation_count=ref_paper.citation_count,
                            is_in_corpus=ref_paper.paper_id in corpus_ids
                        )
                        graph.add_node(ref_paper.paper_id)

                # Add edge: paper_id -> ref_paper (this paper cites ref)
                if ref_paper.paper_id in nodes:
                    graph.add_edge(paper_id, ref_paper.paper_id)

        # Step 3: Calculate metrics
        network = CitationNetwork(graph=graph, nodes=nodes)
        self._calculate_metrics(network)
        self._detect_clusters(network)
        self._identify_key_papers(network)

        return network

    def build_network_minimal(
        self,
        pmids: List[str]
    ) -> CitationNetwork:
        """
        Build a minimal network with just corpus papers and their interconnections

        Faster than full network - only shows citations within the corpus.

        Args:
            pmids: List of PubMed IDs from search results

        Returns:
            CitationNetwork with only corpus papers
        """
        return self.build_network(
            pmids,
            max_citations_per_paper=100,
            max_references_per_paper=100,
            include_external=False
        )

    def _calculate_metrics(self, network: CitationNetwork) -> None:
        """Calculate network metrics for all nodes"""
        graph = network.graph

        # Degree calculations
        for paper_id, node in network.nodes.items():
            node.in_degree = graph.in_degree(paper_id)
            node.out_degree = graph.out_degree(paper_id)

        # PageRank
        if graph.number_of_nodes() > 0:
            try:
                pagerank = nx.pagerank(graph, alpha=0.85)
                for paper_id, score in pagerank.items():
                    if paper_id in network.nodes:
                        network.nodes[paper_id].pagerank = score
            except nx.NetworkXError:
                pass

    def _detect_clusters(self, network: CitationNetwork, min_cluster_size: int = 3) -> None:
        """Detect research clusters using community detection"""
        graph = network.graph

        if graph.number_of_nodes() < 3:
            return

        # Convert to undirected for community detection
        undirected = graph.to_undirected()

        try:
            # Use Louvain algorithm if available, otherwise greedy modularity
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(undirected)
            except (ImportError, AttributeError):
                from networkx.algorithms.community import greedy_modularity_communities
                communities = greedy_modularity_communities(undirected)

            # Assign cluster IDs
            cluster_id = 0
            for community in communities:
                if len(community) >= min_cluster_size:
                    network.clusters[cluster_id] = list(community)
                    for paper_id in community:
                        if paper_id in network.nodes:
                            network.nodes[paper_id].cluster = cluster_id
                    cluster_id += 1

        except Exception:
            # Community detection failed, skip clustering
            pass

    def _identify_key_papers(self, network: CitationNetwork) -> None:
        """Identify hub papers and bridge papers"""
        graph = network.graph

        if graph.number_of_nodes() < 3:
            return

        # Hub papers: High in-degree (highly cited within network)
        in_degrees = [(n, d) for n, d in graph.in_degree() if n in network.nodes]
        in_degrees.sort(key=lambda x: x[1], reverse=True)
        network.hub_papers = [n for n, _ in in_degrees[:10] if network.nodes[n].is_in_corpus]

        # Bridge papers: High betweenness centrality (connect different areas)
        try:
            betweenness = nx.betweenness_centrality(graph)
            bridge_candidates = [(n, score) for n, score in betweenness.items()
                                if n in network.nodes and network.nodes[n].is_in_corpus]
            bridge_candidates.sort(key=lambda x: x[1], reverse=True)
            network.bridge_papers = [n for n, _ in bridge_candidates[:10]]
        except Exception:
            pass

    def get_network_summary(self, network: CitationNetwork) -> Dict[str, Any]:
        """Get summary statistics for a citation network"""
        corpus_nodes = [n for n in network.nodes.values() if n.is_in_corpus]

        return {
            "total_papers": network.num_papers,
            "corpus_papers": len(corpus_nodes),
            "external_papers": network.num_papers - len(corpus_nodes),
            "total_citations": network.num_citations,
            "num_clusters": len(network.clusters),
            "hub_papers": len(network.hub_papers),
            "bridge_papers": len(network.bridge_papers),
            "avg_citations": sum(n.citation_count for n in corpus_nodes) / max(1, len(corpus_nodes)),
            "avg_in_degree": sum(n.in_degree for n in corpus_nodes) / max(1, len(corpus_nodes)),
            "density": nx.density(network.graph) if network.graph.number_of_nodes() > 1 else 0
        }


def create_network_visualization(
    network: CitationNetwork,
    color_by: str = "cluster",
    size_by: str = "citation_count",
    show_labels: bool = True,
    corpus_only: bool = False
) -> Optional[go.Figure]:
    """
    Create an interactive Plotly visualization of the citation network

    Args:
        network: CitationNetwork object
        color_by: How to color nodes ('cluster', 'year', 'corpus')
        size_by: How to size nodes ('citation_count', 'pagerank', 'in_degree')
        show_labels: Whether to show paper titles on hover
        corpus_only: Only show papers from the original corpus

    Returns:
        Plotly Figure object, or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None

    graph = network.graph

    # Filter to corpus only if requested
    if corpus_only:
        nodes_to_show = {n.paper_id for n in network.nodes.values() if n.is_in_corpus}
        subgraph = graph.subgraph(nodes_to_show)
    else:
        subgraph = graph
        nodes_to_show = set(network.nodes.keys())

    if len(nodes_to_show) == 0:
        return None

    # Calculate layout
    try:
        pos = nx.spring_layout(subgraph, k=2/math.sqrt(len(nodes_to_show)), iterations=50)
    except Exception:
        pos = nx.random_layout(subgraph)

    # Build edge traces
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Build node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []

    # Color scales
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for paper_id in nodes_to_show:
        if paper_id not in pos or paper_id not in network.nodes:
            continue

        node = network.nodes[paper_id]
        x, y = pos[paper_id]
        node_x.append(x)
        node_y.append(y)

        # Color
        if color_by == "cluster":
            cluster = node.cluster if node.cluster >= 0 else -1
            color = cluster_colors[cluster % len(cluster_colors)] if cluster >= 0 else '#cccccc'
        elif color_by == "year":
            if node.year:
                # Normalize year to 0-1 range (last 20 years)
                year_norm = max(0, min(1, (node.year - 2004) / 20))
                color = f'rgb({int(255 * (1 - year_norm))}, {int(100 + 155 * year_norm)}, {int(255 * year_norm)})'
            else:
                color = '#cccccc'
        else:  # corpus
            color = '#1f77b4' if node.is_in_corpus else '#cccccc'

        node_colors.append(color)

        # Size
        if size_by == "citation_count":
            size = 5 + min(30, node.citation_count / 10)
        elif size_by == "pagerank":
            size = 5 + node.pagerank * 500
        else:  # in_degree
            size = 5 + min(30, node.in_degree * 2)

        node_sizes.append(size)

        # Hover text
        authors_str = ", ".join(node.authors) if node.authors else "Unknown"
        text = f"<b>{node.title[:80]}...</b><br>" if len(node.title) > 80 else f"<b>{node.title}</b><br>"
        text += f"Authors: {authors_str}<br>"
        text += f"Year: {node.year or 'N/A'}<br>"
        text += f"Citations: {node.citation_count}<br>"
        if node.pmid:
            text += f"PMID: {node.pmid}"
        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#fff')
        )
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Citation Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig


def create_timeline_visualization(
    network: CitationNetwork,
    corpus_only: bool = True
) -> Optional[go.Figure]:
    """
    Create a timeline visualization showing citation flow over time

    Args:
        network: CitationNetwork object
        corpus_only: Only show papers from the original corpus

    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Group papers by year
    papers_by_year: Dict[int, List[NetworkNode]] = defaultdict(list)

    for node in network.nodes.values():
        if corpus_only and not node.is_in_corpus:
            continue
        if node.year:
            papers_by_year[node.year].append(node)

    if not papers_by_year:
        return None

    years = sorted(papers_by_year.keys())

    # Create bar chart of papers per year
    counts = [len(papers_by_year[y]) for y in years]
    avg_citations = [
        sum(p.citation_count for p in papers_by_year[y]) / max(1, len(papers_by_year[y]))
        for y in years
    ]

    fig = go.Figure()

    # Paper count bars
    fig.add_trace(go.Bar(
        x=years,
        y=counts,
        name='Papers',
        marker_color='#1f77b4'
    ))

    # Average citation line on secondary axis
    fig.add_trace(go.Scatter(
        x=years,
        y=avg_citations,
        name='Avg Citations',
        yaxis='y2',
        mode='lines+markers',
        marker_color='#ff7f0e'
    ))

    fig.update_layout(
        title='Publication Timeline',
        xaxis_title='Year',
        yaxis_title='Number of Papers',
        yaxis2=dict(
            title='Avg Citations',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig
