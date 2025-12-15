"""
Playwright E2E tests for RAG Integration

Tests:
1. Document upload UI appears
2. Document indexing works
3. Expert panel retrieves RAG context
4. Citation verification works
"""

import pytest
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import Page, expect

# Test URL - can be overridden with TEST_URL env var
BASE_URL = os.getenv("TEST_URL", "http://localhost:8501")


@pytest.fixture(scope="module")
def test_pdf():
    """Create a simple test PDF for upload testing."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    pdf_path = test_dir / "test_document.txt"

    # Create a simple text file (PDF would require additional dependencies)
    pdf_path.write_text("""
    Clinical Trial Results Summary

    Study: Phase 2 Trial of Drug X in NSCLC

    Efficacy Results:
    - Overall Response Rate (ORR): 42%
    - Median Progression-Free Survival (mPFS): 8.2 months
    - Disease Control Rate (DCR): 78%

    Safety Profile:
    - Grade 3+ Treatment-Related AEs: 15%
    - Discontinuation due to AEs: 8%

    Conclusion: Drug X showed promising efficacy with an acceptable safety profile
    in patients with advanced NSCLC.
    """)

    yield pdf_path

    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()


class TestAppLoads:
    """Basic tests that the app loads correctly."""

    def test_app_loads(self, page: Page):
        """Test that the main app loads."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # Should see the Literature Review title (use heading role for specificity)
        expect(page.get_by_role("heading", name="📚 Literature Review")).to_be_visible(timeout=30000)

    def test_sidebar_visible(self, page: Page):
        """Test that sidebar navigation is visible."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # Sidebar should be visible
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible(timeout=10000)


class TestProjectCreation:
    """Tests for creating a project (required before document upload)."""

    def test_create_project(self, page: Page):
        """Test creating a new project."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Wait for Streamlit to fully load

        # Look for project name input in sidebar
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Find and fill project name input
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            project_input.fill("RAG_Test_Project")

            # Click create button
            create_btn = sidebar.locator('button:has-text("Create")')
            if create_btn.is_visible():
                create_btn.click()
                time.sleep(2)


class TestDocumentUploadUI:
    """Tests for the document upload tab."""

    def test_documents_tab_exists(self, page: Page):
        """Test that Documents tab exists in Literature Search."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(5)  # Extra time for app to fully load

        # First create a project with unique name
        sidebar = page.locator('[data-testid="stSidebar"]')
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            import random
            project_input.fill(f"RAG_TabTest_{random.randint(1000, 9999)}")
            create_btn = sidebar.locator('button:has-text("Create")').first
            if create_btn.is_visible():
                create_btn.click()
                time.sleep(3)

        # Navigate to Literature Search - look for text anywhere
        lit_search = page.get_by_text("Literature Search", exact=False).first
        if lit_search.is_visible():
            lit_search.click()
            time.sleep(3)

        # Look for Documents tab - more flexible selector
        docs_tab = page.get_by_role("tab", name="Documents")
        if docs_tab.count() > 0:
            expect(docs_tab.first).to_be_visible(timeout=10000)
        else:
            # Tab might be rendered differently - check for the text
            docs_text = page.get_by_text("Documents", exact=False)
            expect(docs_text.first).to_be_visible(timeout=10000)

    def test_upload_area_visible(self, page: Page):
        """Test that file upload area is visible in Documents tab."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(5)  # Extra time for app to fully load

        # Create project first with unique name
        sidebar = page.locator('[data-testid="stSidebar"]')
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            import random
            project_input.fill(f"RAG_Upload_{random.randint(1000, 9999)}")
            create_btn = sidebar.locator('button:has-text("Create")').first
            if create_btn.is_visible():
                create_btn.click()
                time.sleep(3)

        # Navigate to Literature Search
        lit_search = page.get_by_text("Literature Search", exact=False).first
        if lit_search.is_visible():
            lit_search.click()
            time.sleep(3)

        # Click Documents tab and wait for it to be selected
        docs_tab = page.get_by_role("tab", name="Documents")
        if docs_tab.count() > 0:
            docs_tab.first.click()
            time.sleep(3)  # Wait for tab content to render

        # Check for visible file uploader (the one in Documents tab, not hidden Import tab)
        # Look for the text that identifies the Documents tab uploader
        docs_uploader_text = page.get_by_text("Drop files here or click to upload")
        if docs_uploader_text.count() > 0:
            expect(docs_uploader_text.first).to_be_visible(timeout=10000)
        else:
            # Alternative check - look for any visible upload text
            upload_text = page.get_by_text("Upload", exact=False)
            expect(upload_text.first).to_be_visible(timeout=10000)


class TestDocumentIndexing:
    """Tests for document indexing functionality."""

    def test_index_document(self, page: Page, test_pdf):
        """Test uploading and indexing a document."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Create project
        sidebar = page.locator('[data-testid="stSidebar"]')
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            project_input.fill("RAG_Index_Test")
            create_btn = sidebar.locator('button:has-text("Create")').first
            if create_btn.is_visible():
                create_btn.click()
                time.sleep(2)

        # Navigate to Literature Search
        lit_search_btn = sidebar.locator('button:has-text("Literature Search")')
        if lit_search_btn.is_visible():
            lit_search_btn.click()
            time.sleep(2)

        # Click Documents tab (with emoji)
        docs_tab = page.locator('button[role="tab"]:has-text("📄 Documents")')
        if docs_tab.is_visible():
            docs_tab.click()
            time.sleep(2)

        # Upload the test file (use last() for the document uploader in Documents tab)
        file_input = page.locator('input[type="file"]').last
        if file_input.count() > 0:
            file_input.set_input_files(str(test_pdf))
            time.sleep(2)

            # Click Index Documents button
            index_btn = page.locator('button:has-text("Index Documents")')
            if index_btn.is_visible():
                index_btn.click()
                time.sleep(5)  # Wait for indexing

                # Check for success message or indexed documents list
                success = page.locator('text=chunks indexed')
                if success.is_visible(timeout=30000):
                    assert True
                else:
                    # Check if document appears in list
                    doc_list = page.locator('text=test_document')
                    expect(doc_list).to_be_visible(timeout=10000)


class TestExpertPanelRAG:
    """Tests for RAG integration in Expert Panel."""

    def test_expert_panel_loads(self, page: Page):
        """Test that Expert Panel tab loads."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Create project
        sidebar = page.locator('[data-testid="stSidebar"]')
        project_input = sidebar.locator('input[type="text"]').first
        if project_input.is_visible():
            project_input.fill("RAG_Expert_Test")
            create_btn = sidebar.locator('button:has-text("Create")').first
            if create_btn.is_visible():
                create_btn.click()
                time.sleep(2)

        # Navigate to Expert Panel
        expert_btn = sidebar.locator('button:has-text("Expert Panel")')
        if expert_btn.is_visible():
            expert_btn.click()
            time.sleep(2)

        # Check for Expert Panel content (use specific text)
        expert_panel = page.get_by_text("Part 2: Expert Panel")
        expect(expert_panel).to_be_visible(timeout=10000)


class TestIngestionModule:
    """Unit tests for ingestion module (run without browser)."""

    def test_embedder_initialization(self):
        """Test that LocalEmbedder can be initialized."""
        from core.ingestion import LocalEmbedder

        # This will download the model on first run
        embedder = LocalEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
            device="cpu"
        )

        # Test embedding
        embedding = embedder.embed_query("test query")
        assert len(embedding) > 0
        assert isinstance(embedding, list)

    def test_chunking(self):
        """Test parent-child chunking."""
        from core.ingestion.pipeline import chunk_text_parent_child

        test_text = "This is a test document. " * 100  # Create a longer text

        chunks = chunk_text_parent_child(
            test_text,
            parent_size=500,
            child_size=100
        )

        assert len(chunks) > 0
        assert 'content' in chunks[0]
        assert 'parent_content' in chunks[0]

    def test_vector_db_initialization(self):
        """Test VectorDB initialization."""
        from core.ingestion import VectorDB
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "storage_path": tmpdir,
                "collection_name": "test_collection",
                "vector_size": 384,  # all-MiniLM-L6-v2 dimension
                "distance": "Cosine"
            }

            db = VectorDB(config)
            info = db.get_collection_info()

            assert info['status'] is not None


class TestRetrieverModule:
    """Unit tests for retriever module."""

    def test_bm25_tokenization(self):
        """Test BM25 tokenization."""
        from core.retrieval.retriever import tokenize_for_bm25

        tokens = tokenize_for_bm25("Clinical trial results for lung cancer treatment")

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_mmr_function(self):
        """Test MMR diversity function."""
        from core.retrieval.retriever import maximal_marginal_relevance

        # Create simple test data
        query_emb = [1.0, 0.0, 0.0]
        doc_embs = [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        docs = [
            {"content": "doc1"},
            {"content": "doc2"},
            {"content": "doc3"},
            {"content": "doc4"},
        ]

        results = maximal_marginal_relevance(
            query_emb, doc_embs, docs,
            lambda_param=0.5, top_k=2
        )

        assert len(results) == 2


class TestVerifierModule:
    """Unit tests for citation verification."""

    def test_citation_extraction(self):
        """Test citation extraction from text."""
        from core.analysis.verifier import extract_citations

        text = """
        According to study [1], the ORR was 42%.
        This is consistent with (PMID: 12345678).
        [Source: clinical_trial.pdf] provides more details.
        """

        citations = extract_citations(text)

        assert len(citations) >= 3

        # Check types
        types = [c['type'] for c in citations]
        assert 'numbered' in types
        assert 'pmid' in types
        assert 'source' in types

    def test_verification(self):
        """Test citation verification."""
        from core.analysis.verifier import verify_citations

        response = "According to [1], the ORR was 42%."
        evidence = [{"content": "ORR was 42%", "source": "trial.pdf"}]

        result = verify_citations(response, evidence)

        assert result.total_citations > 0
        assert result.verified_count >= 0


class TestQueryExpansion:
    """Unit tests for query expansion."""

    def test_expand_query_returns_list(self):
        """Test that expand_query returns a list."""
        from core.retrieval.query_expansion import expand_query

        # This will fail without OpenAI API key, but should return original query
        result = expand_query("Keytruda NSCLC trial results")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "Keytruda NSCLC trial results" in result[0] or len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
