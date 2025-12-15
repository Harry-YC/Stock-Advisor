"""
Mock Tests for Zotero Sync Service
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
from services.zotero_service import ZoteroSyncService
from core.zotero_client import ZoteroClient, ZoteroItem
from core.ingestion.pipeline import IngestionPipeline

# Mock data
MOCK_ITEM = ZoteroItem(
    key="ITEM123",
    item_type="journalArticle",
    title="Deep Learning in Genomics",
    url="http://example.com/paper",
    collections=["COLL1"]
)

MOCK_CHILDREN = {
    "attachments": [
        {
            "key": "ATT1",
            "contentType": "application/pdf",
            "filename": "paper.pdf"
        }
    ],
    "notes": []
}

@pytest.fixture
def mock_deps():
    client = MagicMock(spec=ZoteroClient)
    pipeline = MagicMock(spec=IngestionPipeline)
    return client, pipeline

class TestZoteroSync:
    
    def test_sync_collection_downloads_pdf(self, mock_deps):
        client, pipeline = mock_deps
        service = ZoteroSyncService(client, pipeline)
        
        # Setup mocks
        client.get_items_from_collection.return_value = [MOCK_ITEM]
        client.get_item_children.return_value = MOCK_CHILDREN
        client.download_item_attachment.return_value = True
        
        # Mock file operations
        with patch("builtins.open", mock_open(read_data=b"%PDF-1.4...")):
            with patch("services.zotero_service.ingest_file") as mock_ingest:
                mock_doc = MagicMock()
                mock_doc.metadata = {}
                mock_ingest.return_value = mock_doc
                
                # Execute
                stats = service.sync_collection("COLL1")
                
                # Verify
                assert stats['processed'] == 1
                assert stats['new_indexed'] == 1
                
                # Check calls
                client.download_item_attachment.assert_called_with("ATT1", ANY)
                pipeline.run.assert_called_once()
                assert mock_doc.metadata['zotero_key'] == "ITEM123"

    def test_pdf_download_failure_handling(self, mock_deps):
        client, pipeline = mock_deps
        service = ZoteroSyncService(client, pipeline)
        
        client.get_items_from_collection.return_value = [MOCK_ITEM]
        client.get_item_children.return_value = MOCK_CHILDREN
        client.download_item_attachment.return_value = False  # Failure
        
        stats = service.sync_collection("COLL1")
        
        assert stats['processed'] == 1
        assert stats['new_indexed'] == 0  # Should not index if PDF fails
        pipeline.run.assert_not_called()

    def test_push_insight(self, mock_deps):
        client, pipeline = mock_deps
        service = ZoteroSyncService(client, pipeline)
        
        service.push_insight_to_zotero("ITEM123", "Great paper!")
        
        client.create_note.assert_called_once_with(
            parent_item_key="ITEM123",
            content="<b>AI Expert Insight:</b><br/>Great paper!",
            tags=["AI-Insight"]
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
