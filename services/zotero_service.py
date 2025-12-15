"""
Zotero Sync Service

Orchestrates the synchronization between Zotero Library and the internal RAG system.
Capabilities:
1. Fetch items from Zotero.
2. Download and Ingest PDFs specifically using the block-aware parser.
3. Index content into Vector DB.
4. Push AI insights back to Zotero as notes.
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

from core.zotero_client import ZoteroClient
from core.document_ingestion import DocumentIngestion, ingest_file
from core.ingestion.pipeline import IngestionPipeline

# Import settings, assuming global config access or dependency injection
from config import settings

logger = logging.getLogger(__name__)

class ZoteroSyncService:
    """
    Service to manage bidirectional sync between Zotero and RAG.
    """

    def __init__(self, zotero_client: ZoteroClient, ingestion_pipeline: IngestionPipeline):
        self.client = zotero_client
        self.pipeline = ingestion_pipeline
        self.temp_dir = Path(settings.OUTPUTS_DIR) / "temp_zotero_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def sync_collection(self, collection_key: str, limit: int = 50) -> Dict[str, int]:
        """
        Sync a Zotero collection to the Vector DB.
        
        Args:
            collection_key: Zotero collection key
            limit: Max items to process
            
        Returns:
            Stats dict: {'processed': int, 'new_indexed': int, 'errors': int}
        """
        stats = {'processed': 0, 'new_indexed': 0, 'errors': 0}
        
        try:
            items = self.client.get_items_from_collection(collection_key, limit=limit)
            
            for item in items:
                stats['processed'] += 1
                try:
                    # Check if we should process this item
                    # (In a real app, we'd check a hash or timestamp in DB before re-processing)
                    
                    # 1. Check for PDF attachment
                    children = self.client.get_item_children(item.key)
                    pdf_attachment = None
                    for att in children['attachments']:
                        if att['contentType'] == 'application/pdf':
                            pdf_attachment = att
                            break
                    
                    if pdf_attachment:
                        # 2. Download PDF
                        file_path = self.temp_dir / f"{item.key}.pdf"
                        success = self.client.download_item_attachment(pdf_attachment['key'], str(file_path))
                        
                        if success:
                            # 3. Ingest (Block-aware parsing)
                            # We use ingest_file helper which uses DocumentIngestion
                            with open(file_path, 'rb') as f:
                                doc = ingest_file(f, filename=pdf_attachment['filename'])
                            
                            # Update metadata with Zotero Key for provenance
                            doc.metadata['zotero_key'] = item.key
                            doc.metadata['zotero_link'] = item.url
                            
                            # 4. Index
                            self.pipeline.run([doc])
                            stats['new_indexed'] += 1
                            
                            # Cleanup
                            if file_path.exists():
                                file_path.unlink()
                        else:
                            logger.warning(f"Failed to download PDF for {item.title}")
                    else:
                        # If no PDF, index the metadata/abstract
                        # (Implementation omitted for brevity, focusing on PDF sync)
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing item {item.key}: {e}")
                    stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Sync failed for collection {collection_key}: {e}")
            
        return stats

    def push_insight_to_zotero(self, zotero_key: str, insight: str, source: str = "AI Expert"):
        """
        Push an AI insight (critique/summary) back to Zotero as a note.
        """
        note_html = f"<b>{source} Insight:</b><br/>{insight}"
        self.client.create_note(parent_item_key=zotero_key, content=note_html, tags=["AI-Insight"])

    def cleanup(self):
        """Cleanup temp files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
