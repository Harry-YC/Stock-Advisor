"""
Zotero API Client

Direct integration with Zotero reference manager via API.
Allows:
- Creating collections
- Adding citations to library
- Bulk uploads
- Importing from Zotero collections (bidirectional sync)
- Fetching annotations and notes
- Tag synchronization

Requires: ZOTERO_API_KEY and ZOTERO_USER_ID
Get your API key from: https://www.zotero.org/settings/keys
"""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from core.pubmed_client import Citation


@dataclass
class ZoteroItem:
    """Represents an item from Zotero library"""
    key: str
    item_type: str
    title: str
    creators: List[Dict[str, str]] = field(default_factory=list)
    abstract: str = ""
    publication_title: str = ""
    date: str = ""
    doi: str = ""
    url: str = ""
    pmid: str = ""
    tags: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def to_citation(self) -> Citation:
        """Convert ZoteroItem to Citation object for internal use"""
        authors = []
        for creator in self.creators:
            if creator.get("name"):
                authors.append(creator["name"])
            elif creator.get("firstName") and creator.get("lastName"):
                authors.append(f"{creator['firstName']} {creator['lastName']}")
            elif creator.get("lastName"):
                authors.append(creator["lastName"])

        return Citation(
            pmid=self.pmid or "",
            title=self.title,
            authors=authors,
            journal=self.publication_title,
            year=self.date[:4] if self.date else "",
            abstract=self.abstract,
            doi=self.doi
        )


class ZoteroClient:
    """Client for Zotero API integration"""

    BASE_URL = "https://api.zotero.org"

    def __init__(self, api_key: str, user_id: str):
        """
        Initialize Zotero client

        Args:
            api_key: Zotero API key from https://www.zotero.org/settings/keys
            user_id: Your Zotero user ID (numeric)
        """
        self.api_key = api_key
        self.user_id = user_id
        self.headers = {
            "Zotero-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def test_connection(self) -> bool:
        """Test if API credentials are valid"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/users/{self.user_id}/collections",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def create_collection(self, name: str, parent_collection: Optional[str] = None) -> Dict:
        """
        Create a new collection in Zotero library

        Args:
            name: Collection name
            parent_collection: Parent collection key (optional)

        Returns:
            Dict with collection info including 'key'
        """
        payload = [{
            "name": name,
            "parentCollection": parent_collection if parent_collection else False
        }]

        response = requests.post(
            f"{self.BASE_URL}/users/{self.user_id}/collections",
            headers=self.headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if 'successful' in result and '0' in result['successful']:
                return result['successful']['0']
            raise Exception(f"Collection creation failed: {response.text}")
        else:
            raise Exception(f"API error {response.status_code}: {response.text}")

    def add_citations(self, citations: List[Citation], collection_key: Optional[str] = None) -> Dict:
        """
        Add citations to Zotero library

        Args:
            citations: List of Citation objects
            collection_key: Optional collection to add items to

        Returns:
            Dict with success/failure counts
        """
        # Convert citations to Zotero item format
        items = []
        for citation in citations:
            item = self._citation_to_zotero_item(citation)
            if collection_key:
                item['collections'] = [collection_key]
            items.append(item)

        # Zotero API allows max 50 items per request
        batch_size = 50
        results = {
            'successful': 0,
            'failed': 0,
            'total': len(items)
        }

        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]

            response = requests.post(
                f"{self.BASE_URL}/users/{self.user_id}/items",
                headers=self.headers,
                json=batch,
                timeout=60
            )

            if response.status_code == 200:
                batch_result = response.json()
                if 'successful' in batch_result:
                    results['successful'] += len(batch_result['successful'])
                if 'failed' in batch_result:
                    results['failed'] += len(batch_result['failed'])
            else:
                results['failed'] += len(batch)

        return results

    def get_collections(self, include_item_counts: bool = True) -> List[Dict]:
        """
        Get all collections in the library

        Args:
            include_item_counts: If True, include number of items in each collection

        Returns:
            List of collection dicts with keys: key, name, parentCollection, numItems
        """
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/collections",
            headers=self.headers,
            timeout=30
        )

        if response.status_code == 200:
            collections = response.json()
            # Flatten to simpler format
            result = []
            for col in collections:
                data = col.get("data", col)
                result.append({
                    "key": data.get("key", ""),
                    "name": data.get("name", ""),
                    "parentCollection": data.get("parentCollection", False),
                    "numItems": col.get("meta", {}).get("numItems", 0) if include_item_counts else None
                })
            return result
        else:
            raise Exception(f"Failed to fetch collections: {response.text}")

    def get_items_from_collection(
        self,
        collection_key: str,
        limit: int = 100,
        item_types: Optional[List[str]] = None
    ) -> List[ZoteroItem]:
        """
        Import items from a Zotero collection

        Args:
            collection_key: The collection key to fetch items from
            limit: Maximum number of items to fetch (default 100, max 100 per request)
            item_types: Optional list of item types to filter (e.g., ['journalArticle', 'book'])

        Returns:
            List of ZoteroItem objects
        """
        items = []
        start = 0

        while True:
            params = {
                "limit": min(limit - len(items), 100),
                "start": start,
                "sort": "dateModified",
                "direction": "desc"
            }

            if item_types:
                params["itemType"] = " || ".join(item_types)

            response = requests.get(
                f"{self.BASE_URL}/users/{self.user_id}/collections/{collection_key}/items",
                headers=self.headers,
                params=params,
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"Failed to fetch items: {response.text}")

            batch = response.json()
            if not batch:
                break

            for item_data in batch:
                data = item_data.get("data", item_data)

                # Skip attachments and notes at top level
                if data.get("itemType") in ["attachment", "note"]:
                    continue

                # Extract PMID from extra field
                pmid = ""
                extra = data.get("extra", "")
                if "PMID:" in extra:
                    for line in extra.split("\n"):
                        if line.startswith("PMID:"):
                            pmid = line.replace("PMID:", "").strip()
                            break

                zotero_item = ZoteroItem(
                    key=data.get("key", ""),
                    item_type=data.get("itemType", ""),
                    title=data.get("title", ""),
                    creators=data.get("creators", []),
                    abstract=data.get("abstractNote", ""),
                    publication_title=data.get("publicationTitle", ""),
                    date=data.get("date", ""),
                    doi=data.get("DOI", ""),
                    url=data.get("url", ""),
                    pmid=pmid,
                    tags=[t.get("tag", "") for t in data.get("tags", [])],
                    collections=data.get("collections", [])
                )
                items.append(zotero_item)

            start += len(batch)
            if len(items) >= limit or len(batch) < 100:
                break

        return items

    def get_all_items(
        self,
        limit: int = 500,
        item_types: Optional[List[str]] = None
    ) -> List[ZoteroItem]:
        """
        Get all items from library (not limited to a collection)

        Args:
            limit: Maximum items to fetch
            item_types: Optional filter by item type

        Returns:
            List of ZoteroItem objects
        """
        items = []
        start = 0

        while True:
            params = {
                "limit": min(limit - len(items), 100),
                "start": start,
                "sort": "dateModified",
                "direction": "desc"
            }

            if item_types:
                params["itemType"] = " || ".join(item_types)

            response = requests.get(
                f"{self.BASE_URL}/users/{self.user_id}/items",
                headers=self.headers,
                params=params,
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"Failed to fetch items: {response.text}")

            batch = response.json()
            if not batch:
                break

            for item_data in batch:
                data = item_data.get("data", item_data)

                if data.get("itemType") in ["attachment", "note"]:
                    continue

                pmid = ""
                extra = data.get("extra", "")
                if "PMID:" in extra:
                    for line in extra.split("\n"):
                        if line.startswith("PMID:"):
                            pmid = line.replace("PMID:", "").strip()
                            break

                zotero_item = ZoteroItem(
                    key=data.get("key", ""),
                    item_type=data.get("itemType", ""),
                    title=data.get("title", ""),
                    creators=data.get("creators", []),
                    abstract=data.get("abstractNote", ""),
                    publication_title=data.get("publicationTitle", ""),
                    date=data.get("date", ""),
                    doi=data.get("DOI", ""),
                    url=data.get("url", ""),
                    pmid=pmid,
                    tags=[t.get("tag", "") for t in data.get("tags", [])],
                    collections=data.get("collections", [])
                )
                items.append(zotero_item)

            start += len(batch)
            if len(items) >= limit or len(batch) < 100:
                break

        return items

    def get_item_children(self, item_key: str) -> Dict[str, Any]:
        """
        Get child items (notes, attachments) for a parent item

        Args:
            item_key: The parent item key

        Returns:
            Dict with 'notes' and 'attachments' lists
        """
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/items/{item_key}/children",
            headers=self.headers,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch children: {response.text}")

        children = response.json()
        notes = []
        attachments = []

        for child in children:
            data = child.get("data", child)
            if data.get("itemType") == "note":
                notes.append({
                    "key": data.get("key", ""),
                    "content": data.get("note", ""),
                    "dateModified": data.get("dateModified", "")
                })
            elif data.get("itemType") == "attachment":
                attachments.append({
                    "key": data.get("key", ""),
                    "title": data.get("title", ""),
                    "contentType": data.get("contentType", ""),
                    "filename": data.get("filename", ""),
                    "linkMode": data.get("linkMode", "")
                })

        return {"notes": notes, "attachments": attachments}

    def get_item_annotations(self, attachment_key: str) -> List[Dict[str, Any]]:
        """
        Get PDF annotations for an attachment

        Args:
            attachment_key: The attachment item key (PDF)

        Returns:
            List of annotation dicts with text, comment, color, page, etc.
        """
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/items/{attachment_key}/children",
            headers=self.headers,
            params={"itemType": "annotation"},
            timeout=30
        )

        if response.status_code != 200:
            return []

        annotations = []
        for item in response.json():
            data = item.get("data", item)
            if data.get("itemType") == "annotation":
                annotations.append({
                    "key": data.get("key", ""),
                    "type": data.get("annotationType", ""),
                    "text": data.get("annotationText", ""),
                    "comment": data.get("annotationComment", ""),
                    "color": data.get("annotationColor", ""),
                    "page": data.get("annotationPageLabel", ""),
                    "position": data.get("annotationPosition", {}),
                    "dateModified": data.get("dateModified", "")
                })

        return annotations

    def get_item_with_annotations(self, item_key: str) -> ZoteroItem:
        """
        Get an item with all its notes and PDF annotations

        Args:
            item_key: The item key

        Returns:
            ZoteroItem with notes and annotations populated
        """
        # Get the main item
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/items/{item_key}",
            headers=self.headers,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch item: {response.text}")

        item_data = response.json()
        data = item_data.get("data", item_data)

        # Extract PMID
        pmid = ""
        extra = data.get("extra", "")
        if "PMID:" in extra:
            for line in extra.split("\n"):
                if line.startswith("PMID:"):
                    pmid = line.replace("PMID:", "").strip()
                    break

        zotero_item = ZoteroItem(
            key=data.get("key", ""),
            item_type=data.get("itemType", ""),
            title=data.get("title", ""),
            creators=data.get("creators", []),
            abstract=data.get("abstractNote", ""),
            publication_title=data.get("publicationTitle", ""),
            date=data.get("date", ""),
            doi=data.get("DOI", ""),
            url=data.get("url", ""),
            pmid=pmid,
            tags=[t.get("tag", "") for t in data.get("tags", [])],
            collections=data.get("collections", [])
        )

        # Get children (notes and attachments)
        children = self.get_item_children(item_key)
        zotero_item.notes = [n["content"] for n in children["notes"]]

        # Get annotations from PDF attachments
        all_annotations = []
        for attachment in children["attachments"]:
            if attachment.get("contentType") == "application/pdf":
                annotations = self.get_item_annotations(attachment["key"])
                all_annotations.extend(annotations)

        zotero_item.annotations = all_annotations

        return zotero_item

    def add_tags_to_item(self, item_key: str, tags: List[str]) -> bool:
        """
        Add tags to an existing item

        Args:
            item_key: The item key
            tags: List of tag strings to add

        Returns:
            True if successful
        """
        # First get current item to get version
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/items/{item_key}",
            headers=self.headers,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch item: {response.text}")

        item_data = response.json()
        version = item_data.get("version", 0)
        current_tags = item_data.get("data", {}).get("tags", [])

        # Add new tags (avoid duplicates)
        existing_tag_names = {t.get("tag", "") for t in current_tags}
        for tag in tags:
            if tag not in existing_tag_names:
                current_tags.append({"tag": tag})

        # Update item
        update_headers = {**self.headers, "If-Unmodified-Since-Version": str(version)}
        response = requests.patch(
            f"{self.BASE_URL}/users/{self.user_id}/items/{item_key}",
            headers=update_headers,
            json={"tags": current_tags},
            timeout=30
        )

        return response.status_code in [200, 204]

    def sync_tags_to_zotero(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Sync tags from internal items to Zotero

        Args:
            items: List of dicts with 'zotero_key' and 'tags' fields

        Returns:
            Dict with 'successful' and 'failed' counts
        """
        results = {"successful": 0, "failed": 0}

        for item in items:
            zotero_key = item.get("zotero_key")
            tags = item.get("tags", [])

            if not zotero_key or not tags:
                continue

            try:
                if self.add_tags_to_item(zotero_key, tags):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
            except Exception:
                results["failed"] += 1

        return results

    def search_items(self, query: str, limit: int = 50) -> List[ZoteroItem]:
        """
        Search items in Zotero library

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of matching ZoteroItem objects
        """
        response = requests.get(
            f"{self.BASE_URL}/users/{self.user_id}/items",
            headers=self.headers,
            params={
                "q": query,
                "limit": limit,
                "sort": "relevance"
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Search failed: {response.text}")

        items = []
        for item_data in response.json():
            data = item_data.get("data", item_data)

            if data.get("itemType") in ["attachment", "note"]:
                continue

            pmid = ""
            extra = data.get("extra", "")
            if "PMID:" in extra:
                for line in extra.split("\n"):
                    if line.startswith("PMID:"):
                        pmid = line.replace("PMID:", "").strip()
                        break

            zotero_item = ZoteroItem(
                key=data.get("key", ""),
                item_type=data.get("itemType", ""),
                title=data.get("title", ""),
                creators=data.get("creators", []),
                abstract=data.get("abstractNote", ""),
                publication_title=data.get("publicationTitle", ""),
                date=data.get("date", ""),
                doi=data.get("DOI", ""),
                url=data.get("url", ""),
                pmid=pmid,
                tags=[t.get("tag", "") for t in data.get("tags", [])],
                collections=data.get("collections", [])
            )
            items.append(zotero_item)

        return items
    def download_item_attachment(self, item_key: str, save_path: str) -> bool:
        """
        Download attachment file (e.g. PDF) for an item.
        
        Args:
            item_key: The Zotero key of the attachment item
            save_path: Local path to save the file to
            
        Returns:
            True if successful, False otherwise
        """
        # Zotero API file download endpoint: /users/{userID}/items/{itemKey}/file
        url = f"{self.BASE_URL}/users/{self.user_id}/items/{item_key}/file"
        
        try:
            with requests.get(url, headers=self.headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            # logger.error(f"Failed to download Zotero attachment {item_key}: {e}")
            print(f"Failed to download Zotero attachment {item_key}: {e}")
            return False

    def create_note(self, parent_item_key: str, content: str, tags: List[str] = None) -> str:
        """
        Create a child note for a specific parent item.
        
        Args:
            parent_item_key: Key of the parent item
            content: HTML content of the note
            tags: Optional list of tags
            
        Returns:
            Key of the created note
        """
        note_item = {
            "itemType": "note",
            "parentItem": parent_item_key,
            "note": content,
            "tags": [{"tag": t} for t in (tags or [])]
        }
        
        response = requests.post(
            f"{self.BASE_URL}/users/{self.user_id}/items",
            headers=self.headers,
            json=[note_item],
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'successful' in result and '0' in result['successful']:
                return result['successful']['0']['key']
        
        raise Exception(f"Failed to create note: {response.text}")

    def _citation_to_zotero_item(self, citation: Citation) -> Dict:
        """
        Convert Citation object to Zotero item format

        Zotero API documentation: https://www.zotero.org/support/dev/web_api/v3/write_requests
        """
        # Create authors array
        creators = []
        if citation.authors:
            for author in citation.authors:
                # Try to split into first/last name
                parts = author.split()
                if len(parts) >= 2:
                    creators.append({
                        "creatorType": "author",
                        "firstName": " ".join(parts[:-1]),
                        "lastName": parts[-1]
                    })
                else:
                    creators.append({
                        "creatorType": "author",
                        "name": author
                    })

        # Clean DOI
        doi = ""
        if citation.doi:
            doi = citation.doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')

        # Build Zotero item
        item = {
            "itemType": "journalArticle",
            "title": citation.title or "",
            "creators": creators,
            "abstractNote": citation.abstract or "",
            "publicationTitle": citation.journal or "",
            "date": citation.year or "",
            "DOI": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/" if citation.pmid else "",
            "extra": f"PMID: {citation.pmid}" if citation.pmid else "",
            "tags": [{"tag": "PubMed"}]
        }

        return item


def get_user_id_from_api_key(api_key: str) -> Optional[str]:
    """
    Helper function to get user ID from API key

    Args:
        api_key: Zotero API key

    Returns:
        User ID as string, or None if failed
    """
    try:
        response = requests.get(
            "https://api.zotero.org/keys/current",
            headers={"Zotero-API-Key": api_key},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            return str(data.get('userID', ''))
        return None
    except Exception:
        return None
