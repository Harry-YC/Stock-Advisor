
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integrations.google_search import GoogleSearchClient, GroundingSource, search_with_grounding

class TestGoogleSearchIntegration(unittest.TestCase):

    def setUp(self):
        self.api_key = "fake_key"
        self.client = GoogleSearchClient(api_key=self.api_key)

    @patch('integrations.google_search.GoogleSearchClient._get_model')
    def test_generate_with_grounding_success(self, mock_get_model):
        # Mock the model and response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response content"
        
        # Mock grounding metadata
        mock_candidate = MagicMock()
        mock_metadata = MagicMock()
        
        # Mock web source
        mock_chunk = MagicMock()
        mock_chunk.web.title = "Test Source"
        mock_chunk.web.uri = "http://test.com"
        mock_metadata.grounding_chunks = [mock_chunk]
        
        mock_candidate.grounding_metadata = mock_metadata
        mock_response.candidates = [mock_candidate]
        
        mock_model.generate_content.return_value = mock_response
        mock_get_model.return_value = mock_model

        # Execute
        response = self.client.generate_with_grounding("test prompt")

        # Verify
        self.assertEqual(response.content, "Grounded response content")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].title, "Test Source")
        self.assertEqual(response.sources[0].url, "http://test.com")

    @patch('integrations.google_search.GoogleSearchClient')
    def test_search_with_grounding_convenience(self, MockClient):
        # Setup mock client instance
        mock_instance = MockClient.return_value
        mock_instance.is_available.return_value = True
        
        mock_result = MagicMock()
        mock_result.content = "Search result"
        mock_result.sources = [GroundingSource(title="Test", url="http://test.com")]
        
        mock_instance.generate_with_grounding.return_value = mock_result

        # Execute
        content, sources = search_with_grounding("query")

        # Verify
        self.assertEqual(content, "Search result")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['title'], "Test")

if __name__ == '__main__':
    unittest.main()
