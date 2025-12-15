import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.image_analyzer import detect_image_type, analyze_image, format_for_expert_context, ImageAnalysis

class TestImageAnalyzer(unittest.TestCase):
    
    @patch('core.image_analyzer.genai.GenerativeModel')
    def test_detect_image_type(self, mock_model_class):
        # Mock the model and response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "kaplan_meier"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Test detection
        image_data = b"fake_image_bytes"
        result = detect_image_type(image_data)
        
        self.assertEqual(result, "kaplan_meier")
        mock_model.generate_content.assert_called_once()
        
    @patch('core.image_analyzer.genai.GenerativeModel')
    @patch('core.image_analyzer.detect_image_type')
    def test_analyze_image(self, mock_detect, mock_model_class):
        # Mock type detection
        mock_detect.return_value = "kaplan_meier"
        
        # Mock Gemini response for analysis
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_json = """
        {
            "summary": "This is a survival curve.",
            "key_findings": ["Finding 1", "Finding 2"],
            "clinical_implications": "Drug X works better.",
            "limitations": "No p-value."
        }
        """
        mock_response.text = mock_json
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Test analysis
        image_data = b"fake_image_bytes"
        result = analyze_image(image_data, "test.jpg")
        
        self.assertIsInstance(result, ImageAnalysis)
        self.assertEqual(result.image_type, "kaplan_meier")
        self.assertEqual(result.summary, "This is a survival curve.")
        self.assertEqual(len(result.key_findings), 2)
        
    def test_format_for_expert_context(self):
        analyses = [
            ImageAnalysis(
                image_type="kaplan_meier",
                summary="Summary 1",
                key_findings=["Finding A", "Finding B"],
                clinical_implications="Implication 1",
                limitations="",
                raw_description="",
                extracted_data={}
            )
        ]
        
        text = format_for_expert_context(analyses)
        
        self.assertIn("[IMAGE ANALYSIS CONTEXT]", text)
        self.assertIn("kaplan_meier", text)
        self.assertIn("Summary 1", text)
        self.assertIn("Finding A", text)
        self.assertIn("Implication 1", text)

if __name__ == '__main__':
    unittest.main()
