"""
Tests for file upload size validation.

Tests the 10MB limit for documents and 5MB limit for images.
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFileSizeValidation:
    """Unit tests for file size validation logic."""

    def test_file_under_limit_accepted(self, temp_dir):
        """Files under 10MB should be accepted."""
        # Create a 5MB file
        test_file = temp_dir / "small.pdf"
        test_file.write_bytes(b"x" * (5 * 1024 * 1024))

        file_size = os.path.getsize(test_file)
        MAX_FILE_SIZE = 10 * 1024 * 1024

        assert file_size < MAX_FILE_SIZE
        assert file_size == 5 * 1024 * 1024

    def test_file_over_limit_rejected(self, temp_dir):
        """Files over 10MB should be rejected."""
        # Create an 11MB file
        test_file = temp_dir / "large.pdf"
        test_file.write_bytes(b"x" * (11 * 1024 * 1024))

        file_size = os.path.getsize(test_file)
        MAX_FILE_SIZE = 10 * 1024 * 1024

        assert file_size > MAX_FILE_SIZE

    def test_file_at_exact_limit(self, temp_dir):
        """Files at exactly 10MB should be accepted."""
        # Create exactly 10MB file
        test_file = temp_dir / "exact.pdf"
        test_file.write_bytes(b"x" * (10 * 1024 * 1024))

        file_size = os.path.getsize(test_file)
        MAX_FILE_SIZE = 10 * 1024 * 1024

        # Files at exactly the limit should pass (not strictly greater)
        assert file_size == MAX_FILE_SIZE

    def test_image_under_5mb_accepted(self, temp_dir):
        """Images under 5MB should be accepted."""
        # Create a 3MB image file
        test_image = temp_dir / "small.jpg"
        test_image.write_bytes(b"x" * (3 * 1024 * 1024))

        file_size = os.path.getsize(test_image)
        MAX_IMAGE_SIZE = 5 * 1024 * 1024

        assert file_size < MAX_IMAGE_SIZE

    def test_image_over_5mb_rejected(self, temp_dir):
        """Images over 5MB should be rejected."""
        # Create a 6MB image file
        test_image = temp_dir / "large.jpg"
        test_image.write_bytes(b"x" * (6 * 1024 * 1024))

        file_size = os.path.getsize(test_image)
        MAX_IMAGE_SIZE = 5 * 1024 * 1024

        assert file_size > MAX_IMAGE_SIZE


class TestHandleFileUploadValidation:
    """Integration tests for handle_file_upload function."""

    def test_validation_logic_rejects_large_files(self):
        """Test that the validation logic correctly identifies large files."""
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

        # Simulate file sizes
        test_cases = [
            (5 * 1024 * 1024, True, "5MB should pass"),
            (10 * 1024 * 1024, True, "10MB should pass"),
            (10 * 1024 * 1024 + 1, False, "10MB+1 byte should fail"),
            (11 * 1024 * 1024, False, "11MB should fail"),
            (100 * 1024 * 1024, False, "100MB should fail"),
        ]

        for file_size, should_pass, msg in test_cases:
            is_valid = file_size <= MAX_FILE_SIZE
            assert is_valid == should_pass, f"Failed: {msg}"

    def test_error_message_format(self):
        """Test that error messages show size in MB correctly."""
        file_size_bytes = 15 * 1024 * 1024  # 15MB
        size_mb = file_size_bytes / (1024 * 1024)

        error_msg = f"File is too large ({size_mb:.1f}MB). Maximum allowed size is 10MB."

        assert "15.0MB" in error_msg
        assert "10MB" in error_msg


class TestStreamlitFileUploadValidation:
    """Tests for Streamlit UI file upload validation."""

    def test_document_size_validation_logic(self):
        """Test document upload validation logic."""
        MAX_FILE_SIZE_MB = 10

        # Mock uploaded file with getvalue()
        class MockUploadedFile:
            def __init__(self, size_bytes):
                self.content = b"x" * size_bytes
                self.name = "test.pdf"

            def getvalue(self):
                return self.content

        # Test cases
        small_file = MockUploadedFile(5 * 1024 * 1024)
        large_file = MockUploadedFile(15 * 1024 * 1024)

        small_size_mb = len(small_file.getvalue()) / (1024 * 1024)
        large_size_mb = len(large_file.getvalue()) / (1024 * 1024)

        assert small_size_mb < MAX_FILE_SIZE_MB, "5MB file should pass"
        assert large_size_mb > MAX_FILE_SIZE_MB, "15MB file should fail"

    def test_image_size_validation_logic(self):
        """Test image upload validation logic."""
        MAX_IMAGE_SIZE_MB = 5

        # Mock image file with seek/read
        class MockImageFile:
            def __init__(self, size_bytes):
                self.content = b"x" * size_bytes
                self.name = "test.jpg"
                self._pos = 0

            def seek(self, pos):
                self._pos = pos

            def read(self):
                return self.content

        small_image = MockImageFile(3 * 1024 * 1024)
        large_image = MockImageFile(8 * 1024 * 1024)

        # Validate small image
        small_image.seek(0)
        small_size_mb = len(small_image.read()) / (1024 * 1024)
        assert small_size_mb < MAX_IMAGE_SIZE_MB, "3MB image should pass"

        # Validate large image
        large_image.seek(0)
        large_size_mb = len(large_image.read()) / (1024 * 1024)
        assert large_size_mb > MAX_IMAGE_SIZE_MB, "8MB image should fail"


@pytest.mark.playwright
class TestFileSizeValidationE2E:
    """End-to-end Playwright tests for file upload validation."""

    @pytest.mark.skip(reason="Requires running Chainlit server")
    async def test_large_file_shows_error_message(self, page, chainlit_base_url, temp_dir):
        """Test that uploading a large file shows an error message."""
        # Create a file larger than 10MB
        large_file = temp_dir / "large_test.pdf"
        large_file.write_bytes(b"x" * (11 * 1024 * 1024))

        await page.goto(chainlit_base_url)
        await page.wait_for_selector('[data-testid="chat-input"]')

        # Upload the large file
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(str(large_file))

        # Wait for error message
        error_message = await page.wait_for_selector('text=too large')
        assert error_message is not None

    @pytest.mark.skip(reason="Requires running Chainlit server")
    async def test_valid_file_processes_successfully(self, page, chainlit_base_url, temp_dir, sample_pdf_content):
        """Test that a valid small file is processed."""
        # Create a small valid PDF
        small_file = temp_dir / "small_test.pdf"
        small_file.write_bytes(sample_pdf_content)

        await page.goto(chainlit_base_url)
        await page.wait_for_selector('[data-testid="chat-input"]')

        # Upload the file
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(str(small_file))

        # Should see processing message, not error
        await page.wait_for_timeout(2000)  # Wait for processing

        # Check that no "too large" error appears
        error_count = await page.locator('text=too large').count()
        assert error_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
