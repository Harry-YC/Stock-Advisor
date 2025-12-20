# Travel Planner Test Suite

Comprehensive test suite for the Travel Planner application using pytest and Playwright.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-playwright playwright

# Install Playwright browsers (first time only)
playwright install

# Run all unit tests
python3 tests/run_tests.py

# Run with verbose output
python3 -m pytest tests/ -v
```

## Test Structure

```
tests/
├── conftest.py                       # Shared fixtures and configuration
├── run_tests.py                      # Test runner script
├── test_file_upload_validation.py    # File size limit tests
├── test_excel_traveler_extraction.py # Excel export regex tests
├── test_security.py                  # Security feature tests
├── test_places_enrichment.py         # Google Places integration tests
├── test_export_e2e.py                # Excel/Word export E2E tests
└── test_travel_planner.py            # Core functionality tests
```

## Test Categories

### 1. File Upload Validation (`test_file_upload_validation.py`)

Tests file size limits to prevent DoS and memory issues.

| Test | Description |
|------|-------------|
| `test_file_under_limit_accepted` | Files under 10MB pass |
| `test_file_over_limit_rejected` | Files over 10MB rejected |
| `test_image_under_5mb_accepted` | Images under 5MB pass |
| `test_image_over_5mb_rejected` | Images over 5MB rejected |

**Limits:**
- Documents: 10MB max
- Images: 5MB max per image

### 2. Excel Export (`test_excel_traveler_extraction.py`)

Tests traveler count extraction from various formats.

| Format | Example | Expected |
|--------|---------|----------|
| Group format | `Group (4+)` | 4 |
| People format | `4 people` | 4 |
| Adults format | `2 adults` | 2 |
| Chinese format | `四人` | 4 |

### 3. Security Tests (`test_security.py`)

Tests for security hardening features.

#### Exception Message Sanitization
- Error messages shown to users are generic
- No stack traces, file paths, or internal details exposed
- Full exception details logged server-side with `exc_info=True`

#### Prompt Injection Protection
- `_sanitize_for_prompt()` function tests
- Blocks injection patterns like "ignore previous instructions"
- Removes special characters that could manipulate prompts
- Truncates input to 200 characters max

#### Thread-Safe Cache
- GooglePlacesClient cache uses `RLock`
- Concurrent access tests with multiple threads
- No race conditions in cache operations

### 4. Google Places Enrichment (`test_places_enrichment.py`)

Tests the Places API integration and trust scoring.

#### Trust Score Algorithm

| Score | Emoji | Criteria |
|-------|-------|----------|
| HIGH | ✅ | 200+ reviews, 4.0+ rating |
| MEDIUM | ⚠️ | 50-200 reviews, decent rating |
| LOW | ❓ | <50 reviews or low rating |
| SUSPICIOUS | 🚩 | Perfect 5.0 with <100 reviews |
| NOT_FOUND | ❌ | Place not on Google Maps |

#### Place Name Extraction
- Extracts from bold markdown (`**Place Name**`)
- Extracts from recommendation phrases ("I recommend X")
- Deduplicates case-insensitively
- Limits to 10 places max

## Running Tests

### Unit Tests (No Server Required)

```bash
# All unit tests
python3 tests/run_tests.py

# Specific test file
python3 -m pytest tests/test_security.py -v

# Specific test
python3 -m pytest tests/test_security.py::TestPromptInjectionProtection -v

# Pattern matching
python3 -m pytest tests/ -k "sanitize" -v
```

### E2E Tests (Requires Running Server)

```bash
# Start the Chainlit server first
chainlit run app_tp2.py &

# Run E2E tests
python3 tests/run_tests.py --e2e

# Or with pytest directly
python3 -m pytest tests/ -m "playwright" -v
```

### With Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
python3 tests/run_tests.py --coverage

# View HTML report
open coverage_html/index.html
```

## Test Fixtures

### From `conftest.py`

| Fixture | Description |
|---------|-------------|
| `temp_dir` | Temporary directory for test files |
| `sample_pdf_content` | Minimal valid PDF bytes |
| `large_file_content` | 11MB file for limit testing |
| `sample_trip_config` | Sample trip configuration dict |
| `sample_expert_responses` | Sample expert response dict |
| `mock_places_response` | Mock Google Places API response |
| `chainlit_helper` | Helper class for Chainlit UI testing |

### Using Fixtures

```python
def test_example(temp_dir, sample_trip_config):
    # temp_dir is a Path to a temporary directory
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")

    # sample_trip_config is a dict with trip details
    assert sample_trip_config["destination"] == "Barcelona, Spain"
```

## Writing New Tests

### Unit Test Template

```python
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestNewFeature:
    """Tests for the new feature."""

    def test_basic_functionality(self):
        """Test that basic functionality works."""
        result = some_function()
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            some_function(invalid_input)
```

### Playwright E2E Test Template

```python
import pytest

@pytest.mark.playwright
class TestNewFeatureE2E:
    """E2E tests for new feature."""

    @pytest.mark.skip(reason="Requires running server")
    async def test_ui_interaction(self, page, chainlit_base_url):
        """Test UI interaction."""
        await page.goto(chainlit_base_url)
        await page.wait_for_selector('[data-testid="chat-input"]')

        # Interact with the UI
        input_field = page.locator('[data-testid="chat-input"]')
        await input_field.fill("Test message")
        await input_field.press("Enter")

        # Verify result
        response = await page.wait_for_selector('.message-content')
        assert response is not None
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          playwright install

      - name: Run tests
        run: python3 -m pytest tests/ -v -m "not playwright"
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd "/Users/nelsonliu/Travel Planner"
python3 -m pytest tests/ -v
```

**Playwright Not Installed**
```bash
pip install pytest-playwright playwright
playwright install
```

**Tests Timeout**
```bash
# Increase timeout for slow tests
python3 -m pytest tests/ -v --timeout=120
```

## Test Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `services/` | 80% | ~70% |
| `integrations/` | 70% | ~60% |
| `core/` | 60% | ~50% |

## Contributing

1. Write tests for any new features
2. Ensure all tests pass before submitting PR
3. Maintain or improve test coverage
4. Use descriptive test names and docstrings
