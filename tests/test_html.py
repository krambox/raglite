"""Test Markdown conversion."""

from pathlib import Path

from raglite._html import document_to_html


def test_pdf_with_missing_font_sizes() -> None:
    """Test conversion of a PDF with missing font sizes."""
    # Convert a PDF whose parsed font sizes are all equal to 1.
    doc_path = Path(__file__).parent / "2023_12_11_Duesseldorfer_Tabelle_-2024.pdf"
    doc = document_to_html(doc_path)
    # Verify that we can reconstruct the font sizes and heading levels regardless of the missing
    # font size data.
    expected_heading = '<html lang="en">\n\n<h2>DÜSSELDORFER TABELLE 1</h2>\n<h2>A. Kindesunterhalt</h2>'
    assert doc.startswith(expected_heading)
