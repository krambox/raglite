"""Convert any document to Markdown."""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def document_to_html(doc_path: Path) -> str:
    """Convert any document to GitHub Flavored Markdown."""
     # Parse the PDF with pdftext and convert it to Markdown.
    pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(doc_path)
    doc=result.document.export_to_html(html_head='')
    #remove '<!DOCTYPE html>\n'
    doc=doc[16:]
    return doc
