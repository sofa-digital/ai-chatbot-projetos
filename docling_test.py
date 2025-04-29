from docling.document_converter import DocumentConverter

source = "onboarding.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_text())  # output: "## Docling Technical Report[...]"