#!/usr/bin/env python3
"""
Script to test Pillow installation and PDF processing
"""
import sys
import importlib.util


def check_pillow():
    """Check if Pillow is properly installed and accessible"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check Pillow
    try:
        from PIL import Image, __version__
        print(f"Pillow is installed (version {__version__})")
        print(f"Pillow package location: {Image.__file__}")
        return True
    except ImportError as e:
        print(f"Error importing PIL: {e}")
        return False


def check_pdf_libraries():
    """Check if PDF processing libraries are installed"""
    pdf_libs = [
        "pdf2image",
        "PyPDF2",
        "pdfminer",
        "pdfminer.six",
        "pypdf"
    ]

    for lib in pdf_libs:
        spec = importlib.util.find_spec(lib)
        if spec is not None:
            try:
                module = importlib.import_module(lib)
                version = getattr(module, "__version__", "unknown")
                print(f"{lib} is installed (version {version})")
                print(f"{lib} location: {module.__file__}")
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Found {lib} but error importing: {e}")
        else:
            print(f"{lib} is NOT installed")


def test_pdf_processing():
    """Test if PDF processing works"""
    try:
        from pdf2image import convert_from_path
        print("PDF2Image is working correctly")
    except Exception as e:
        print(f"Error with pdf2image: {e}")

    try:
        import PyPDF2
        print("PyPDF2 is working correctly")
    except Exception as e:
        print(f"Error with PyPDF2: {e}")

    try:
        from pdfminer.high_level import extract_text
        print("PDFMiner is working correctly")
    except Exception as e:
        print(f"Error with pdfminer: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Pillow and PDF Library Test")
    print("=" * 50)

    pillow_ok = check_pillow()

    print("\nChecking PDF libraries:")
    check_pdf_libraries()

    if pillow_ok:
        print("\nTesting PDF processing:")
        test_pdf_processing()

    print("\nTest complete")
