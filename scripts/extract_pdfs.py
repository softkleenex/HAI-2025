import os
import glob
from pypdf import PdfReader

def extract_text_from_pdfs(directories):
    for input_dir in directories:
        print(f"--- Processing directory: {input_dir} ---")
        
        # Create output directory for text files
        output_dir = os.path.join(input_dir, "extracted_text")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            continue

        for pdf_path in pdf_files:
            try:
                filename = os.path.basename(pdf_path)
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                output_path = os.path.join(output_dir, txt_filename)
                
                print(f"Extracting: {filename} -> {txt_filename}")
                
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                
                if not text.strip():
                    print(f"Warning: No text extracted from {filename} (might be image-based).")
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    target_dirs = ["info", "talks"]
    extract_text_from_pdfs(target_dirs)