import re
import os

def clean_wikitext_specific(input_file, output_file, chunk_size=10*1024*1024):
    """
    Clean WikiText with specific attention to its format
    """
    
    print(f"Processing {input_file} for WikiText-specific cleaning...")
    
    file_size = os.path.getsize(input_file)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")
    
    total_original = 0
    total_cleaned = 0
    chunk_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        buffer = ""
        
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                if buffer:
                    cleaned_chunk = clean_wikitext_chunk(buffer)
                    outfile.write(cleaned_chunk)
                    total_cleaned += len(cleaned_chunk)
                break
            
            buffer += chunk
            total_original += len(chunk)
            
            # Split at article boundaries (= Title =)
            articles = buffer.split('\n= ')
            
            # Process all complete articles except the last (might be incomplete)
            for i, article in enumerate(articles[:-1]):
                if i == 0:
                    # First chunk might not start with =, handle normally
                    processed = clean_wikitext_chunk(article)
                else:
                    # Add back the = that was removed by split
                    processed = clean_wikitext_chunk('= ' + article)
                
                outfile.write(processed)
                if not processed.endswith('\n'):
                    outfile.write('\n')
                total_cleaned += len(processed)
            
            # Keep the last incomplete article in buffer
            buffer = articles[-1] if articles else ""
            
            chunk_count += 1
            processed_mb = total_original / 1024 / 1024
            print(f"Processed chunk {chunk_count}, {processed_mb:.1f}MB / {file_size/1024/1024:.1f}MB ({processed_mb/file_size*1024*1024*100:.1f}%)")
    
    reduction = (total_original - total_cleaned) / total_original * 100
    print(f"\nCleaning complete!")
    print(f"Original: {total_original:,} characters ({total_original/1024/1024:.1f} MB)")
    print(f"Cleaned: {total_cleaned:,} characters ({total_cleaned/1024/1024:.1f} MB)")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Estimated tokens: {total_cleaned//4:,}")
    
    return total_cleaned

def clean_wikitext_chunk(text):
    """
    Clean WikiText chunk with specific patterns
    """
    
    # Fix @-@ symbols (common in WikiText)
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'@\s*,\s*@', ',', text)  # Handle @,@
    text = re.sub(r'@\s*\.\s*@', '.', text)  # Handle @.@
    
    # Remove <unk> tokens (unknown words)
    text = re.sub(r'<unk>', '[UNKNOWN]', text)  # Replace with placeholder or remove entirely
    # Or to remove completely: text = re.sub(r'<unk>\s*', '', text)
    
    # Clean reference artifacts
    text = re.sub(r'\[\s*note\s*\d+\s*\]', '', text)
    text = re.sub(r'\[\s*citation needed\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)  # Remove [1], [2], etc.
    
    # Clean section headers - keep them but normalize
    text = re.sub(r'^=\s*([^=]+?)\s*=$', r'= \1 =', text, flags=re.MULTILINE)
    text = re.sub(r'^==\s*([^=]+?)\s*==$', r'== \1 ==', text, flags=re.MULTILINE)
    text = re.sub(r'^===\s*([^=]+?)\s*===$', r'=== \1 ===', text, flags=re.MULTILINE)
    
    # Remove very short sections that are likely artifacts
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip very short lines that look like artifacts
        if (len(line) > 0 and len(line) < 10 and 
            not line.startswith('=') and  # Keep section headers
            not re.match(r'^[A-Z][a-z]+$', line) and  # Skip single words
            not re.match(r'^\d+$', line)):  # Skip lone numbers
            continue
            
        # Clean up common artifacts
        if any(artifact in line.lower() for artifact in [
            'thumb|', 'left|', 'right|', 'center|', 'px', 'file:', 'image:'
        ]):
            continue
            
        cleaned_lines.append(line)
    
    # Rejoin and clean whitespace
    text = '\n'.join(cleaned_lines)
    
    # Remove excessive whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)
    
    # Remove paragraphs that are too short (likely artifacts)
    paragraphs = text.split('\n\n')
    good_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        # Keep if it's a section header or substantial content
        if (para.startswith('=') or len(para) > 100 or 
            (len(para) > 20 and para.count('.') > 0)):
            good_paragraphs.append(para)
    
    text = '\n\n'.join(good_paragraphs)
    
    return text

def analyze_sample(filename, num_lines=50):
    """
    Show analysis of the text before and after cleaning
    """
    print(f"\n--- Analysis of {filename} ---")
    
    patterns_to_check = [
        (r'@-@', 'Hyphen artifacts (@-@)'),
        (r'<unk>', 'Unknown tokens (<unk>)'),
        (r'\[\s*note\s*\d+\s*\]', 'Note references'),
        (r'\[\s*\d+\s*\]', 'Number references'),
        (r'^\s*=+\s*.*?\s*=+\s*$', 'Section headers'),
    ]
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"File size: {len(content):,} characters")
        print(f"Estimated tokens: {len(content)//4:,}")
        
        print("\nPattern analysis:")
        for pattern, description in patterns_to_check:
            matches = len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
            print(f"  {description}: {matches:,} occurrences")
        
        print(f"\nFirst {num_lines} lines:")
        lines = content.split('\n')
        for i, line in enumerate(lines[:num_lines]):
            if line.strip():
                print(f"{i+1:3d}: {line[:100]}{'...' if len(line) > 100 else ''}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Configuration
    input_file = "/home/akshat/GPT_from_scratch/text_data/wikitext_full.txt"  # Change to your actual filename
    output_file = "wikitext_cleaned.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Available .txt files:")
        for f in os.listdir('.'):
            if f.endswith('.txt'):
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"  {f} ({size_mb:.1f} MB)")
        exit(1)
    
    # Analyze before cleaning
    print("=== BEFORE CLEANING ===")
    analyze_sample(input_file)
    
    # Clean the file
    print("\n=== CLEANING ===")
    cleaned_chars = clean_wikitext_specific(input_file, output_file)
    
    # Analyze after cleaning
    print("\n=== AFTER CLEANING ===")
    analyze_sample(output_file, 20)
    
    # Recommendations
    estimated_tokens = cleaned_chars // 4
    optimal_params = estimated_tokens // 20  # Chinchilla scaling
    
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Optimal model size (Chinchilla): {optimal_params:,} parameters")
    
    if optimal_params > 50000000:  # > 50M params
        print(f"Model size range: {optimal_params//2:,} - {optimal_params*2:,} parameters")
        print("This is excellent data for a substantial model!")
        print("Recommended architecture:")
        print(f"  - Parameters: {optimal_params:,}")
        print("  - Hidden dim: 1024-2048")
        print("  - Layers: 12-24")
        print("  - Attention heads: 16-32")
    elif optimal_params > 1000000:  # > 1M params
        print(f"Model size range: {optimal_params//2:,} - {optimal_params*2:,} parameters")
        print("Good data for a small-medium model!")
        print("Recommended architecture:")
        print(f"  - Parameters: {optimal_params:,}")
        print("  - Hidden dim: 512-1024") 
        print("  - Layers: 6-12")
        print("  - Attention heads: 8-16")
    else:
        print("Still a relatively small dataset.")
        print("Consider combining with other sources for better results.")