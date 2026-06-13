import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
from agentic_rag import config

logger = logging.getLogger(__name__)

# List of common section titles in scientific papers
COMMON_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "method", "proposed method", "approach",
    "preliminaries", "definitions", "experiments", "experimental setup", "evaluation", "results",
    "discussion", "conclusion", "references", "future work", "limitations"
]

IMPORTANT_BLOCKS = [
    "definition", "example", "theorem", "proposition",
    "lemma", "remark", "corollary", "notation"
]

class PDFProcessor:
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_blocks_with_styles(self, pdf_path: Path) -> Tuple[List[Dict[str, Any]], float]:
        """
        Extracts text blocks along with their font sizes and bold status.
        Also determines the most common font size (assumed to be body text size).
        """
        doc = fitz.open(pdf_path)
        blocks_data = []
        font_sizes = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # get page text as dictionary of blocks, lines, spans
            page_dict = page.get_text("dict")
            
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 is text block
                    continue
                    
                block_text = ""
                block_sizes = []
                block_flags = []
                
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        line_text += span_text
                        block_sizes.append(span.get("size", 10.0))
                        block_flags.append(span.get("flags", 0))
                    block_text += line_text + "\n"
                
                block_text = block_text.strip()
                if not block_text:
                    continue
                
                # Determine average size and if bold
                avg_size = sum(block_sizes) / len(block_sizes) if block_sizes else 10.0
                is_bold = any(flag & 4 for flag in block_flags)  # flag 4 is bold in fitz
                
                blocks_data.append({
                    "text": block_text,
                    "size": avg_size,
                    "is_bold": is_bold,
                    "page": page_num + 1
                })
                font_sizes.extend(block_sizes)
                
        doc.close()
        
        # Most common font size is the body text size
        body_font_size = 10.0
        if font_sizes:
            size_counts = Counter([round(s, 1) for s in font_sizes])
            body_font_size = size_counts.most_common(1)[0][0]
            
        return blocks_data, body_font_size

    def is_section_heading(self, text: str, size: float, is_bold: bool, body_font_size: float) -> bool:
        """Determines if a block is a section heading."""
        clean_text = text.strip().lower()
        if not clean_text:
            return False
            
        # Ignore references block if it's too long (headings are short)
        if len(clean_text) > 100:
            return False
            
        # Regular expressions for section numbers (e.g., "1. Introduction", "3.2 Methodology")
        numbered_sec_pattern = r'^(?:[0-9]+\.)+[0-9]*\s+[a-zA-Z\s]+'
        roman_sec_pattern = r'^[i|v|x|l|c]+\.?\s+[a-zA-Z\s]+'
        
        matches_pattern = bool(
            re.match(numbered_sec_pattern, clean_text) or
            re.match(roman_sec_pattern, clean_text) or
            any(clean_text.startswith(sec) or clean_text == sec for sec in COMMON_SECTIONS)
        )
        
        # Styles check: larger than body font or bold, plus matches regex or is short and prominent
        if matches_pattern:
            return True
            
        # If it's bold and significantly larger than body text, and short
        if len(clean_text) < 60 and (size > body_font_size + 1.5 or (is_bold and size >= body_font_size)):
            # Ensure it doesn't look like page numbers or small figure captions
            if not re.match(r'^(fig\.|figure|table|page|\d+$)', clean_text):
                return True
                
        return False

    def parse_pdf_to_sections(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parses the PDF and groups text by sections."""
        try:
            blocks, body_font_size = self.extract_blocks_with_styles(pdf_path)
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return []

        sections = []
        current_section = {
            "title": "Abstract / Preamble",
            "text": "",
            "page_start": 1,
            "page_end": 1
        }
        
        for block in blocks:
            text = block["text"]
            size = block["size"]
            is_bold = block["is_bold"]
            page = block["page"]
            
            if self.is_section_heading(text, size, is_bold, body_font_size):
                # Save previous section if it has text
                if current_section["text"].strip():
                    current_section["page_end"] = page
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": text.replace("\n", " ").strip(),
                    "text": "",
                    "page_start": page,
                    "page_end": page
                }
            else:
                # Add text to current section
                current_section["text"] += text + "\n"
                current_section["page_end"] = page

        # Append final section
        if current_section["text"].strip():
            sections.append(current_section)
            
        return sections

    def detect_block_type(self, text: str) -> str:
        """Detect semantic math/science blocks such as Definition, Theorem, Example."""
        clean_text = text.strip().lower()
        if not clean_text:
            return "paragraph"

        for block_type in IMPORTANT_BLOCKS:
            pattern = rf"^{block_type}\b(?:\s+[0-9ivxlc]+(?:\.[0-9]+)*\.?)?(?:\s|\:|\.|\(|$)"
            if re.match(pattern, clean_text):
                return block_type

        return "paragraph"

    def split_section_into_blocks(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a section into semantic blocks before regular size-based chunking."""
        text = section["text"]
        lines = text.splitlines()
        blocks = []
        current_lines = []
        current_type = "paragraph"

        def flush_current() -> None:
            nonlocal current_lines, current_type
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append({
                    "block_type": current_type,
                    "text": block_text
                })
            current_lines = []
            current_type = "paragraph"

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append("")
                continue

            detected_type = self.detect_block_type(stripped)
            if detected_type != "paragraph" and current_lines:
                flush_current()

            if detected_type != "paragraph":
                current_type = detected_type

            current_lines.append(stripped)

        flush_current()

        if not blocks and text.strip():
            blocks.append({
                "block_type": "paragraph",
                "text": text.strip()
            })

        return blocks

    def chunk_text_block(
        self,
        text: str,
        section: Dict[str, Any],
        paper_metadata: Dict[str, Any],
        block_type: str,
        block_index: int
    ) -> List[Dict[str, Any]]:
        """Splits a large section into overlapping chunks while maintaining context."""
        title = section["title"]
        page_start = section["page_start"]
        page_end = section["page_end"]
        
        # Split text into sentences/paragraphs roughly
        paragraphs = text.split("\n\n")
        chunks = []
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_len = len(para)
            
            # If paragraph itself is larger than chunk size, split by sentences
            if para_len > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sentence_len = len(sentence)
                    if current_length + sentence_len > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append(chunk_text)
                        
                        # Apply overlap by keeping last few elements
                        overlap_chars = 0
                        overlap_chunk = []
                        for prev_s in reversed(current_chunk):
                            if overlap_chars + len(prev_s) < self.chunk_overlap:
                                overlap_chunk.insert(0, prev_s)
                                overlap_chars += len(prev_s)
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_length = overlap_chars
                        
                    current_chunk.append(sentence)
                    current_length += sentence_len
            else:
                if current_length + para_len > self.chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Apply overlap
                    overlap_chars = 0
                    overlap_chunk = []
                    for prev_p in reversed(current_chunk):
                        if overlap_chars + len(prev_p) < self.chunk_overlap:
                            overlap_chunk.insert(0, prev_p)
                            overlap_chars += len(prev_p)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_chars
                    
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Format output chunks with rich metadata
        final_chunks = []
        for i, chunk_text in enumerate(chunks):
            final_chunks.append({
                "arxiv_id": paper_metadata["id"],
                "title": paper_metadata["title"],
                "authors": ", ".join(paper_metadata.get("authors", [])),
                "section": title,
                "block_type": block_type,
                "block_index": block_index,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": i,
                "text": chunk_text
            })
            
        return final_chunks

    def chunk_section(self, section: Dict[str, Any], paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits a section into semantic blocks, then chunks each block by size."""
        final_chunks = []
        semantic_blocks = self.split_section_into_blocks(section)

        for block_index, block in enumerate(semantic_blocks):
            block_chunks = self.chunk_text_block(
                block["text"],
                section,
                paper_metadata,
                block["block_type"],
                block_index
            )
            for chunk in block_chunks:
                chunk["chunk_index"] = len(final_chunks)
                final_chunks.append(chunk)

        return final_chunks

    def process_paper(self, pdf_path: Path, paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses and chunks a single paper."""
        logger.info(f"Processing paper {paper_metadata['id']} from {pdf_path}")
        sections = self.parse_pdf_to_sections(pdf_path)
        
        all_chunks = []
        for section in sections:
            chunks = self.chunk_section(section, paper_metadata)
            all_chunks.extend(chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks for paper {paper_metadata['id']}")
        return all_chunks
