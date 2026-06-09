import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from agentic_rag import config

logger = logging.getLogger(__name__)

# XML namespaces used by arXiv API
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
    'arxiv': 'http://arxiv.org/schemas/atom'
}

def clean_arxiv_id(url_or_id: str) -> str:
    """Extracts a clean, versionless or versioned arXiv ID from a URL or raw string."""
    # Examples:
    # "http://arxiv.org/abs/2103.00020v1" -> "2103.00020"
    # "arXiv:2103.00020" -> "2103.00020"
    # "2103.00020v2" -> "2103.00020"
    match = re.search(r'(?:arxiv:)?([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?', url_or_id.lower())
    if match:
        return match.group(1)
    
    # Old format e.g. hep-th/9711200
    match_old = re.search(r'([a-z\-]+(?:\.[a-z]+)?/[0-9]{7})(?:v[0-9]+)?', url_or_id.lower())
    if match_old:
        return match_old.group(1)
        
    return url_or_id

class ArxivSearchAgent:
    def __init__(self):
        self.pdf_dir = config.PDF_DIR

    def search(self, query: str, max_results: int = 10, start: int = 0) -> List[Dict[str, Any]]:
        """
        Searches arXiv API for papers matching the query.
        Returns a list of dictionaries containing paper metadata.
        """
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start={start}&max_results={max_results}"
        
        logger.info(f"Querying arXiv API: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AgenticRagArxiv/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            
            return self._parse_feed(xml_data)
        except Exception as e:
            logger.error(f"Error querying arXiv: {e}")
            return []

    def fetch_by_ids(self, arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches metadata for specific arXiv IDs."""
        cleaned_ids = [clean_arxiv_id(aid) for aid in arxiv_ids]
        ids_str = ",".join(cleaned_ids)
        url = f"http://export.arxiv.org/api/query?id_list={ids_str}"
        
        logger.info(f"Querying arXiv API for IDs: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AgenticRagArxiv/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            
            return self._parse_feed(xml_data)
        except Exception as e:
            logger.error(f"Error fetching arXiv IDs {arxiv_ids}: {e}")
            return []

    def _parse_feed(self, xml_data: bytes) -> List[Dict[str, Any]]:
        """Parses the XML Atom feed returned by the arXiv API."""
        root = ET.fromstring(xml_data)
        results = []
        
        for entry in root.findall('atom:entry', NAMESPACES):
            # ID
            raw_id = entry.find('atom:id', NAMESPACES).text or ""
            arxiv_id = clean_arxiv_id(raw_id)
            
            # Title (clean whitespace and newlines)
            title = entry.find('atom:title', NAMESPACES).text or ""
            title = re.sub(r'\s+', ' ', title).strip()
            
            # Summary (Abstract)
            summary = entry.find('atom:summary', NAMESPACES).text or ""
            summary = re.sub(r'\s+', ' ', summary).strip()
            
            # Published Date
            published = entry.find('atom:published', NAMESPACES).text or ""
            
            # Authors
            authors = []
            for author_node in entry.findall('atom:author', NAMESPACES):
                name_node = author_node.find('atom:name', NAMESPACES)
                if name_node is not None:
                    authors.append(name_node.text.strip())
            
            # PDF Link
            pdf_link = ""
            for link_node in entry.findall('atom:link', NAMESPACES):
                title_attr = link_node.attrib.get('title', '')
                type_attr = link_node.attrib.get('type', '')
                href_attr = link_node.attrib.get('href', '')
                if title_attr == 'pdf' or 'pdf' in type_attr or 'pdf' in href_attr:
                    pdf_link = href_attr
                    break
            
            # If pdf link wasn't explicitly found, construct standard one
            if not pdf_link and arxiv_id:
                pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
            # Categories
            primary_cat_node = entry.find('arxiv:primary_category', NAMESPACES)
            primary_cat = primary_cat_node.attrib.get('term', '') if primary_cat_node is not None else ""
            
            categories = [cat.attrib.get('term', '') for cat in entry.findall('atom:category', NAMESPACES)]

            results.append({
                "id": arxiv_id,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors,
                "pdf_link": pdf_link,
                "primary_category": primary_cat,
                "categories": categories
            })
            
        return results

    def download_pdf(self, arxiv_id: str, pdf_link: Optional[str] = None) -> Optional[Path]:
        """Downloads the PDF for a given arXiv ID and saves it in the PDF directory."""
        arxiv_id = clean_arxiv_id(arxiv_id)
        # Avoid slashes in filenames (e.g. old IDs like hep-th/9711200)
        safe_filename = arxiv_id.replace("/", "_")
        dest_path = self.pdf_dir / f"{safe_filename}.pdf"
        
        if dest_path.exists():
            logger.info(f"PDF for {arxiv_id} already exists at {dest_path}")
            return dest_path
            
        if not pdf_link:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
        logger.info(f"Downloading PDF from {pdf_link} to {dest_path}")
        try:
            req = urllib.request.Request(pdf_link, headers={'User-Agent': 'AgenticRagArxiv/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                # Check headers to see if we got PDF or HTML (sometimes arXiv blocks or returns HTML error)
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type:
                    logger.error(f"ArXiv returned HTML instead of PDF for {arxiv_id}. Possibly rate-limited or page not found.")
                    return None
                    
                with open(dest_path, 'wb') as f:
                    f.write(response.read())
                    
            logger.info(f"Successfully downloaded {arxiv_id}.pdf")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # delete partial file
            return None
