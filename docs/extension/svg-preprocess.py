"""
SVG Pre-processor for Draw.io exported SVGs
Fixes text rendering and background issues for PDF conversion
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from sphinx.util import logging
from sphinx.transforms import SphinxTransform
from docutils import nodes
import copy

logger = logging.getLogger(__name__)

# XML namespaces
SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}
XLINK_NS = {'xlink': 'http://www.w3.org/1999/xlink'}

class SVGPreprocessor:
    """Pre-process Draw.io SVG files for better PDF conversion"""
    
    def __init__(self):
        self.processed_cache = {}
    
    def process_svg(self, svg_path, for_pdf=False):
        """Process an SVG file and return the processed content"""
        
        # Check cache
        cache_key = f"{svg_path}_{for_pdf}"
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        try:
            # Parse SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            if for_pdf:
                # Fix background
                self._fix_background(root)
                
                # Convert foreignObject to text
                self._convert_foreign_objects(root)
                
                # Fix text rendering
                self._fix_text_elements(root)
            
            # Convert to string
            ET.register_namespace('', 'http://www.w3.org/2000/svg')
            ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
            
            result = ET.tostring(root, encoding='unicode', method='xml')
            
            # Cache result
            self.processed_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to process SVG {svg_path}: {str(e)}")
            # Return original content if processing fails
            with open(svg_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _fix_background(self, root):
        """Convert CSS background to SVG rect element"""
        
        # Check for background in style attribute
        style_attr = root.get('style', '')
        bg_match = re.search(r'background(?:-color)?:\s*([^;]+)', style_attr)
        
        if bg_match:
            bg_color = bg_match.group(1).strip()
            
            # Handle light-dark() function
            if 'light-dark' in bg_color:
                # Extract the dark mode color (second parameter)
                match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if match:
                    bg_color = f"rgb({match.group(1)}, {match.group(2)}, {match.group(3)})"
                else:
                    bg_color = '#5E5B61'  # Default dark background
            elif 'rgb(' in bg_color:
                # Already in RGB format
                pass
            else:
                # Try to use color as-is
                pass
            
            # Get SVG dimensions
            width = root.get('width', '100%')
            height = root.get('height', '100%')
            viewbox = root.get('viewBox', '')
            
            # Parse viewBox if present
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    x, y, vb_width, vb_height = parts
                else:
                    x, y = '0', '0'
                    vb_width, vb_height = width, height
            else:
                x, y = '0', '0'
                vb_width, vb_height = width, height
            
            # Create background rect as first element
            rect = ET.Element('rect')
            rect.set('x', x)
            rect.set('y', y)
            rect.set('width', vb_width if vb_width != '100%' else width)
            rect.set('height', vb_height if vb_height != '100%' else height)
            rect.set('fill', bg_color)
            
            # Insert as first child
            root.insert(0, rect)
            
            # Remove background from style
            new_style = re.sub(r'background(?:-color)?:[^;]+;?\s*', '', style_attr)
            if new_style.strip():
                root.set('style', new_style)
            elif 'style' in root.attrib:
                del root.attrib['style']
    
    def _convert_foreign_objects(self, root):
        """Convert foreignObject elements to native SVG text"""
        
        # Find all foreignObject elements
        foreign_objects = root.findall('.//foreignObject', {'': 'http://www.w3.org/2000/svg'})
        
        for foreign_obj in foreign_objects:
            # Get position and size
            x = float(foreign_obj.get('x', '0'))
            y = float(foreign_obj.get('y', '0'))
            width = float(foreign_obj.get('width', '100'))
            height = float(foreign_obj.get('height', '20'))
            
            # Extract text from HTML content
            text_content = self._extract_text_from_foreign(foreign_obj)
            
            if text_content:
                # Extract style information
                style_info = self._extract_style_from_foreign(foreign_obj)
                
                # Create SVG text element
                text_elem = ET.Element('text')
                text_elem.set('x', str(x + width / 2))  # Center horizontally
                text_elem.set('y', str(y + height / 2))  # Center vertically
                text_elem.set('text-anchor', 'middle')
                text_elem.set('dominant-baseline', 'middle')
                
                # Apply styles
                if style_info['color']:
                    text_elem.set('fill', style_info['color'])
                if style_info['font-family']:
                    text_elem.set('font-family', style_info['font-family'])
                if style_info['font-size']:
                    text_elem.set('font-size', style_info['font-size'])
                
                # Handle multi-line text
                lines = text_content.strip().split('\n')
                if len(lines) == 1:
                    text_elem.text = lines[0]
                else:
                    # Create tspan elements for multiple lines
                    line_height = float(style_info.get('font-size', '12px').replace('px', '')) * 1.2
                    start_y = y + height / 2 - (len(lines) - 1) * line_height / 2
                    
                    for i, line in enumerate(lines):
                        tspan = ET.SubElement(text_elem, 'tspan')
                        tspan.set('x', str(x + width / 2))
                        tspan.set('dy', str(line_height if i > 0 else 0))
                        tspan.text = line
                
                # Replace foreignObject with text element
                parent = self._find_parent(root, foreign_obj)
                if parent is not None:
                    index = list(parent).index(foreign_obj)
                    parent.remove(foreign_obj)
                    parent.insert(index, text_elem)
    
    def _extract_text_from_foreign(self, foreign_obj):
        """Extract text content from foreignObject's HTML"""
        
        # Get all text content recursively
        text_parts = []
        
        def extract_text(elem):
            if elem.text:
                text_parts.append(elem.text)
            for child in elem:
                extract_text(child)
                if child.tail:
                    text_parts.append(child.tail)
        
        extract_text(foreign_obj)
        
        # Join and clean up
        text = ' '.join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle special cases for line breaks
        text = text.replace('word-wrap: normal;', '')
        
        return text
    
    def _extract_style_from_foreign(self, foreign_obj):
        """Extract style information from foreignObject's HTML"""
        
        style_info = {
            'color': '#FFFFFF',  # Default white
            'font-size': '12px',
            'font-family': 'Arial, sans-serif'
        }
        
        # Look for div elements with style
        for elem in foreign_obj.iter():
            style = elem.get('style', '')
            
            # Extract color
            color_match = re.search(r'color:\s*([^;]+)', style)
            if color_match:
                color = color_match.group(1).strip()
                # Handle light-dark function
                if 'light-dark' in color:
                    # Use the light mode color for PDF
                    style_info['color'] = '#FFFFFF'
                elif 'rgb' in color:
                    style_info['color'] = color
                else:
                    style_info['color'] = color
            
            # Extract font-size
            size_match = re.search(r'font-size:\s*([^;]+)', style)
            if size_match:
                style_info['font-size'] = size_match.group(1).strip()
            
            # Extract font-family
            family_match = re.search(r'font-family:\s*([^;]+)', style)
            if family_match:
                style_info['font-family'] = family_match.group(1).strip().strip('"\'')
        
        return style_info
    
    def _fix_text_elements(self, root):
        """Fix existing text elements for better PDF rendering"""
        
        # Find all text elements
        text_elements = root.findall('.//text', {'': 'http://www.w3.org/2000/svg'})
        
        for text_elem in text_elements:
            # Ensure text has proper baseline
            if not text_elem.get('dominant-baseline'):
                text_elem.set('dominant-baseline', 'middle')
            
            # Ensure text color is set
            if not text_elem.get('fill'):
                text_elem.set('fill', '#FFFFFF')
    
    def _find_parent(self, root, element):
        """Find parent of an element in the tree"""
        for parent in root.iter():
            if element in parent:
                return parent
        return None


def process_svg_for_pdf(app, env, docnames):
    """Process SVG files for PDF output during the env-get-updated event"""
    
    # Only process for LaTeX/PDF output
    if app.builder.format != 'latex':
        return
    
    processor = SVGPreprocessor()
    processed_files = {}
    
    # Process all SVG files in the environment
    for docname in docnames:
        doctree = env.get_doctree(docname)
        
        for img_node in doctree.traverse(nodes.image):
            uri = img_node.get('uri', '')
            
            # Only process SVG files
            if uri.endswith('.svg'):
                # Get absolute path
                if uri.startswith('/'):
                    svg_path = Path(app.srcdir) / uri[1:]
                else:
                    # Handle relative paths
                    doc_dir = Path(app.srcdir) / Path(docname).parent
                    svg_path = (doc_dir / uri).resolve()
                
                if svg_path.exists():
                    # Use a deterministic path for processed SVG
                    if uri.startswith('/'):
                        processed_name = uri[1:].replace('/', '_')
                    else:
                        processed_name = f"{docname.replace('/', '_')}_{uri.replace('/', '_')}"
                    
                    processed_file = Path(app.srcdir) / '_processed_svgs' / processed_name
                    
                    # Only process if not already done
                    if str(svg_path) not in processed_files:
                        # Process SVG
                        processed_content = processor.process_svg(svg_path, for_pdf=True)
                        
                        # Save processed SVG
                        processed_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(processed_file, 'w', encoding='utf-8') as f:
                            f.write(processed_content)
                        
                        processed_files[str(svg_path)] = processed_file
                        logger.info(f"Pre-processed SVG for PDF: {svg_path.name} -> {processed_file.name}")
                    else:
                        processed_file = processed_files[str(svg_path)]
                    
                    # Update image URI to the processed file (relative to srcdir)
                    new_uri = str(processed_file.relative_to(app.srcdir))
                    img_node['uri'] = new_uri
                    
                    # Also update in the environment's image dictionary if present
                    if hasattr(env, 'images'):
                        # Find and update the image entry
                        for key in list(env.images.keys()):
                            if uri in key or key.endswith(uri):
                                env.images[key] = env.images.get(key, (set(), None))


def setup(app):
    """Setup the SVG pre-processor extension"""
    
    # Connect to the env-get-updated event to process SVGs
    app.connect('env-get-updated', process_svg_for_pdf)
    
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': '1.0.0'
    }
