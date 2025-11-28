"""
Custom SVG to PDF converter for Draw.io SVGs
Processes SVGs to fix text and background issues before conversion
"""

import os
import re
import shutil
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from sphinx.util import logging
from sphinx.transforms import SphinxTransform
from sphinx.builders.latex import LaTeXBuilder
from docutils import nodes

logger = logging.getLogger(__name__)

class DrawioSVGProcessor:
    """Process Draw.io SVGs to fix text and backgrounds"""
    
    def __init__(self):
        self.processed_cache = {}
    
    def preprocess_svg(self, svg_path):
        """Pre-process Draw.io SVG to fix text and backgrounds"""
        
        try:
            # Parse SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Fix background
            self._fix_background(root)
            
            # Convert foreignObject to text
            self._convert_foreign_objects(root)
            
            # Fix text rendering
            self._fix_text_elements(root)
            
            # Register namespaces
            ET.register_namespace('', 'http://www.w3.org/2000/svg')
            ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
            
            # Return modified tree
            return ET.tostring(root, encoding='unicode', method='xml')
            
        except Exception as e:
            logger.warning(f"Failed to preprocess SVG {svg_path}: {str(e)}")
            with open(svg_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _fix_background(self, root):
        """Convert CSS background to SVG rect element"""
        
        style_attr = root.get('style', '')
        bg_match = re.search(r'background(?:-color)?:\s*([^;]+)', style_attr)
        
        if bg_match:
            bg_color = bg_match.group(1).strip()
            
            # Handle light-dark() function - extract the first color (light mode)
            if 'light-dark' in bg_color:
                # Try to extract RGB values
                match = re.search(r'#([0-9A-Fa-f]{6})|rgb\((\d+),\s*(\d+),\s*(\d+)\)', bg_color)
                if match:
                    if match.group(1):  # Hex color
                        bg_color = f"#{match.group(1)}"
                    else:  # RGB color
                        bg_color = f"rgb({match.group(2)}, {match.group(3)}, {match.group(4)})"
                else:
                    bg_color = '#5E5B61'  # Default fallback
            
            # Get dimensions
            width = root.get('width', '100%')
            height = root.get('height', '100%')
            viewbox = root.get('viewBox', '')
            
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    x, y, vb_width, vb_height = parts
                else:
                    x, y, vb_width, vb_height = '0', '0', width, height
            else:
                x, y, vb_width, vb_height = '0', '0', width, height
            
            # Create background rect
            rect = ET.Element('rect')
            rect.set('x', x)
            rect.set('y', y)
            rect.set('width', vb_width)
            rect.set('height', vb_height)
            rect.set('fill', bg_color)
            
            # Insert as first element
            root.insert(0, rect)
            
            # Clean up style attribute
            new_style = re.sub(r'background(?:-color)?:[^;]+;?\s*', '', style_attr)
            if new_style.strip():
                root.set('style', new_style)
            elif 'style' in root.attrib:
                del root.attrib['style']
    
    def _convert_foreign_objects(self, root):
        """Convert foreignObject elements to native SVG text"""
        
        # First handle switch elements (Draw.io specific)
        for switch in list(root.iter('{http://www.w3.org/2000/svg}switch')):
            self._process_switch_element(root, switch)
        for switch in list(root.iter('switch')):
            self._process_switch_element(root, switch)
        
        # Then handle remaining foreignObject elements
        for foreign_obj in list(root.iter('{http://www.w3.org/2000/svg}foreignObject')):
            self._convert_single_foreign_object(root, foreign_obj)
        for foreign_obj in list(root.iter('foreignObject')):
            self._convert_single_foreign_object(root, foreign_obj)
    
    def _process_switch_element(self, root, switch):
        """Process Draw.io switch elements that contain foreignObject and text fallbacks"""
        
        # Find foreignObject and text elements within switch
        foreign_obj = None
        text_elem = None
        
        for child in switch:
            tag = child.tag.replace('{http://www.w3.org/2000/svg}', '')
            if tag == 'foreignObject':
                foreign_obj = child
            elif tag == 'text':
                text_elem = child
        
        if foreign_obj is not None:
            # Extract full text from foreignObject
            text_content = self._extract_text(foreign_obj)
            
            if text_content and text_elem is not None:
                # Update the existing text element with full content
                x = text_elem.get('x', '0')
                y = text_elem.get('y', '0')
                fill = text_elem.get('fill', '#000000')
                font_family = text_elem.get('font-family', 'Arial')
                font_size = text_elem.get('font-size', '12px')
                text_anchor = text_elem.get('text-anchor', 'middle')
                
                # Clear existing text content
                text_elem.clear()
                text_elem.tag = text_elem.tag  # Preserve tag
                text_elem.set('x', x)
                text_elem.set('y', y)
                text_elem.set('fill', fill)
                text_elem.set('font-family', font_family)
                text_elem.set('font-size', font_size)
                text_elem.set('text-anchor', text_anchor)
                
                # Handle multiline text
                lines = text_content.strip().split('\n')
                if len(lines) == 1:
                    text_elem.text = lines[0]
                else:
                    # Use tspan elements for multiline text
                    for i, line in enumerate(lines):
                        tspan = ET.SubElement(text_elem, 'tspan')
                        tspan.set('x', x)
                        if i > 0:
                            tspan.set('dy', '1.2em')
                        tspan.text = line
                
                # Remove the foreignObject since we've extracted its content
                switch.remove(foreign_obj)
    
    def _convert_single_foreign_object(self, root, foreign_obj):
        """Convert a single foreignObject to text"""
        
        try:
            x = float(foreign_obj.get('x', '0'))
            y = float(foreign_obj.get('y', '0'))
            width = float(foreign_obj.get('width', '100'))
            height = float(foreign_obj.get('height', '20'))
            
            # Extract text
            text_content = self._extract_text(foreign_obj)
            
            if text_content:
                # Extract styles
                style_info = self._extract_style(foreign_obj)
                
                # Create text element
                text_elem = ET.Element('text')
                text_elem.set('x', str(x + width / 2))
                text_elem.set('y', str(y + height / 2))
                text_elem.set('text-anchor', 'middle')
                text_elem.set('dominant-baseline', 'middle')
                text_elem.set('fill', style_info.get('color', '#000000'))
                text_elem.set('font-family', style_info.get('font-family', 'Arial'))
                text_elem.set('font-size', style_info.get('font-size', '12px'))
                
                # Handle multiline text
                lines = text_content.strip().split('\n')
                if len(lines) == 1:
                    text_elem.text = lines[0]
                else:
                    for i, line in enumerate(lines):
                        tspan = ET.SubElement(text_elem, 'tspan')
                        tspan.set('x', str(x + width / 2))
                        if i > 0:
                            tspan.set('dy', '1.2em')
                        tspan.text = line
                
                # Replace foreignObject
                parent = self._find_parent(root, foreign_obj)
                if parent is not None:
                    idx = list(parent).index(foreign_obj)
                    parent.remove(foreign_obj)
                    parent.insert(idx, text_elem)
        except Exception as e:
            logger.debug(f"Could not convert foreignObject: {e}")
    
    def _extract_text(self, elem):
        """Extract text from element"""
        text_parts = []
        
        def get_text(e):
            if e.text:
                text_parts.append(e.text.strip())
            for child in e:
                get_text(child)
                if child.tail:
                    text_parts.append(child.tail.strip())
        
        get_text(elem)
        return ' '.join(text_parts).strip()
    
    def _extract_style(self, elem):
        """Extract style information"""
        style_info = {
            'color': '#000000',
            'font-size': '12px',
            'font-family': 'Arial'
        }
        
        # Look for style attributes
        for e in elem.iter():
            style = e.get('style', '')
            
            # Extract color
            color_match = re.search(r'color:\s*([^;]+)', style)
            if color_match:
                color = color_match.group(1).strip()
                if 'rgb' in color:
                    style_info['color'] = color
                elif color.startswith('#'):
                    style_info['color'] = color
                elif color in ['white', 'black', 'red', 'blue', 'green']:
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
        """Ensure text elements have proper attributes"""
        
        for text_elem in root.iter('text'):
            if not text_elem.get('fill'):
                text_elem.set('fill', '#000000')
    
    def _find_parent(self, root, element):
        """Find parent of element"""
        for parent in root.iter():
            if element in list(parent):
                return parent
        return None


def convert_svg_to_pdf(svg_path, pdf_path):
    """Convert a single SVG file to PDF with preprocessing"""
    
    processor = DrawioSVGProcessor()
    
    # Preprocess the SVG
    processed_content = processor.preprocess_svg(svg_path)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False, encoding='utf-8') as tmp_svg:
        tmp_svg.write(processed_content)
        tmp_path = tmp_svg.name
    
    try:
        # Use Inkscape to convert
        inkscape_cmd = [
            'inkscape',
            '--export-area-drawing',
            '--export-text-to-path',
            f'--export-filename={pdf_path}',
            '--export-type=pdf',
            tmp_path
        ]
        
        # Try to run Inkscape
        result = subprocess.run(inkscape_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Inkscape conversion failed for {svg_path}: {result.stderr}")
            return False
        
        return os.path.exists(pdf_path)
            
    except Exception as e:
        logger.warning(f"Error converting {svg_path} to PDF: {e}")
        return False
        
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


class SVGToPDFTransform(SphinxTransform):
    """Transform that converts SVG images to PDF for LaTeX output"""
    
    # Run late to ensure images are properly resolved
    default_priority = 999
    
    def apply(self, **kwargs):
        """Apply the transformation to convert SVGs to PDFs"""
        
        # Only process for LaTeX builder
        if not isinstance(self.app.builder, LaTeXBuilder):
            return
        
        # Check if inkscape is available
        try:
            result = subprocess.run(['inkscape', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Inkscape not available, SVG to PDF conversion disabled")
                return
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Inkscape not available, SVG to PDF conversion disabled")
            return
        
        # Process all image nodes in the document
        for node in self.document.traverse(nodes.image):
            uri = node.get('uri', '')
            
            # Skip non-SVG files
            if not uri.endswith('.svg'):
                continue
            
            # Get the source file path
            if uri.startswith('/'):
                # Absolute path from source directory
                src_path = os.path.join(self.app.srcdir, uri.lstrip('/'))
            else:
                # Relative path from current document
                docdir = os.path.dirname(self.env.doc2path(self.env.docname))
                src_path = os.path.join(self.app.srcdir, docdir, uri)
            
            # Also check in _remote_images if not found
            if not os.path.exists(src_path):
                alt_path = os.path.join(self.app.srcdir, '_remote_images', os.path.basename(uri))
                if os.path.exists(alt_path):
                    src_path = alt_path
                else:
                    logger.warning(f"SVG file not found: {src_path}")
                    continue
            
            # Determine output path for PDF
            # Get the relative path from source dir
            try:
                rel_path = os.path.relpath(src_path, self.app.srcdir)
            except ValueError:
                # If on different drives on Windows
                rel_path = uri.lstrip('/')
            
            # Change extension to .pdf
            pdf_rel_path = os.path.splitext(rel_path)[0] + '.pdf'
            
            # Full path in build directory
            pdf_path = os.path.join(self.app.outdir, pdf_rel_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            
            # Check if conversion is needed (SVG is newer than PDF)
            if os.path.exists(pdf_path):
                svg_mtime = os.path.getmtime(src_path)
                pdf_mtime = os.path.getmtime(pdf_path)
                if pdf_mtime >= svg_mtime:
                    # PDF is up to date, just update the URI
                    node['uri'] = pdf_rel_path
                    logger.debug(f"Using existing PDF for {uri}")
                    continue
            
            # Convert SVG to PDF
            logger.info(f"Converting SVG to PDF: {src_path} -> {pdf_path}")
            if convert_svg_to_pdf(src_path, pdf_path):
                # Update the node to reference the PDF
                node['uri'] = pdf_rel_path
                logger.info(f"Successfully converted {uri} to PDF")
            else:
                logger.error(f"Failed to convert {uri} to PDF")


def setup(app):
    """Setup the custom SVG to PDF converter"""
    
    # Add custom transform for SVG to PDF conversion
    app.add_post_transform(SVGToPDFTransform)
    
    # Configuration for inkscape converter
    app.add_config_value('inkscape_converter_bin', 'inkscape', 'env')
    app.add_config_value('inkscape_converter_args', [
        '--export-area-drawing',
        '--export-text-to-path'
    ], 'env')
    
    logger.info("Custom Draw.io SVG to PDF converter registered")
    
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': '3.0.0'
    }
