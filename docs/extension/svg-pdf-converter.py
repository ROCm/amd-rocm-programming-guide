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

            # A transparent background must not become a filled rect. Emitting
            # fill="transparent" makes inkscape render the area black when
            # flattening to PDF, which fills the rounded corners with black.
            # Skip the rect entirely so the page shows through.
            if bg_color.lower() in ('transparent', 'none'):
                return

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

            # Skip Draw.io's hidden diagram metadata. The whole switch is
            # non-visual: both the foreignObject and its <text> fallback carry
            # the URL-encoded mxGraphModel string, so remove the entire switch,
            # not just the foreignObject. Rendering it corrupts the drawing
            # bounds and leaks the raw text into the diagram.
            if self._is_nonvisual_metadata(text_content):
                parent = self._find_parent(root, switch)
                if parent is not None:
                    parent.remove(switch)
                return

            if text_content and text_elem is not None:
                # Update the existing text element with full content
                x = text_elem.get('x', '0')
                y = text_elem.get('y', '0')
                fill = text_elem.get('fill', '#000000')
                font_family = text_elem.get('font-family', 'Arial')
                font_size = text_elem.get('font-size', '12px')
                text_anchor = text_elem.get('text-anchor', 'middle')

                # Draw.io's <text> fallback hardcodes fill (often #FFFFFF),
                # but the real color lives in the foreignObject's inline style
                # as a light-dark() value. Prefer that so labels on the light
                # page background remain visible in the PDF.
                fo_color = self._extract_style(foreign_obj).get('color')
                if fo_color:
                    fill = fo_color
                
                # Clear existing text content
                text_elem.clear()
                text_elem.tag = text_elem.tag  # Preserve tag
                text_elem.set('x', x)
                text_elem.set('y', y)
                text_elem.set('fill', fill)
                text_elem.set('font-family', font_family)
                text_elem.set('font-size', font_size)
                text_elem.set('text-anchor', text_anchor)
                
                # Draw.io wraps long labels inside a fixed-width foreignObject
                # box (two lines in HTML). The single-line <text> fallback keeps
                # the full string, so it overflows the canvas and gets clipped in
                # the PDF. Re-wrap to the box width to reproduce the layout.
                box_width = self._foreign_object_box_width(foreign_obj)
                lines = text_content.strip().split('\n')
                if len(lines) == 1 and box_width:
                    lines = self._wrap_text(lines[0], box_width, font_size)

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
    
    def _is_nonvisual_metadata(self, text_content):
        """Detect Draw.io's hidden editable-diagram metadata.

        Draw.io embeds the URL-encoded mxGraphModel source in a foreignObject
        styled with font-size: 0 and a transparent color. It is not meant to
        render; converting it to a visible <text> produces a huge off-canvas
        string that blows out --export-area-drawing and squishes the diagram.
        """
        if not text_content:
            return False
        stripped = text_content.lstrip()
        return stripped.startswith('%3CmxGraphModel') or stripped.startswith('<mxGraphModel')

    def _convert_single_foreign_object(self, root, foreign_obj):
        """Convert a single foreignObject to text"""

        try:
            x = float(foreign_obj.get('x', '0'))
            y = float(foreign_obj.get('y', '0'))
            width = float(foreign_obj.get('width', '100'))
            height = float(foreign_obj.get('height', '20'))

            # Extract text
            text_content = self._extract_text(foreign_obj)

            # Skip Draw.io's hidden diagram metadata; rendering it corrupts the
            # drawing bounds.
            if self._is_nonvisual_metadata(text_content):
                parent = self._find_parent(root, foreign_obj)
                if parent is not None:
                    parent.remove(foreign_obj)
                return

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

    def _foreign_object_box_width(self, foreign_obj):
        """Return the wrapping width Draw.io uses for a label, in px.

        The visible text sits in a nested <div> whose inline style carries a
        pixel width (e.g. "width: 61px"). That width is what constrains
        line-wrapping in HTML; the foreignObject itself is usually width="100%".
        Returns None when no explicit width is found.
        """
        for e in foreign_obj.iter():
            style = e.get('style', '')
            m = re.search(r'(?<!max-)(?<!min-)width:\s*([\d.]+)px', style)
            if m:
                w = float(m.group(1))
                if w > 1:
                    return w
        return None

    def _wrap_text(self, text, box_width, font_size):
        """Greedily wrap text to fit box_width, approximating glyph advance.

        Only wraps between words; a single word wider than the box is left on
        its own line rather than being split.
        """
        try:
            size = float(re.match(r'([\d.]+)', str(font_size)).group(1))
        except (AttributeError, ValueError):
            size = 12.0
        # Average glyph advance for a proportional sans font is ~0.55em.
        char_w = size * 0.55
        max_chars = max(1, int(box_width / char_w))

        words = text.split()
        if not words:
            return [text]
        lines = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= max_chars:
                current += ' ' + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _resolve_color(self, color):
        """Resolve a CSS color value to something valid for an SVG fill.

        Draw.io uses light-dark(<light>, <dark>) for theme-aware colors. The
        PDF is rendered on a light background, so take the first (light-mode)
        argument. Returns an empty string for values that can't be used.
        """
        color = color.strip()
        if color.startswith('light-dark('):
            inner = color[len('light-dark('):]
            # Split on the top-level comma (rgb(...) contains commas too).
            depth = 0
            split_at = None
            for i, ch in enumerate(inner):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == ',' and depth == 0:
                    split_at = i
                    break
            color = (inner[:split_at] if split_at is not None else inner).strip()
        if color.startswith('#') or color.startswith('rgb'):
            return color
        if color in ('white', 'black', 'red', 'blue', 'green'):
            return color
        return ''

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
            
            # Extract color. Draw.io emits light-dark(<light>, <dark>) for
            # theme-aware text; resolve it to the light-mode value since the
            # PDF is rendered on a light background.
            color_match = re.search(r'(?<!-)color:\s*([^;]+)', style)
            if color_match:
                color = self._resolve_color(color_match.group(1).strip())
                if color:
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
