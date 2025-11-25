from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
import requests
import re
import os
from pathlib import Path
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class BranchAwareRemoteContent(Directive):
    """
    Directive that downloads and includes content from other repositories,
    matching the branch/tag of the current documentation build.
    Usage:
    .. remote-content::
       :repo: owner/repository
       :path: path/to/file.rst
       :default_branch: docs/develop  # Branch to use when not on a release
       :tag_prefix: Docs/  # Optional
       :replace: old_text|new_text  # Replace old_text with new_text (can be used multiple times)
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False
    option_spec = {
        'repo': str,
        'path': str,
        'default_branch': str,  # Branch to use when not on a release tag
        'start_line': int,      # Include the file from a specific line
        'tag_prefix': str,      # Prefix for release tags (e.g., 'Docs/')
        'replace': str,         # Text replacement in format "old|new"
    }

    def get_current_version(self):
        """Get current version/branch being built"""
        env = self.state.document.settings.env
        html_context = env.config.html_context

        # Check if building from a tag
        if "official_branch" in html_context:
            if html_context["official_branch"] == 0:
                if "version" in html_context:
                    # Remove any 'v' prefix
                    version = html_context["version"]
                    if re.match(r'^\d+\.\d+\.\d+$', version):
                        return version

        # Not a version tag, so we'll use the default branch
        return None

    def get_target_ref(self):
        """Get target reference for the remote repository"""
        current_version = self.get_current_version()

        # If it's a version number, use tag prefix and version
        if current_version:
            tag_prefix = self.options.get('tag_prefix', '')
            return f'{tag_prefix}{current_version}'

        # For any other case, use the specified default branch
        if 'default_branch' not in self.options:
            logger.warning('No default_branch specified and not building from a version tag')
            return None

        return self.options['default_branch']

    def construct_raw_url(self, repo, path, ref):
        """Construct the raw.githubusercontent.com URL"""
        return f'https://raw.githubusercontent.com/{repo}/{ref}/{path}'

    def apply_replacements(self, content):
        """Apply text replacements to content"""
        if 'replace' not in self.options:
            return content
        
        # Get replacement specification
        replace_spec = self.options['replace']
        
        # Split by pipe character to get old and new text
        if '|' not in replace_spec:
            logger.warning('Replace option must be in format "old_text|new_text"')
            return content
        
        parts = replace_spec.split('|', 1)  # Split only on first pipe
        old_text = parts[0]
        new_text = parts[1]
        
        # Perform replacement
        modified_content = content.replace(old_text, new_text)
        
        if modified_content != content:
            logger.info(f'Replaced "{old_text}" with "{new_text}"')
        
        return modified_content

    def get_image_cache_dir(self):
        """Get or create the directory for cached remote images"""
        env = self.state.document.settings.env
        cache_dir = Path(env.app.outdir).parent / '_remote_images' / self.options['repo'].replace('/', '_')
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def download_image(self, image_path, ref):
        """Download an image from the remote repository"""
        # Construct the raw URL for the image
        image_url = self.construct_raw_url(
            self.options['repo'],
            image_path,
            ref
        )
        
        try:
            logger.info(f'Downloading image from {image_url}')
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Create cache directory and save the image
            cache_dir = self.get_image_cache_dir()
            
            # Preserve the directory structure of the image
            image_rel_path = Path(image_path)
            local_image_path = cache_dir / image_rel_path
            local_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the image content
            local_image_path.write_bytes(response.content)
            logger.info(f'Saved image to {local_image_path}')
            
            return local_image_path
            
        except requests.exceptions.RequestException as e:
            logger.warning(f'Failed to download image from {image_url}: {str(e)}')
            return None

    def process_images(self, content, ref):
        """Process image references in the content and download images"""
        # Pattern to match image and figure directives
        # Matches: .. image:: path or .. figure:: path
        image_pattern = r'\.\.\s+(image|figure)::\s+([^\s]+)'
        
        def replace_image_path(match):
            directive = match.group(1)
            original_path = match.group(2)
            
            # Skip if it's already an absolute URL
            if original_path.startswith(('http://', 'https://', '/')):
                return match.group(0)
            
            # Get the directory of the source file
            source_dir = str(Path(self.options['path']).parent)
            
            # Resolve the image path relative to the source file
            if source_dir and source_dir != '.':
                image_path = str(Path(source_dir) / original_path)
            else:
                image_path = original_path
            
            # Normalize the path
            image_path = str(Path(image_path).as_posix())
            
            # Download the image
            local_image_path = self.download_image(image_path, ref)
            
            if local_image_path:
                # Get the Sphinx source directory
                env = self.state.document.settings.env
                srcdir = Path(env.srcdir)
                
                # Make the path relative to the source directory
                try:
                    rel_path = local_image_path.relative_to(srcdir.parent)
                    new_path = '/' + str(rel_path.as_posix())
                except ValueError:
                    # If we can't make it relative, use absolute path
                    new_path = str(local_image_path.as_posix())
                
                logger.info(f'Replaced image path: {original_path} -> {new_path}')
                return f'.. {directive}:: {new_path}'
            else:
                # If download failed, keep the original path
                logger.warning(f'Keeping original image path due to download failure: {original_path}')
                return match.group(0)
        
        # Replace all image paths
        processed_content = re.sub(image_pattern, replace_image_path, content)
        
        return processed_content

    def fetch_and_parse_content(self, url, source_path, ref):
        """Fetch content and parse it as RST"""
        response = requests.get(url)
        response.raise_for_status()
        content = response.text

        # Apply text replacements before parsing
        content = self.apply_replacements(content)

        # Process images: download them and update paths
        content = self.process_images(content, ref)

        start_line = self.options.get('start_line', 0)

        # Create ViewList for parsing
        line_count = 0
        content_list = ViewList()
        for line_no, line in enumerate(content.splitlines()):
            if line_count >= start_line:
                content_list.append(line, source_path, line_no)
            line_count+=1 

        # Create a section node and parse content
        node = nodes.section()
        nested_parse_with_titles(self.state, content_list, node)

        return node.children

    def run(self):
        if 'repo' not in self.options or 'path' not in self.options:
            logger.warning('Both repo and path options are required')
            return []

        target_ref = self.get_target_ref()
        if not target_ref:
            return []

        raw_url = self.construct_raw_url(
            self.options['repo'],
            self.options['path'],
            target_ref
        )

        try:
            logger.info(f'Attempting to fetch content from {raw_url}')
            return self.fetch_and_parse_content(raw_url, self.options['path'], target_ref)
        except requests.exceptions.RequestException as e:
            logger.warning(f'Failed to fetch content from {raw_url}: {str(e)}')

            # If we failed on a tag, try falling back to default_branch
            if re.match(r'^\d+\.\d+\.\d+$', target_ref) or target_ref.startswith('Docs/'):
                if 'default_branch' in self.options:
                    try:
                        fallback_ref = self.options['default_branch']
                        logger.info(f'Attempting fallback to {fallback_ref}...')

                        fallback_url = self.construct_raw_url(
                            self.options['repo'],
                            self.options['path'],
                            fallback_ref
                        )

                        return self.fetch_and_parse_content(fallback_url, self.options['path'], fallback_ref)
                    except requests.exceptions.RequestException as e2:
                        logger.warning(f'Fallback also failed: {str(e2)}')

            return []

def setup(app):
    app.add_directive('remote-content', BranchAwareRemoteContent)

    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
