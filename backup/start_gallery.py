#!/usr/bin/env python3
"""
Simple HTTP server to serve the embedding gallery application.
"""

import http.server
import socketserver
import os
import webbrowser
import threading
import time
import json
import subprocess
import sys
from urllib.parse import urlparse, parse_qs

class GalleryRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for API endpoints."""
        if self.path == '/api/generate-embeddings':
            self.handle_generate_embeddings()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_generate_embeddings(self):
        """Handle embedding generation requests."""
        try:
            # Get the content length
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            input_dir = data.get('input_dir', '')
            analysis_type = data.get('analysis_type', 'pca')
            use_downstream = data.get('use_downstream', False)
            specify_layer = data.get('specify_layer', False)
            layer_number = data.get('layer_number', '')
            
            if not input_dir:
                self.send_error(400, "Input directory is required")
                return
            
            # Build command
            command = [sys.executable, 'generate_embeddings_data.py', input_dir, '--output', 'embedding_data']
            if analysis_type == 'lda':
                command.append('--lda')
            if use_downstream:
                if specify_layer and layer_number:
                    command.extend(['--downstream', str(layer_number)])
                else:
                    command.extend(['--downstream', 'projected_states'])
            
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                response_data = {
                    'success': True,
                    'message': 'Embeddings generated successfully',
                    'command': ' '.join(command),
                    'stdout': result.stdout
                }
                self.send_json_response(response_data)
            else:
                response_data = {
                    'success': False,
                    'message': 'Error generating embeddings',
                    'command': ' '.join(command),
                    'error': result.stderr,
                    'stdout': result.stdout
                }
                self.send_json_response(response_data, 500)
                
        except Exception as e:
            response_data = {
                'success': False,
                'message': f'Error processing request: {str(e)}'
            }
            self.send_json_response(response_data, 500)
    
    def send_json_response(self, data, status_code=200):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_server(port=8000):
    """
    Start a simple HTTP server to serve the gallery application.
    
    Args:
        port (int): Port number to serve on (default: 8000)
    """
    # Get the current directory
    current_dir = os.getcwd()
    
    # Change to the project directory
    os.chdir(current_dir)
    
    # Create server
    with socketserver.TCPServer(("", port), GalleryRequestHandler) as httpd:
        print(f"Serving gallery application at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        print()
        print("Opening browser...")
        
        # Open browser in a separate thread after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f"http://localhost:{port}/embedding_gallery.html")
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            os.chdir(current_dir)

if __name__ == "__main__":
    start_server()
