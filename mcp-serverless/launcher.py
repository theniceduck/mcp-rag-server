#!/usr/bin/env python3
"""
MCP Serverless Launcher
Starts Docker container on-demand when LLM calls a tool, stops when session ends.
"""
import sys
import os
import json
import time
import threading
import signal
from pathlib import Path
import docker
from docker.errors import NotFound, APIError

# ---------- CONFIG ----------
CONTAINER_NAME = "mcp-rag-serverless"
IMAGE_NAME = "mcp-rag-server:latest"
COMPOSE_DIR = Path(__file__).parent
PROJECT_ROOT = COMPOSE_DIR.parent
CHROMA_DB_PATH = PROJECT_ROOT / "GianPDF" / "chroma_db"
UPLOAD_DIR = PROJECT_ROOT / "mcp-serverless" / "uploads"
# ----------------------------

class ServerlessLauncher:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.container_stdin = None
        self.container_stdout = None
        self.running = False
        self.log_file = open(COMPOSE_DIR / "launcher.log", "a", buffering=1)
        self.log(f"=== Serverless MCP Launcher Started ===")
        self.log(f"CHROMA_DB_PATH: {CHROMA_DB_PATH}")
        self.log(f"UPLOAD_DIR: {UPLOAD_DIR}")
        
    def log(self, msg: str):
        """Log to stderr and file"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg, file=sys.stderr, flush=True)
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.log(f"✅ Directories ready")
    
    def build_image(self):
        """Build Docker image if needed"""
        try:
            self.client.images.get(IMAGE_NAME)
            self.log(f"✅ Image {IMAGE_NAME} exists")
        except NotFound:
            self.log(f"⚠️  Image {IMAGE_NAME} not found, building...")
            # Build from mcp-rag-server directory
            build_dir = PROJECT_ROOT / "mcp-rag-server"
            self.log(f"Building from {build_dir}")
            image, logs = self.client.images.build(
                path=str(build_dir),
                tag=IMAGE_NAME,
                rm=True
            )
            for line in logs:
                if 'stream' in line:
                    self.log(line['stream'].strip())
            self.log(f"✅ Image built: {IMAGE_NAME}")
    
    def start_container(self):
        """Start the MCP server container"""
        if self.container:
            self.log("⚠️  Container already running")
            return
        
        self.log(f"🚀 Starting container {CONTAINER_NAME}...")
        
        # Remove existing container if present
        try:
            old_container = self.client.containers.get(CONTAINER_NAME)
            self.log(f"⚠️  Removing old container...")
            old_container.remove(force=True)
        except NotFound:
            pass
        
        # Start container with same config as docker-compose
        try:
            self.container = self.client.containers.run(
                IMAGE_NAME,
                name=CONTAINER_NAME,
                detach=True,
                stdin_open=True,
                tty=False,
                network_mode="host",
                volumes={
                    str(CHROMA_DB_PATH.absolute()): {'bind': '/data/chroma_db', 'mode': 'rw'},
                    str(UPLOAD_DIR.absolute()): {'bind': '/data/uploads', 'mode': 'rw'},
                },
                environment={
                    'OLLAMA_HOST': 'http://localhost:11434',
                    'EMBEDDING_MODEL': 'embeddinggemma:300m',
                    'LLM_MODEL': 'deepseek-r1:1.5b',
                    'CHROMA_DIR': '/data/chroma_db',
                    'UPLOAD_DIR': '/data/uploads',
                    'CHUNK_SIZE': '500',
                    'CHUNK_OVERLAP': '100',
                    'TOP_K_DEFAULT': '3',
                    'TEMPERATURE': '0.1',
                    'MAX_CONTEXT_CHARS': '4000',
                },
                remove=False  # We'll remove manually on shutdown
            )
            
            # Wait a bit for server to initialize
            time.sleep(2)
            
            # Get stdin/stdout streams
            self.container_stdin = self.container.attach_socket(params={'stdin': 1, 'stream': 1})
            self.container_stdout = self.container.attach(stdout=True, stderr=False, stream=True, logs=False)
            
            self.running = True
            self.log(f"✅ Container {CONTAINER_NAME} started")
            
        except Exception as e:
            self.log(f"❌ Failed to start container: {e}")
            raise
    
    def stop_container(self):
        """Stop and remove the container"""
        if not self.container:
            return
        
        self.log(f"🛑 Stopping container {CONTAINER_NAME}...")
        self.running = False
        
        try:
            self.container.stop(timeout=5)
            self.container.remove()
            self.log(f"✅ Container stopped and removed")
        except Exception as e:
            self.log(f"⚠️  Error stopping container: {e}")
        
        self.container = None
        self.container_stdin = None
        self.container_stdout = None
    
    def proxy_stdin_to_container(self):
        """Forward stdin to container"""
        try:
            while self.running:
                line = sys.stdin.buffer.readline()
                if not line:
                    self.log("📭 stdin EOF detected, stopping container")
                    break
                
                if self.container_stdin:
                    self.container_stdin._sock.sendall(line)
        except Exception as e:
            self.log(f"❌ stdin proxy error: {e}")
        finally:
            self.stop_container()
    
    def proxy_container_to_stdout(self):
        """Forward container output to stdout"""
        try:
            for chunk in self.container_stdout:
                if not self.running:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        except Exception as e:
            self.log(f"❌ stdout proxy error: {e}")
    
    def run(self):
        """Main run loop"""
        try:
            # Ensure environment is ready
            self.ensure_directories()
            self.build_image()
            
            # Start container
            self.start_container()
            
            # Start proxy threads
            stdin_thread = threading.Thread(target=self.proxy_stdin_to_container, daemon=False)
            stdout_thread = threading.Thread(target=self.proxy_container_to_stdout, daemon=False)
            
            stdin_thread.start()
            stdout_thread.start()
            
            # Wait for threads to complete
            stdin_thread.join()
            stdout_thread.join()
            
        except KeyboardInterrupt:
            self.log("🛑 Interrupted by user")
        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            self.stop_container()
            self.log("=== Launcher stopped ===")
            self.log_file.close()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown signal received", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run launcher
    launcher = ServerlessLauncher()
    launcher.run()

