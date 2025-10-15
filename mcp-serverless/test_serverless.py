#!/usr/bin/env python3
"""
Test script for MCP Serverless launcher
Verifies that the launcher starts, processes requests, and shuts down properly.
"""
import json
import subprocess
import sys
import time
import threading

def send_jsonrpc_request(proc, method, params=None, request_id=1):
    """Send a JSON-RPC request to the launcher process"""
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {}
    }
    
    request_line = json.dumps(request) + "\n"
    print(f"📤 Sending: {method}", file=sys.stderr)
    proc.stdin.write(request_line.encode())
    proc.stdin.flush()

def read_jsonrpc_response(proc, timeout=30):
    """Read a JSON-RPC response from the launcher process"""
    print(f"📥 Waiting for response (timeout: {timeout}s)...", file=sys.stderr)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        
        try:
            response = json.loads(line.decode())
            print(f"✅ Received response: {response.get('method', response.get('result', 'unknown'))}", file=sys.stderr)
            return response
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON: {line.decode()[:100]}", file=sys.stderr)
            continue
    
    print(f"❌ Timeout after {timeout}s", file=sys.stderr)
    return None

def test_serverless_launcher():
    """Test the serverless launcher"""
    print("=" * 60, file=sys.stderr)
    print("🧪 Testing MCP Serverless Launcher", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Start launcher
    print("\n1️⃣  Starting launcher.py...", file=sys.stderr)
    proc = subprocess.Popen(
        [sys.executable, "launcher.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    # Give it time to start
    print("⏳ Waiting 5s for container startup...", file=sys.stderr)
    time.sleep(5)
    
    try:
        # Test 1: Initialize
        print("\n2️⃣  Sending initialize request...", file=sys.stderr)
        send_jsonrpc_request(proc, "initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }, request_id=1)
        
        response = read_jsonrpc_response(proc, timeout=10)
        if response and "result" in response:
            print("✅ Initialize successful", file=sys.stderr)
        else:
            print("❌ Initialize failed", file=sys.stderr)
            return False
        
        # Test 2: List tools
        print("\n3️⃣  Sending list_tools request...", file=sys.stderr)
        send_jsonrpc_request(proc, "tools/list", {}, request_id=2)
        
        response = read_jsonrpc_response(proc, timeout=10)
        if response and "result" in response:
            tools = response["result"].get("tools", [])
            print(f"✅ Tools listed: {len(tools)} tools found", file=sys.stderr)
            for tool in tools[:3]:
                print(f"   - {tool.get('name')}", file=sys.stderr)
        else:
            print("❌ List tools failed", file=sys.stderr)
            return False
        
        # Test 3: Call a tool (list_sessions)
        print("\n4️⃣  Calling tool: list_sessions...", file=sys.stderr)
        send_jsonrpc_request(proc, "tools/call", {
            "name": "list_sessions",
            "arguments": {}
        }, request_id=3)
        
        response = read_jsonrpc_response(proc, timeout=30)
        if response and "result" in response:
            print("✅ Tool call successful", file=sys.stderr)
            content = response["result"].get("content", [])
            if content:
                try:
                    data = json.loads(content[0].get("text", "{}"))
                    print(f"   Sessions found: {data.get('count', 0)}", file=sys.stderr)
                except:
                    pass
        else:
            print("❌ Tool call failed", file=sys.stderr)
            return False
        
        print("\n5️⃣  All tests passed! ✅", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"\n❌ Test error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n6️⃣  Shutting down launcher...", file=sys.stderr)
        proc.stdin.close()
        
        # Wait for graceful shutdown
        try:
            proc.wait(timeout=10)
            print("✅ Launcher stopped gracefully", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("⚠️  Forcing shutdown...", file=sys.stderr)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        
        # Print stderr output
        stderr_output = proc.stderr.read().decode()
        if stderr_output:
            print("\n📋 Launcher logs:", file=sys.stderr)
            print(stderr_output, file=sys.stderr)

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════╗
║   MCP Serverless Launcher - Test Suite       ║
╚═══════════════════════════════════════════════╝

This test will:
  1. Start launcher.py
  2. Wait for Docker container to start (~5s)
  3. Send MCP initialize request
  4. List available tools
  5. Call list_sessions tool
  6. Shutdown gracefully

""", file=sys.stderr)
    
    success = test_serverless_launcher()
    
    if success:
        print("\n" + "=" * 60, file=sys.stderr)
        print("✅ ALL TESTS PASSED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(0)
    else:
        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ TESTS FAILED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

