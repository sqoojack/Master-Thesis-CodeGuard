import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

# Import the defense framework functions from main.py
from main import (
    build_arg_parser,
    apply_params_file,
    load_model_and_tokenizer,
    build_guardrails,
    run_pipeline
)

# Global variables for shared defense states
PROXY_ARGS = None
PROXY_GUARDRAILS = None
TARGET_API_URL = "https://api.openai.com/v1/chat/completions"

def initialize_defense():
    """Perform one-time setup of the pipeline models and guardrails."""
    global PROXY_ARGS, PROXY_GUARDRAILS
    print("Pre-loading model guardrails for Inline API Proxy...")
    parser = build_arg_parser()
    PROXY_ARGS = parser.parse_args(["--attack_type", "default_attack", "--mode", "all"])
    try:
        PROXY_ARGS = apply_params_file(PROXY_ARGS)
    except Exception:
        print("Using standard baseline parameters for proxy pipeline configuration.")
    
    model, tokenizer, device = load_model_and_tokenizer(PROXY_ARGS)
    PROXY_GUARDRAILS = build_guardrails(PROXY_ARGS, model, tokenizer, device)
    print("Proxy pipeline initialization successfully finalized.")

def sanitize_prompt_text(text: str) -> str:
    """Find markdown code blocks within the text and pass them to the pipeline."""
    block_pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

    def replacement_hook(match):
        lang = match.group(1).strip() or "c"
        inner_code = match.group(2)
        print(f"[Proxy] Intercepted code block target language: {lang}")
        res = run_pipeline(inner_code, lang, PROXY_GUARDRAILS, PROXY_ARGS)
        return f"```{lang}\n{res.get('final_code', inner_code)}```"

    return block_pattern.sub(replacement_hook, text)

class InterceptingProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Intercept outbound API completion payloads, clean them, and proxy forward."""
        content_length = int(self.headers.get('Content-Length', 0))
        raw_body = self.rfile.read(content_length)
        
        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except Exception:
            payload = None

        if payload and "messages" in payload:
            print("[Proxy] Scanning inbound chat completion message array...")
            for message in payload["messages"]:
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    original_content = message["content"]
                    sanitized_content = sanitize_prompt_text(original_content)
                    message["content"] = sanitized_content
            
            raw_body = json.dumps(payload).encode('utf-8')

        # Forward request to upstream provider endpoints
        print(f"[Proxy] Forwarding sanitized payload upstream to: {TARGET_API_URL}")
        upstream_headers = {
            "Content-Type": "application/json",
        }
        if "Authorization" in self.headers:
            upstream_headers["Authorization"] = self.headers["Authorization"]
        if "api-key" in self.headers:
            upstream_headers["api-key"] = self.headers["api-key"]

        upstream_req = urllib.request.Request(
            TARGET_API_URL,
            data=raw_body,
            headers=upstream_headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(upstream_req) as upstream_response:
                response_status = upstream_response.status
                response_body = upstream_response.read()
                response_headers = upstream_response.info()
        except urllib.error.HTTPError as http_err:
            response_status = http_err.code
            response_body = http_err.read()
            response_headers = http_err.info()
        except Exception as generic_err:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Proxy Forward Failure: {generic_err}".encode('utf-8'))
            return

        # Send response status and data back to caller IDE client
        self.send_response(response_status)
        for key, value in response_headers.items():
            if key.lower() not in ["content-length", "transfer-encoding", "connection"]:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

def run_proxy_server(port=8080):
    initialize_defense()
    server_address = ('', port)
    httpd = HTTPServer(server_address, InterceptingProxyHandler)
    print(f"[Proxy] Inline interceptor running on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_proxy_server()