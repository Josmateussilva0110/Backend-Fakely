import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLAYGROUND_DIR = Path(__file__).resolve().parent.parent / "tools" / "playground"
DEFAULT_PORT = 5173


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    handler = partial(SimpleHTTPRequestHandler, directory=str(PLAYGROUND_DIR))
    print(f"Playground em http://localhost:{port}  (Ctrl+C para parar)")
    print(f'A API precisa estar rodando (python -m app) e com "http://localhost:{port}" em CORS_ORIGINS.')
    # Só a própria máquina acessa a página
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nPlayground encerrado.")


if __name__ == "__main__":
    main()
