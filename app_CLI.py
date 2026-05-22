"""
CLI entry point for the DIEM Agentic RAG project.

Provides a simple terminal interface for interacting with the DIEM
assistant. All system logic is delegated to core.py.

Usage:
    python app_CLI.py
"""

from core import ask, initialize_system, update_history


def main() -> None:
    print("\n" + "=" * 65)
    print(" DIEM RAG — Assistente Virtuale del Dipartimento ".center(65))
    print("=" * 65)
    print(" Digita la tua domanda oppure 'exit' per uscire.\n")

    graph = initialize_system()
    chat_history = []

    while True:
        try:
            query = input("[Utente]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArrivederci.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "esci"):
            print("Arrivederci.")
            break

        answer = ask(graph, query, chat_history)
        print(f"\n[DIEM]: {answer}\n")
        chat_history = update_history(chat_history, query, answer)


if __name__ == "__main__":
    main()