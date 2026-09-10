"""Phase 4 CLI - thin wrapper around answer_conversational(), with in-memory
conversation state carried across turns in the interactive loop.

Usage:
    python src/chat.py                      # interactive loop, multi-turn
    python src/chat.py --query "..."        # single-shot, non-interactive
    python src/chat.py --debug              # also print intent/retrieval routing info
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from answer import answer_conversational
from conversation_state import ConversationState
from generate import format_timestamp


def print_result(result: dict, debug: bool = False) -> None:
    if debug:
        print(
            f"[intent: {result['query_type']} | retrieval: {result['retrieval_mode']} | "
            f"chunks: {len(result['retrieved_chunk_ids'])} | unknown_episodes: {result['unknown_episodes']}]"
        )
    print(f"\n{result['answer']}\n")
    if result["citations"]:
        print("Sources:")
        for c in result["citations"]:
            print(
                f"  - Episode {c['episode_number']}: {c['episode_title']} "
                f"({format_timestamp(c['start'])}-{format_timestamp(c['end'])})"
            )
    print()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask Tracer a question about the podcast catalogue.")
    parser.add_argument("--query", help="Single question to ask (non-interactive mode)")
    parser.add_argument("--debug", action="store_true", help="Print intent + retrieval routing info")
    args = parser.parse_args()

    state = ConversationState()

    if args.query:
        result = answer_conversational(args.query, state)
        print_result(result, debug=args.debug)
        return

    print("Tracer - ask a question about the podcast catalogue. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        result = answer_conversational(query, state)
        print_result(result, debug=args.debug)


if __name__ == "__main__":
    main()
