"""
Example demonstrating automatic logging via tracked OpenAI client.

This example relies on the client-level tracking configured in core/client.py.
No decorators needed - all OpenAI API calls are automatically logged to Opik.
"""

from llm_debate_assistant.core.assistant import DebateAssistant


def main():
    # Initialize the DebateAssistant
    # The OpenAI client is automatically tracked via track_openai() in core/client.py
    print("Initializing DebateAssistant...")
    assistant = DebateAssistant(model="gpt-5-mini-2025-08-07")

    # Define debate topic and side
    topic = "死刑应该/不应该被废除"
    side = "不应该"  # or "negative"

    print("\nGenerating debate outline for:")
    print(f"Topic: {topic}")
    print(f"Side: {side}")
    print("-" * 70)

    # Call the generate_debate_outline function
    # All LLM calls are automatically logged to Opik
    outline = assistant.generate_debate_outline(topic=topic, side=side)

    # Display the generated outline
    print("\n=== DEBATE OUTLINE ===\n")

    print("📚 Keywords & Definitions:")
    for kw in outline.get("keywords_definitions", []):
        print(f"  • {kw['keyword']}: {kw['definition']}")

    print("\n⚖️ Weighing Criterion:")
    print(f"  {outline.get('weighing_criterion', 'N/A')}")

    print("\n💡 Arguments:")
    for i, arg in enumerate(outline.get("arguments", []), 1):
        print(f"\n  Argument {i}:")
        print(f"    Claim: {arg['argument']}")
        print(f"    Warrant: {arg['warrant']}")
        print("    Evidence Needed:")
        for evidence in arg.get("evidence_needed", []):
            print(f"      - {evidence}")

    print("\n" + "=" * 70)
    print("✅ Debate outline generated successfully!")
    print("📊 Check your Opik dashboard for logged traces!")

    return outline


if __name__ == "__main__":
    result = main()
