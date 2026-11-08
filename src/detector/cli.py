import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv

from detector.classifier import ClassifierRefused, classify_email
from detector.prompt_config import PromptConfig


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Classify one support email.")
    parser.add_argument("email", help="Email text, or - to read from stdin")
    parser.add_argument("--prompt", default="v1", help="Prompt version in prompts/ (default: v1)")
    args = parser.parse_args()

    email = sys.stdin.read() if args.email == "-" else args.email
    config = PromptConfig.load(args.prompt)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("No Claude API key. Put ANTHROPIC_API_KEY=... in the .env file in this folder.")

    try:
        result = classify_email(email, config)
    except anthropic.AuthenticationError:
        sys.exit("No valid Claude API key. Put ANTHROPIC_API_KEY=... in the .env file in this folder.")
    except ClassifierRefused as e:
        sys.exit(f"Model declined to classify: {e}")

    print(f"prompt:   {config.version} ({config.model}, effort={config.effort})")
    print(f"category: {result.category}")
    print(f"summary:  {result.summary}")


if __name__ == "__main__":
    main()
