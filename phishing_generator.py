import json
import random
import argparse
import os
from datetime import datetime

# --- Optional APIs ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# --- Static phishing templates ---
EASY_TEMPLATES = [
    {
        "subject": "Urgent! Your account will be closed",
        "sender": "support@paypall-security.com",
        "body": "Dear Customer, Your acc0unt is blocked. Cl1ck here to fix: http://fake-link.com",
        "technique": "Generic scam with spelling mistakes"
    },
    {
        "subject": "YOU WON $1,000,000",
        "sender": "lottery@bigprize.ru",
        "body": "Congratulation dear user!! Click fast to cliam your big prize: http://scamlink.win",
        "technique": "Lottery scam"
    }
]

INTERMEDIATE_TEMPLATES = [
    {
        "subject": "Password Reset Request",
        "sender": "it-support@micros0ft.com",
        "body": "We detected unusual login attempts. Reset your password now: http://security-microsoft-login.com",
        "technique": "Fake IT support"
    },
    {
        "subject": "Delivery Issue: Package Held",
        "sender": "noreply@fedex-support.co",
        "body": "We could not deliver your package. Please confirm address: http://track-fedex.info",
        "technique": "Delivery phishing"
    }
]

ADVANCED_TEMPLATES = [
    {
        "subject": "Updated Project Files",
        "sender": "john.doe@trustedpartner.com",
        "body": "Hi Sarah,\n\nAttached are the latest Q3 project reports. Please review before our meeting tomorrow.\n\nRegards,\nJohn",
        "technique": "Spear phishing (business email compromise)"
    },
    {
        "subject": "Action Required: Payroll Verification",
        "sender": "hr@company-payroll.com",
        "body": "Dear Employee,\n\nWe updated our payroll system. Please verify your banking info securely: http://company-hr-portal.com/login",
        "technique": "HR-themed phishing"
    }
]

TEMPLATES = {
    "easy": EASY_TEMPLATES,
    "intermediate": INTERMEDIATE_TEMPLATES,
    "advanced": ADVANCED_TEMPLATES
}


def generate_samples_static(difficulty, count):
    """Generate phishing email samples using static templates."""
    if difficulty not in TEMPLATES:
        raise ValueError(f"Invalid difficulty level: {difficulty}")

    samples = []
    for _ in range(count):
        template = random.choice(TEMPLATES[difficulty])
        sample = {
            "difficulty": difficulty,
            "subject": template["subject"],
            "sender": template["sender"],
            "body": template["body"],
            "technique": template["technique"],
            "timestamp": datetime.now().isoformat()
        }
        samples.append(sample)
    return samples


def generate_samples_openai(difficulty, count, model="gpt-4o-mini"):
    """Generate phishing samples with OpenAI API."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not installed. Run `pip install openai`.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    Generate {count} synthetic phishing email samples for SOC training.
    Difficulty: {difficulty}.
    Output must be JSON array with: difficulty, subject, sender, body, technique.
    Use only FAKE URLs.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You generate phishing samples for SOC training."},
                  {"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)


import re

def generate_samples_groq(difficulty, count, model="llama-3.1-8b-instant"):
    """Generate phishing samples with Groq API."""
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq not installed. Run `pip install groq`.")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""
    Generate {count} synthetic phishing email samples for SOC training.
    Difficulty: {difficulty}.
    Output ONLY valid JSON array with objects containing:
    - difficulty
    - subject
    - sender
    - body
    - technique
    Example:
    [
      {{
        "difficulty": "easy",
        "subject": "Urgent! Your account will be closed",
        "sender": "support@paypall-security.com",
        "body": "Dear Customer, your acc0unt is blocked. Cl1ck here to fix: http://fake-link.com",
        "technique": "Generic scam"
      }}
    ]
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a JSON generator. Always return valid JSON only."},
                  {"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content.strip()

    # Try direct JSON parse
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Extract JSON part with regex if model added extra text
        match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("Groq response was not valid JSON:\n" + raw_output)



def save_to_json(samples, output_file="phishing_samples.json"):
    """Save samples to JSON file (append)."""
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.extend(samples)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Email Generator (SOC Training)")
    parser.add_argument("-d", "--difficulty", type=str, required=True,
                        choices=["easy", "intermediate", "advanced"], help="Difficulty level")
    parser.add_argument("-n", "--number", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("-o", "--output", type=str, default="phishing_samples.json", help="Output JSON file")
    parser.add_argument("--llm", type=str, choices=["openai", "groq"], help="Choose LLM provider")

    args = parser.parse_args()

    if args.llm == "openai" and os.getenv("OPENAI_API_KEY"):
        print("[+] Using OpenAI API...")
        samples = generate_samples_openai(args.difficulty, args.number)
    elif args.llm == "groq" and os.getenv("GROQ_API_KEY"):
        print("[+] Using Groq API...")
        samples = generate_samples_groq(args.difficulty, args.number)
    else:
        print("[+] Using static templates...")
        samples = generate_samples_static(args.difficulty, args.number)

    save_to_json(samples, args.output)
    print(f"[+] Generated {args.number} {args.difficulty} phishing samples → {args.output}")
