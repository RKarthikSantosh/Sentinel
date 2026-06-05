import os

from dotenv import load_dotenv
import google.generativeai as genai

from attack_mapper import get_attack_description


# ------------------------------------
# Load API Keys
# ------------------------------------

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)


# ------------------------------------
# Report Generator
# ------------------------------------

def generate_report(
    attack_name,
    confidence,
    risk_score,
    threat_level
):

    attack_description = get_attack_description(
        attack_name
    )

    prompt = f"""
You are a cybersecurity assistant.

Generate a professional security incident report for a NON-TECHNICAL USER.

Attack Type:
{attack_description}

Confidence:
{confidence:.2f}%

Risk Score:
{risk_score}/100

Threat Level:
{threat_level}

The report must contain:

1. Incident Summary
2. What This Means
3. Potential Impact
4. Recommended Actions

Use simple language.
Avoid technical jargon.
"""

    try:

        response = model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return f"""
SECURITY INCIDENT REPORT

Attack Type:
{attack_description}

Threat Level:
{threat_level}

Confidence:
{confidence:.2f}%

Risk Score:
{risk_score}/100

Error generating AI report:
{str(e)}
"""


# ------------------------------------
# Testing
# ------------------------------------

if __name__ == "__main__":

    report = generate_report(
        attack_name="neptune",
        confidence=100.0,
        risk_score=100,
        threat_level="Critical"
    )

    print(report)