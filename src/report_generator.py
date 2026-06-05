import re

from fpdf import FPDF

from attack_mapper import get_attack_description


ATTACK_DETAILS = {
    "neptune": {
        "what_detected": (
            "The system identified network traffic patterns that closely match a "
            "Denial of Service (DoS) attack.\n\n"
            "This activity suggests an attempt to overload network resources with "
            "excessive requests, which may affect the availability and performance "
            "of services."
        ),
        "impact": [
            "Slower network performance",
            "Service interruptions",
            "Increased resource consumption",
            "Reduced availability of applications",
        ],
        "why_flagged": [
            "Abnormally high connection activity",
            "Large traffic volume in a short period",
            "Patterns similar to known DoS attacks",
        ],
    },
    "smurf": {
        "what_detected": (
            "The system identified traffic patterns consistent with a "
            "Distributed Denial of Service (DDoS) attack.\n\n"
            "This activity may involve amplified or coordinated traffic aimed at "
            "overwhelming network devices or services."
        ),
        "impact": [
            "Widespread service degradation",
            "Network congestion",
            "Loss of availability for critical systems",
            "Increased load on firewalls and routers",
        ],
        "why_flagged": [
            "High volume of ICMP or broadcast-related traffic",
            "Sudden spikes from multiple sources",
            "Behavior matching known DDoS patterns",
        ],
    },
    "satan": {
        "what_detected": (
            "The system detected activity that resembles port scanning, often "
            "associated with reconnaissance before an attack.\n\n"
            "An attacker may be probing open ports to find vulnerable services."
        ),
        "impact": [
            "Exposure of running services",
            "Increased risk of follow-up exploitation",
            "Unnecessary load on targeted hosts",
            "Early-stage indicator of a larger attack",
        ],
        "why_flagged": [
            "Repeated connection attempts across many ports",
            "Short-lived probes to multiple services",
            "Patterns similar to known scanning tools",
        ],
    },
    "ipsweep": {
        "what_detected": (
            "The system detected host discovery activity (IP sweep) on the network.\n\n"
            "This behavior often indicates an attacker mapping live hosts before "
            "targeting specific systems."
        ),
        "impact": [
            "Visibility into active hosts on the network",
            "Higher risk of targeted follow-up attacks",
            "Unusual ICMP or probe traffic",
            "Potential precursor to intrusion attempts",
        ],
        "why_flagged": [
            "Sequential probes across IP ranges",
            "Many hosts contacted in a short window",
            "Patterns consistent with network reconnaissance",
        ],
    },
    "portsweep": {
        "what_detected": (
            "The system detected port sweep activity across one or more hosts.\n\n"
            "This suggests an attempt to identify which services are reachable "
            "on target systems."
        ),
        "impact": [
            "Discovery of open services",
            "Increased attack surface awareness for adversaries",
            "Noise and load on monitored hosts",
            "Possible staging for exploitation",
        ],
        "why_flagged": [
            "Multiple ports tested on the same host",
            "Rapid, systematic connection attempts",
            "Similarity to known port-scan signatures",
        ],
    },
    "nmap": {
        "what_detected": (
            "The system detected network mapping activity consistent with "
            "reconnaissance tools.\n\n"
            "This may indicate an attacker cataloging services, versions, or "
            "topology before a targeted attack."
        ),
        "impact": [
            "Detailed fingerprinting of network assets",
            "Easier planning for follow-up attacks",
            "Policy violations on production networks",
            "Elevated insider or external threat risk",
        ],
        "why_flagged": [
            "Structured probes across hosts and ports",
            "Behavior aligned with mapping tool signatures",
            "Unusual timing or volume of discovery traffic",
        ],
    },
}

DEFAULT_DETAILS = {
    "what_detected": (
        "The system identified network traffic patterns that match a known "
        "intrusion or anomaly signature.\n\n"
        "This activity differs from normal baseline behavior and may indicate "
        "hostile or unauthorized use of the network."
    ),
    "impact": [
        "Degraded network or application performance",
        "Possible unauthorized access attempts",
        "Increased operational and security risk",
        "Need for further investigation by the security team",
    ],
    "why_flagged": [
        "Traffic features inconsistent with normal activity",
        "High model confidence on an attack classification",
        "Similarity to patterns seen in historical attack data",
    ],
}

IMMEDIATE_ACTIONS = [
    "Review firewall and network logs",
    "Monitor incoming traffic",
    "Block suspicious IP addresses if identified",
    "Verify service availability",
]

FOLLOW_UP_ACTIONS = [
    "Investigate affected systems",
    "Check for repeated attack attempts",
    "Update firewall rules if necessary",
]


def _get_details(attack_name):
    return ATTACK_DETAILS.get(attack_name, DEFAULT_DETAILS)


def _bullet_list(items):
    return "\n".join(f"* {item}" for item in items)


def format_security_report(
    attack_name,
    confidence,
    risk_score,
    threat_level,
):
    attack_description = get_attack_description(attack_name)
    details = _get_details(attack_name)

    confidence_str = f"{confidence:.2f}%"
    risk_str = f"{risk_score} / 100"

    return f"""# Security Incident Report

## Detection Summary

| Item         | Value                   |
| ------------ | ----------------------- |
| Attack Type  | {attack_description} |
| Threat Level | {threat_level}                |
| Confidence   | {confidence_str}                  |
| Risk Score   | {risk_str}               |

---

## What Was Detected?

{details["what_detected"]}

---

## Potential Impact

{_bullet_list(details["impact"])}

---

## Why Was It Flagged?

The detection model observed unusual traffic behavior, including:

{_bullet_list(details["why_flagged"])}

---

## Recommended Actions

### Immediate

{_bullet_list(IMMEDIATE_ACTIONS)}

### Follow-Up

{_bullet_list(FOLLOW_UP_ACTIONS)}

---

## Risk Assessment

This event has been classified as **{threat_level}** due to the high confidence of detection and the potential impact on network availability.

Immediate investigation is recommended.
"""


def generate_report(
    attack_name,
    confidence,
    risk_score,
    threat_level,
):
    return format_security_report(
        attack_name=attack_name,
        confidence=confidence,
        risk_score=risk_score,
        threat_level=threat_level,
    )


def _clean_markdown(text):
    return text.replace("**", "").strip()


def _is_table_separator(line):
    return bool(re.match(r"^\|[\s\-:|]+\|$", line))


def _pdf_write_line(pdf, text, height, style="", size=10):
    pdf.set_x(pdf.l_margin)
    if style:
        pdf.set_font("Helvetica", style, size)
    else:
        pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(pdf.epw, height, text)


def report_to_pdf_bytes(report_markdown):
    """Convert the markdown security report to a PDF byte stream."""
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for line in report_markdown.splitlines():
        stripped = line.strip()

        if not stripped or stripped == "---":
            continue

        if stripped.startswith("# "):
            _pdf_write_line(pdf, _clean_markdown(stripped[2:]), 10, "B", 16)
            pdf.ln(2)
            continue

        if stripped.startswith("## "):
            pdf.ln(4)
            _pdf_write_line(pdf, _clean_markdown(stripped[3:]), 8, "B", 13)
            pdf.ln(1)
            continue

        if stripped.startswith("### "):
            pdf.ln(2)
            _pdf_write_line(pdf, _clean_markdown(stripped[4:]), 7, "B", 11)
            continue

        if stripped.startswith("|"):
            if _is_table_separator(stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) >= 2 and cells[0].lower() != "item":
                row_text = f"{cells[0]}: {cells[1]}"
            else:
                row_text = " | ".join(cells)
            _pdf_write_line(pdf, row_text, 6)
            continue

        if stripped.startswith("* "):
            _pdf_write_line(pdf, f"- {_clean_markdown(stripped[2:])}", 6)
            continue

        _pdf_write_line(pdf, _clean_markdown(stripped), 6)

    return bytes(pdf.output())


if __name__ == "__main__":
    print(
        generate_report(
            attack_name="neptune",
            confidence=99.85,
            risk_score=100,
            threat_level="Critical",
        )
    )
