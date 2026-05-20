"""
De-identification validation for the 10Web capstone dataset.

Synthetic tests: seeds known PII into test sentences and verifies every pattern
is caught. Hard pass/fail.

BigQuery residual scan (--bq-scan): queries the live chatbot_messages_deidentified
table and counts rows with regex-detectable PII patterns for manual review.

Usage:
    python validate_deidentification.py
    python validate_deidentification.py --bq-scan
"""

import re
import sys
import argparse
import logging

from deidentify import deidentify_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


SYNTHETIC_CASES: list[tuple[str, str, list[str]]] = [
    ("Email in plain text",
     "Please contact me at anna.minasyan@example.com for help.",
     ["anna.minasyan@example.com"]),
    ("Email with plus alias",
     "My address is support+test@mycompany.io.",
     ["support+test@mycompany.io"]),
    ("IPv4 address",
     "The server at 192.168.10.254 is down.",
     ["192.168.10.254"]),
    ("IPv4 in URL context",
     "Try http://10.0.0.1/wp-admin/",
     ["10.0.0.1"]),
    ("Full HTTPS URL",
     "Visit https://anna-mysite.com/wp-admin/users.php for the dashboard.",
     ["anna-mysite.com/wp-admin", "https://anna-mysite.com"]),
    ("WordPress admin URL without scheme",
     "Go to mystore.net/wp-login.php to reset.",
     ["mystore.net/wp-login"]),
    ("www URL",
     "I found it at www.mywebsite.com/account/settings",
     ["www.mywebsite.com"]),
    ("International phone",
     "Call me at +1-800-555-1234 any time.",
     ["+1-800-555-1234"]),
    ("Password disclosure phrase",
     "I set my password is SuperSecret99! but now I can't log in.",
     ["SuperSecret99!"]),
    ("Bearer token",
     "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def",
     ["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"]),
    ("Stripe-style API key",
     "I used sk-live_abc123XYZ456qrs789 for the Stripe integration.",
     ["sk-live_abc123XYZ456qrs789"]),
    ("OTP disclosure",
     "The verification code is 847392 but it says expired.",
     ["847392"]),
    ("Person name in greeting",
     "Hi, my name is John Smith and I need help with my account.",
     ["John Smith"]),
    ("Message with multiple entity types",
     "Hi I'm Alex (alex@demo.net), my site demo.net/wp-admin is down. "
     "My account acct_99201 and my server 203.0.113.5 are unreachable. "
     "password: WrongPass1!",
     ["alex@demo.net", "demo.net/wp-admin", "203.0.113.5", "WrongPass1!"]),
]


RESIDUAL_RISK_PATTERNS: list[tuple[str, str]] = [
    ("email",            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("ipv4",             r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("url_https",        r"https?://[^\s]+"),
    ("url_www",          r"\bwww\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}"),
    ("wp_admin",         r"/wp-(?:admin|login|json|content)"),
    ("password_phrase",  r"(?i)password\s*(?:is|:|=|was)\s*\S+"),
    ("bearer_token",     r"(?i)Bearer\s+[A-Za-z0-9]{10,}"),
    ("api_key_prefix",   r"\b(?:sk|pk|rk|whsec|xoxb)[-_][A-Za-z0-9]{10,}"),
]


def run_synthetic_tests() -> tuple[int, int]:
    passed, failed = 0, 0
    failures: list[str] = []

    print("\n" + "=" * 60)
    print("SYNTHETIC DE-IDENTIFICATION TESTS")
    print("=" * 60)

    for description, input_text, must_not_appear in SYNTHETIC_CASES:
        output_text = deidentify_text(input_text, run_ner=True)
        case_passed = all(f.lower() not in output_text.lower() for f in must_not_appear)

        if case_passed:
            passed += 1
            print(f"  PASS  {description}")
        else:
            failed += 1
            print(f"  FAIL  {description}")
            for forbidden in must_not_appear:
                if forbidden.lower() in output_text.lower():
                    failures.append(
                        f"    '{description}' still contains: '{forbidden}'\n"
                        f"    Input : {input_text[:120]}\n"
                        f"    Output: {output_text[:120]}"
                    )

    print()
    if failures:
        print("FAILURE DETAILS:")
        for f in failures:
            print(f)
        print()

    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60 + "\n")
    return passed, failed


def run_bq_scan() -> None:
    """Scan chatbot_messages_deidentified for residual PII patterns and print row counts."""
    try:
        from google.cloud import bigquery
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from gcp_config import PROJECT_ID, bq_table
    except ImportError as e:
        log.error("BigQuery scan requires google-cloud-bigquery: %s", e)
        return

    client = bigquery.Client(project=PROJECT_ID)
    table  = bq_table("chatbot_messages_deidentified")

    print("\n" + "=" * 60)
    print("BIGQUERY RESIDUAL RISK SCAN")
    print(f"Table: {table}")
    print("=" * 60)

    for risk_type, pattern in RESIDUAL_RISK_PATTERNS:
        query = f"""
            SELECT COUNT(*) AS hits
            FROM `{table}`
            WHERE REGEXP_CONTAINS(content, r'{pattern}')
        """
        try:
            result = client.query(query).result()
            hits = list(result)[0]["hits"]
            status = "CLEAN" if hits == 0 else f"REVIEW NEEDED — {hits} rows"
            print(f"  {risk_type:20s}: {status}")
        except Exception as e:
            print(f"  {risk_type:20s}: ERROR — {e}")

    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate de-identification pipeline."
    )
    parser.add_argument(
        "--bq-scan", action="store_true", default=False,
        help="Also run a residual risk scan against the live BigQuery table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passed, failed = run_synthetic_tests()

    if args.bq_scan:
        run_bq_scan()

    if failed > 0:
        print(f"PIPELINE VALIDATION FAILED: {failed} test(s) did not pass.")
        sys.exit(1)
    else:
        print("PIPELINE VALIDATION PASSED: All synthetic tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
