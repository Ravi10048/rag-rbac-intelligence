"""Generate the synthetic enterprise dataset.

We invent a self-contained dataset: a small-but-realistic corpus that exercises
every capability of the system:

    * Three PDFs spanning different departments and sensitivity levels.
    * One CSV with employee records (the classic "who can see salary?" test).
    * One JSON file of security audit events (sensitive operational data).
    * Two policy files declaring sensitivity per document + per-user roles.

Run once with `python scripts/generate_data.py` - re-running overwrites.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

# Allow running this script directly without `pip install -e .` -- we just
# add the project root to sys.path so `src.config` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from src import config


# Deterministic output -- the demo should produce the same answers on every
# fresh checkout. Faker + random share this seed.
random.seed(42)


# ---------------------------------------------------------------------------
# PDF writers
# ---------------------------------------------------------------------------
def write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    """Render a multi-section PDF using ReportLab's high-level Platypus API.

    `sections` is a list of (heading, body_text) pairs. We use Platypus
    rather than raw canvas drawing because it handles pagination + word
    wrapping for us automatically.
    """
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    flow = [Paragraph(title, styles["Title"]), Spacer(1, 18)]
    for heading, body in sections:
        flow.append(Paragraph(heading, styles["Heading2"]))
        flow.append(Spacer(1, 6))
        # Replace newlines with HTML <br/> so paragraphs render correctly.
        flow.append(Paragraph(body.replace("\n", "<br/>"), styles["BodyText"]))
        flow.append(Spacer(1, 12))
    doc.build(flow)


def generate_pdfs() -> None:
    """Build the three demonstration PDFs.

    Why these three?
        * HR handbook  -> "internal" sensitivity, broadly readable. Covers
          the common-knowledge case (everyone can ask about parental leave).
        * Finance Q4   -> "confidential", only finance + executives. Covers
          the "departmentally siloed" case.
        * Eng security -> "restricted", only engineering security + execs.
          Covers the "most-sensitive incident response" case.
    """
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # --- HR handbook ----------------------------------------------------
    write_pdf(
        config.DOCS_DIR / "hr_employee_handbook.pdf",
        "Acme Corp - Employee Handbook (2026)",
        [
            ("Code of Conduct",
             "All employees are expected to act with integrity and respect. "
             "Harassment of any kind is grounds for immediate dismissal. "
             "Conflicts of interest must be disclosed to the HR department "
             "within 14 days of arising."),
            ("Parental Leave Policy",
             "Acme Corp offers 16 weeks of fully-paid parental leave to "
             "all primary caregivers regardless of gender, and 6 weeks of "
             "fully-paid leave to secondary caregivers. Leave may be taken "
             "any time within the first 12 months following birth or adoption."),
            ("Remote Work Policy",
             "Employees may work remotely up to 3 days per week with manager "
             "approval. Fully remote arrangements require VP-level sign-off "
             "and are reviewed annually."),
            ("Performance Reviews",
             "Performance reviews occur every six months. Compensation "
             "adjustments are made annually each January based on the "
             "preceding two review cycles."),
        ],
    )

    # --- Finance Q4 report ---------------------------------------------
    write_pdf(
        config.DOCS_DIR / "finance_q4_2025_report.pdf",
        "Acme Corp - Q4 2025 Financial Report (CONFIDENTIAL)",
        [
            ("Executive Summary",
             "Acme Corp closed Q4 2025 with revenue of $48.2M, up 22% "
             "year-over-year. Gross margin improved to 64%, driven primarily "
             "by efficiency gains in the cloud infrastructure division."),
            ("Revenue Breakdown",
             "Cloud Services contributed $28.1M (58%), Professional Services "
             "$11.6M (24%), and Hardware Sales $8.5M (18%). The Enterprise "
             "segment grew 35% YoY while SMB held flat at 4% growth."),
            ("Operating Expenses",
             "Total opex was $29.4M, of which payroll accounted for $19.8M. "
             "Marketing spend was reduced 12% compared to Q3 as we shifted "
             "to organic-led growth motions."),
            ("Forward Guidance",
             "Q1 2026 revenue is forecast at $51-53M with continued margin "
             "expansion. The board has approved a $5M strategic investment "
             "in the new AI Platform team announced internally on Dec 4."),
        ],
    )

    # --- Engineering security audit (restricted) -----------------------
    write_pdf(
        config.DOCS_DIR / "engineering_security_audit.pdf",
        "Engineering Security Audit - Q4 2025 (RESTRICTED)",
        [
            ("Scope",
             "This audit covers the production infrastructure of the Cloud "
             "Services platform, including the payment processing subsystem "
             "and the customer data warehouse. The audit was conducted "
             "between Oct 15 and Nov 30, 2025."),
            ("Critical Findings",
             "Finding C-1: The payment processing API logs full credit card "
             "numbers under a debug flag that was left enabled in two "
             "staging clusters. Remediated on Nov 8 (ticket SEC-2241). "
             "Finding C-2: An IAM role granting S3 write access to the "
             "customer warehouse was over-privileged. Scope reduced Nov 22."),
            ("Recommendations",
             "1. Adopt structured log redaction at the SDK layer rather than "
             "relying on grep-based scrubbing. 2. Quarterly rotation of all "
             "IAM credentials used by CI. 3. Block deployment of debug=true "
             "to any environment via OPA policy."),
            ("Incident Response Drill",
             "A tabletop exercise simulating a ransomware event was held "
             "Nov 18. The response team achieved containment within 47 "
             "minutes, exceeding the 2-hour target. Detailed timeline is "
             "tracked in the security incident log."),
        ],
    )


# ---------------------------------------------------------------------------
# Structured data: CSV employees + JSON security log
# ---------------------------------------------------------------------------
def generate_employee_csv() -> None:
    """Produce a small employee directory with salary information.

    Salary is the prototypical "RBAC litmus test" - HR and execs should
    see it, but a peer engineer asking "what does Bob make?" must be
    denied even if the document is semantically retrievable.
    """
    config.DB_DIR.mkdir(parents=True, exist_ok=True)

    employees = [
        # (id, name, department, role, hire_date, salary_usd)
        ("E-1001", "Bob Singh",       "engineering", "Senior Engineer",   "2021-03-14", 165_000),
        ("E-1002", "Alice Chen",      "hr",          "HR Manager",        "2019-08-02", 138_000),
        ("E-1003", "Carol Diaz",      "finance",     "Finance Analyst",   "2022-01-19", 122_000),
        ("E-1004", "David Patel",     "executive",   "CEO",               "2018-05-10", 410_000),
        ("E-1005", "Eli Roberts",     "engineering", "Staff Engineer",    "2020-11-23", 198_000),
        ("E-1006", "Farah Khan",      "engineering", "Engineering Manager","2019-02-04", 215_000),
        ("E-1007", "Grace Liu",       "finance",     "Controller",        "2017-06-30", 175_000),
        ("E-1008", "Henry Adams",     "hr",          "HR Business Partner","2023-04-11", 98_000),
    ]

    out = config.DB_DIR / "employees.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["employee_id", "name", "department", "role",
                    "hire_date", "salary_usd"])
        w.writerows(employees)


def generate_security_log() -> None:
    """Synthetic SIEM-style audit log.

    Storing it as a JSON array (vs. JSONL) keeps the ingestion path trivial
    and mirrors what many enterprise systems export when you "download all
    events". Each event becomes its own chunk during ingestion.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    events = [
        {
            "event_id": "EVT-9001",
            "timestamp": "2025-11-08T02:14:33Z",
            "severity": "high",
            "category": "data_exposure",
            "summary": "Debug flag enabled in payment-staging-2 caused full "
                       "PAN numbers to be written to application logs.",
            "affected_systems": ["payment-staging-2"],
            "remediation": "Flag disabled; logs purged; ticket SEC-2241 filed.",
        },
        {
            "event_id": "EVT-9002",
            "timestamp": "2025-11-12T18:42:01Z",
            "severity": "medium",
            "category": "iam",
            "summary": "Overly permissive IAM role 'warehouse-writer' allowed "
                       "S3 PutObject across the entire customer-data bucket.",
            "affected_systems": ["customer-warehouse"],
            "remediation": "Scope reduced to specific prefixes on 2025-11-22.",
        },
        {
            "event_id": "EVT-9003",
            "timestamp": "2025-11-18T15:00:00Z",
            "severity": "informational",
            "category": "drill",
            "summary": "Quarterly tabletop ransomware drill executed. "
                       "Containment achieved in 47 minutes.",
            "affected_systems": [],
            "remediation": "Post-mortem stored in SEC-DRILLS folder.",
        },
        {
            "event_id": "EVT-9004",
            "timestamp": "2025-12-02T09:21:11Z",
            "severity": "low",
            "category": "auth",
            "summary": "Three failed SSH login attempts on bastion host "
                       "from IP 203.0.113.42 followed by automatic block.",
            "affected_systems": ["bastion-prod-1"],
            "remediation": "IP added to deny list; no further action required.",
        },
    ]

    (config.LOGS_DIR / "security_audit_log.json").write_text(
        json.dumps(events, indent=2)
    )


# ---------------------------------------------------------------------------
# RBAC: document sensitivity, users, and policy
# ---------------------------------------------------------------------------
def generate_document_metadata() -> None:
    """Tag every source document with its owning department + sensitivity.

    Ingestion reads this file to attach metadata to each chunk. Doing it
    in a separate JSON (rather than guessing from filenames) means
    sensitivity can be changed without touching code.
    """
    metadata = {
        "hr_employee_handbook.pdf": {
            "doc_id": "DOC-HR-HANDBOOK",
            "title": "Acme Employee Handbook 2026",
            "department": "hr",
            "sensitivity": "internal",
            "tags": ["handbook", "policy", "benefits", "remote", "leave"],
        },
        "finance_q4_2025_report.pdf": {
            "doc_id": "DOC-FIN-Q4",
            "title": "Q4 2025 Financial Report",
            "department": "finance",
            "sensitivity": "confidential",
            "tags": ["revenue", "earnings", "quarterly", "forecast"],
        },
        "engineering_security_audit.pdf": {
            "doc_id": "DOC-SEC-AUDIT",
            "title": "Engineering Security Audit Q4 2025",
            "department": "engineering",
            "sensitivity": "restricted",
            "tags": ["security", "audit", "iam", "incident"],
        },
        "employees.csv": {
            "doc_id": "DB-EMPLOYEES",
            "title": "Employee Directory",
            "department": "hr",
            "sensitivity": "confidential",
            "tags": ["employees", "salary", "directory"],
        },
        "security_audit_log.json": {
            "doc_id": "LOG-SEC-AUDIT",
            "title": "Security Audit Event Log",
            "department": "engineering",
            "sensitivity": "restricted",
            "tags": ["siem", "security", "incident", "log"],
        },
    }
    config.POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    (config.POLICIES_DIR / "documents_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def generate_users() -> None:
    """The four demo personas.

    Picked specifically to exercise different access patterns:
        Alice (HR)        - reads employee CSV (incl. salary) + HR handbook;
                            denied finance / eng security.
        Bob (Engineer)    - reads HR handbook + eng security log;
                            denied salary CSV + finance.
        Carol (Finance)   - reads HR handbook + finance Q4 report;
                            denied salary CSV + eng security.
        David (CEO)       - reads everything (executive clearance).
    """
    users = {
        "alice@acme.com": {
            "user_id": "U-001",
            "name": "Alice Chen",
            "department": "hr",
            "role": "hr_manager",
            "clearance": "confidential",
            "accessible_departments": ["hr"],
        },
        "bob@acme.com": {
            "user_id": "U-002",
            "name": "Bob Singh",
            "department": "engineering",
            "role": "engineer",
            "clearance": "restricted",
            "accessible_departments": ["engineering", "hr"],
        },
        "carol@acme.com": {
            "user_id": "U-003",
            "name": "Carol Diaz",
            "department": "finance",
            "role": "finance_analyst",
            "clearance": "confidential",
            "accessible_departments": ["finance", "hr"],
        },
        "david@acme.com": {
            "user_id": "U-004",
            "name": "David Patel",
            "department": "executive",
            "role": "ceo",
            "clearance": "restricted",
            # Empty list is treated as "all departments" by the RBAC engine
            # -- this is the CEO/admin escape hatch.
            "accessible_departments": [],
        },
    }
    config.POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    (config.POLICIES_DIR / "user_roles.json").write_text(
        json.dumps(users, indent=2)
    )


def generate_access_policies() -> None:
    """Declarative policy rules.

    The RBAC engine evaluates these in order. The first thing to know is
    that *clearance* gates sensitivity (you need at-least the document's
    sensitivity level), and *accessible_departments* gates which silos
    the user can touch. Special roles (`ceo`, `ciso`) bypass dept filters.
    """
    policies = {
        "default": {
            "description": "Base rule - users may only access documents "
                           "matching their cleared sensitivity AND assigned "
                           "departments.",
            "min_clearance": "public",
        },
        "salary_data": {
            "description": "Salary information requires HR or executive role "
                           "regardless of clearance level (need-to-know).",
            "allowed_roles": ["hr_manager", "hr_business_partner",
                              "ceo", "cfo"],
        },
        "financial_reports": {
            "description": "Quarterly / strategic finance docs are restricted "
                           "to the finance department and executives.",
            "allowed_departments": ["finance", "executive"],
        },
        "security_incidents": {
            "description": "Security incident details may only be read by "
                           "the engineering department or executives.",
            "allowed_departments": ["engineering", "executive"],
        },
    }
    config.POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    (config.POLICIES_DIR / "access_policies.json").write_text(
        json.dumps(policies, indent=2)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating synthetic enterprise dataset...")
    generate_pdfs()
    print(f"  PDFs        -> {config.DOCS_DIR}")
    generate_employee_csv()
    print(f"  CSV         -> {config.DB_DIR}")
    generate_security_log()
    print(f"  JSON log    -> {config.LOGS_DIR}")
    generate_document_metadata()
    generate_users()
    generate_access_policies()
    print(f"  Policies    -> {config.POLICIES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
