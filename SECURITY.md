# Security Policy

## Supported Versions

Only the latest commit on the `main` branch receives security fixes. No backports are made to older commits, tags, or branches.

| Version / Branch | Supported |
|------------------|-----------|
| `main` (latest)  | Yes       |
| Any tagged release older than current `main` | No |
| Feature branches | No        |

If you are running an older version, upgrade to the current `main` before reporting an issue.

---

## Reporting a Vulnerability

**Do not open a public GitHub Issue to report a security vulnerability.** Public disclosure before a patch is available puts all users at risk.

### How to Report

Use **GitHub Private Security Advisories**:

1. Go to the repository on GitHub.
2. Click **Settings** → **Security** → **Advisories**.
3. Click **New draft security advisory**.
4. Fill in the report and submit.

Alternatively, if you do not have a GitHub account, email the maintainer directly using the contact listed in `pyproject.toml`. Encrypt the message with PGP if the content is sensitive.

### What to Include

A useful vulnerability report includes:

- **Description** — a clear explanation of the vulnerability, what it allows an attacker to do, and under what conditions
- **Reproduction steps** — a minimal sequence of commands, HTTP requests, or code that demonstrates the issue
- **Affected component** — e.g., RAG server API, dependency, configuration, data handling
- **Severity assessment** — your estimate of impact (critical / high / medium / low) and your reasoning
- **Suggested fix** (optional) — if you have a patch or mitigation in mind

The more detail you provide, the faster a fix can be produced.

---

## Response Timeline

| Severity | Acknowledgement | Target Patch |
|----------|----------------|--------------|
| Critical | Within 3 business days | Within 14 calendar days |
| High | Within 3 business days | Within 14 calendar days |
| Medium | Within 3 business days | Within 30 calendar days |
| Low / Informational | Within 3 business days | Best effort |

These timelines are targets, not guarantees. Complex vulnerabilities involving third-party dependencies may take longer. We will keep you informed of progress throughout the process.

---

## Scope

The following are **in scope** for security reports:

### RAG Server API
- **Authentication bypass** — requests that succeed without a valid `API_KEY` when one is configured
- **Prompt injection / jailbreak via API input** — crafted inputs that extract the system prompt, override model behavior, or exfiltrate data
- **Denial of Service** — inputs or request patterns that cause the server to crash, hang, or consume unbounded resources
- **Path traversal or file read** — endpoints that could be abused to read files outside the intended data directory

### Dependencies
- **CVEs in direct or transitive dependencies** — critical or high-severity CVEs in packages declared in `pyproject.toml` or `requirements*.txt`

### Data Exposure
- **Unintended data leakage** — scenarios where user query history, system prompts, or RAG context are exposed to unauthorized callers

---

## Out of Scope

The following are **not** considered security vulnerabilities for this project:

- **Rate limiting the Ollama model itself** — Ollama is a local inference server; rate limiting is outside the scope of this project. Run a reverse proxy if you need rate limiting.
- **Bible data being incorrect or incomplete** — translation accuracy and textual criticism are theological and editorial matters, not security issues.
- **Self-XSS** — attacks that require the user to run arbitrary code in their own browser context.
- **Social engineering** — phishing or deception attacks targeting individual users.
- **Issues only reproducible on unsupported configurations** — e.g., running on a branch other than `main` or with a modified codebase.

---

## Disclosure Policy

This project follows **coordinated disclosure**:

1. You report the vulnerability privately.
2. The maintainer investigates and develops a patch.
3. A fix is committed and released (or documented as a configuration change).
4. A public security advisory is published after the patch is available.

We ask that you:
- Allow reasonable time for a fix before any public disclosure.
- Not exploit the vulnerability beyond what is necessary to demonstrate it.
- Not access, modify, or delete user data during testing.

We will credit you by name (or handle) in the advisory unless you prefer to remain anonymous.

---

## Security Model Notes

### API Key Authentication

The RAG server supports optional API key authentication via the `API_KEY` environment variable.

- **Default (no `API_KEY` set):** The server accepts all requests without authentication. This is intentional for **localhost development** where the server is not exposed to external networks.
- **Production:** Always set `API_KEY` to a strong random value before exposing the server to any network, even a private LAN.

### Production Deployment Checklist

If you are deploying the Bible AI Assistant in any environment beyond your local machine:

1. **Set `API_KEY`** — generate a strong key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
2. **Run behind a reverse proxy** (e.g., nginx, Caddy) that terminates TLS. Never expose the RAG server directly on a public port over plain HTTP.
3. **Enable TLS** — obtain a certificate via Let's Encrypt or your organization's PKI.
4. **Restrict network access** — use firewall rules to allow only the UI or authorized clients to reach the RAG server port.
5. **Do not store sensitive personal data** — the system is designed for Bible study. Do not feed it PII, medical records, or confidential documents.
6. **Keep dependencies updated** — run `pip list --outdated` regularly and apply security patches promptly.

Failing to follow these steps in a networked deployment is a configuration issue, not a vulnerability in the project itself, but we are happy to advise.
