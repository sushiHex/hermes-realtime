# Security Policy

## Supported versions

Until the first tagged release, security fixes target the current `main` branch. After releases begin, the latest released minor line and current `main` will receive security fixes when practical.

This project is experimental alpha software. It has not received an independent security audit and should not be exposed directly to an untrusted network.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** form on the repository's Security tab. This creates a private security advisory visible to the maintainer.

Please include:

- affected version or commit;
- the boundary or component involved;
- minimal reproduction steps;
- the security impact;
- any suggested mitigation.

Do not include live credentials, bearer material, private audio, transcripts, personal data, or unrelated logs. Replace sensitive values with `[REDACTED]`.

Do not open a public issue for an undisclosed vulnerability. If GitHub private vulnerability reporting is unavailable, contact the repository owner through the private contact method listed on their GitHub profile.

## Response process

The maintainer will acknowledge a usable report when available, reproduce it against an exact commit, and coordinate disclosure after a fix or documented mitigation exists. No fixed response-time or bounty commitment is currently offered.

## Security boundaries

Hermes Realtime:

- runs third-party provider and plugin code in process;
- uses short-lived LiveKit and browser credentials but is not a sandbox;
- assumes loopback runs on a trusted single-user workstation;
- can dispatch Hermes work that mutates local or remote state;
- treats external source text as untrusted evidence, not instructions.

Security reports should distinguish defects in this repository from issues in Hermes Agent, LiveKit, browsers, operating systems, or configured providers.
