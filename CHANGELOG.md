# Changelog

All notable changes will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases will use semantic versioning once the first public version is tagged.

## [Unreleased]

### Added

- Initial public, MIT-licensed source distribution.
- Contribution, security, support, and community policies.
- Runtime-generated TLS test certificates; no private-key fixtures are committed.

### Security

- The bridge hello between the realtime host and the Hermes companion now authenticates both
  sides with HMAC proofs over fresh nonces, and the token no longer crosses the wire. The
  `mutual_auth` capability is required, so a host and a companion upgrade together.
