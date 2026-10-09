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
  sides with HMAC proofs over fresh nonces, and the token no longer crosses the wire. Before
  both sides have proved the token, the companion discloses nothing but a nonce and its proof;
  the negotiated capabilities and the runtime attestation arrive only in the authenticated
  final acceptance. The `mutual_auth` capability is required, so a host and a companion
  upgrade together.
