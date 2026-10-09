# Proposed Tailscale remote profile

This is the design for [issue #214](https://github.com/sushiHex/hermes-realtime/issues/214),
not an installed or qualified profile. No setup command or phone acceptance result exists yet.
The [local LiveKit profile](local-livekit.md) remains the desktop development baseline.

## Boundary and inputs

The remote LiveKit executable has one per-user path outside every checkout:
`%LOCALAPPDATA%\hermes-realtime\tools\livekit-1.13.4-tailnet\livekit-server.exe`.
`scripts/local_livekit.py` will resolve, install, and verify that path using the **same**
version, release archive, archive SHA-256, bounded archive inspection, and executable
SHA-256 as the existing shared resolver. It will verify the remote copy immediately
before launch. The existing `shared_path()`, `path`, and `serve` contracts remain the
loopback profile; the remote CLI will have explicit, separately named operations.
There must be no checkout-relative fallback and no unverified executable override.

The distinct path gives Windows Firewall a separate program identity and keeps
the two launch configurations apart, even though the bytes are identical. The
loopback path has inbound Block rules created when its Windows Firewall prompt was
cancelled. An ordinary Allow rule cannot override a matching Block rule under
[Windows Firewall rule precedence](https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/troubleshoot-windows-firewall-with-advanced-security-guidance).
The remote setup must not edit, disable, or remove those Block rules. It must refuse
to continue if a matching Block rule covers the remote path and intended traffic.

Owner-supplied inputs are the host's current Tailscale IPv4 address and adapter,
the adapter's **observed** Windows network profile, a trusted LiveKit `wss://`
endpoint with TLS termination, a trusted HTTPS certificate and canonical origin for
the browser host, and strong non-development LiveKit credentials kept in the
restricted runtime environment. The setup must derive the local IPv4 address from
the adapter and confirm it equals the Tailscale CLI's IPv4 address; it must not
infer the profile from an assumed `Private` or `Public` default. The Tailscale
address range `100.64.0.0/10` is documented by
[Tailscale](https://tailscale.com/docs/reference/reserved-ip-addresses). It is an
address scope, **not** peer identity authorization: another service can use the
same CGNAT space, and tailnet policy and the exact peer check remain necessary.
This initial profile is IPv4 only; IPv6 needs a separately verified rule and
candidate design.

The adapter's current network category is read by interface index through
[`Get-NetConnectionProfile`](https://learn.microsoft.com/en-us/powershell/module/netconnection/get-netconnectionprofile),
then checked again at firewall verification time.

## LiveKit configuration

The remote launch uses a private, generated configuration for pinned LiveKit 1.13.4,
without `--dev` or the public development keys. The intended network fields are:

```yaml
port: 7880
rtc:
  use_external_ip: false
  node_ip: <verified-host-tailnet-IPv4>
  tcp_port: 7881
  udp_port: 7882
  ips:
    includes:
      - <verified-host-tailnet-IPv4>/32
```

The placeholders are replaced locally, never committed with a real address. No
`rtc.port_range_start` or `rtc.port_range_end` is set when `rtc.udp_port` is used.
The pinned [LiveKit sample configuration](https://github.com/livekit/livekit/blob/v1.13.4/config-sample.yaml)
defines these fields, and the [port reference](https://docs.livekit.io/transport/self-hosting/ports-firewall/)
distinguishes signaling TCP 7880, ICE/TCP 7881, and UDP mux 7882. Keep signaling
bound to `127.0.0.1` via the server's `--bind` argument; a trusted TLS terminator
on the Tailscale address forwards `wss://` to that loopback listener. LiveKit's
[deployment guidance](https://docs.livekit.io/transport/self-hosting/deployment/)
requires trusted TLS for browser signaling. Its TLS terminator and browser HTTPS
listener need their own narrowly scoped ingress rules; neither justifies opening
LiveKit TCP 7880 to the tailnet. The generated configuration and runtime environment
must be access restricted, and credentials must not enter arguments, logs, tests,
or this repository. A running loopback LiveKit instance already owns TCP 7880 and
possibly RTC ports; setup must detect that conflict and refuse a simultaneous launch.

The intended candidate is the verified Tailscale IPv4 address on the fixed RTC
ports. This remains an acceptance condition, not an assumption from the YAML:
inspect the actual advertised and selected candidate pairs during the real-device
test, and reject a LAN or public candidate that bypasses the intended tailnet path.

## Firewall ownership

The proposed owner-run setup creates exactly two inbound LiveKit Allow rules in
the local persistent policy store. Their immutable rule `Name` values are
`HermesRealtime.Tailnet.LiveKit.TCP.v1` and
`HermesRealtime.Tailnet.LiveKit.UDP.v1`. `DisplayName` is explanatory only and is
never used to select a rule for update or deletion. Each rule must have:

| Field | TCP rule | UDP rule |
| --- | --- | --- |
| Direction, action, enabled | Inbound, Allow, True | Inbound, Allow, True |
| Program | exact verified remote executable path | exact verified remote executable path |
| Protocol, local port | TCP, 7881 | UDP, 7882 |
| Remote address | `100.64.0.0/10` | `100.64.0.0/10` |
| Local address | exact verified host Tailscale IPv4 | exact verified host Tailscale IPv4 |
| Interface alias | exact discovered Tailscale adapter alias | exact discovered Tailscale adapter alias |
| Profile | current observed profile of that adapter | current observed profile of that adapter |

These filters are supported by
[Microsoft's `New-NetFirewallRule` reference](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule).
Do not use `Any` for the listed program, local/remote address, interface,
profile, or local port scopes; do not use all profiles, a broad program rule,
a port range, or an unscoped IPv6 companion rule. The Tailscale adapter can be
on a profile other than the one expected by an earlier hand-made rule. The setup checks the
effective profile and local policy merge before declaring the rules active.

The implementation will read each rule by its complete `Name` and require one
exact name match. It will compare the rule's direction, action, enabled state,
profile, policy store, plus its application, port, address, and interface filters
to the desired values by equality. A same-name rule with unexpected fields is
a collision, not permission to overwrite it. An exact existing pair is a no-op.
If the adapter address or profile changes, reconciliation may replace only a
rule whose previous complete fields match the recorded owned state; otherwise it
fails closed and gives a read-only diagnosis. No selector uses a `DisplayName`,
program suffix, wildcard `Name`, or global LiveKit port match for removal.

Read-back must use the effective firewall policy as well as the local policy
store: confirm both owned Allow rules are active on the adapter's actual profile,
local firewall rules are merged there, and no enabled overlapping Block rule
defeats them. This includes Block rules with the remote program path and
program-independent Block rules on the intended address, interface, profile,
protocol and port. Ambiguous policy, uninspectable filters, or a disabled firewall
is a failed verification. Read-back proves configured policy only; a remote peer
probe is still required to prove packet delivery. The operator should also review
other effective Allow rules. If a matching rule exposes either RTC listener beyond
the intended address, interface, profile, program, protocol, or local port scope,
verification must refuse a scoped-ready verdict and report the conflicting rule
for owner remediation. It must not silently delete unrelated rules.

[`Get-NetFirewallRule`](https://learn.microsoft.com/en-us/powershell/module/netsecurity/get-netfirewallrule)
documents the associated application, port, address, and interface filter objects;
[`Get-NetFirewallProfile -PolicyStore ActiveStore`](https://learn.microsoft.com/en-us/powershell/module/netsecurity/get-netfirewallprofile)
reads the active profile settings. A rule's own displayed fields alone are not
enough to verify the match.

## Owner-run lifecycle

The planned explicit setup command first verifies the pinned remote binary,
Tailscale address/adapter/profile, secure signaling and credential prerequisites,
and absence of conflicting LiveKit listeners. It renders the private config,
reads firewall state, then adds the two narrowly scoped rules in an elevated
step. It reads back the effective rules and refuses conflicting Block or broad
Allow rules before launching its own remote LiveKit child. It then verifies the
child's actual listeners and address before reporting local setup readiness;
real-device candidate verification remains a separate acceptance step. Repeating
setup with the same inputs must recognize its existing child by verified process
ownership and exact rules as a no-op; it must never adopt or kill a listener it
did not start. A failed preflight must make no firewall changes.

On a partial failure after the first rule is created, the command stops its own
child if launched, then removes only rules it created in that invocation, by the
exact `Name` and full expected field match, and reports the remaining state. A
rollback command stops only its owned
remote process, removes only its exact named and verified owned rules, and
rechecks their absence. It leaves the loopback binary and its Block rules, other
firewall rules, certificate material, credentials, and Tailscale installation
untouched. Any unexpected same-name rule, failed removal, listener left alive,
or changed effective policy is reported as incomplete rather than success.
Owner review of a `-WhatIf` preview precedes applying elevated changes.

The browser side uses the existing remote host mode from
[the local LiveKit guide](local-livekit.md#remote-full-host-activation-seam):
`--remote --tailnet-launch`, a trusted HTTPS origin and TLS keypair, strong
`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`, and the exact allowed node StableID in
`HERMES_REALTIME_TAILNET_NODE_STABLE_ID`. The
[`TailnetPeerAuthorizer`](../src/hermes_realtime/client/tailnet.py) checks the
socket peer with `tailscale whois --json --proto tcp` and exact `Node.StableID`
equality for Connect. That browser authorization does not replace LiveKit
credentials, tailnet access policy, or the firewall rules. The owner must verify
the HTTPS listener's separate rule has equivalent address, interface, profile,
and peer scope before remote use.

## Acceptance evidence

### Implementation slices

Keep implementation under #214, after this design is reviewed, and build against
current main. The existing launcher remains authoritative until the integration
slice is qualified.

1. Extend `scripts/local_livekit.py` with an explicit remote profile that reuses
   its pinned archive and executable verification. Add a pure configuration
   renderer and desired firewall-rule specification. Test path separation, pin
   mismatch, exact ports/address restrictions, and refusal of development keys;
   this slice must neither launch a service nor edit Windows policy.
2. Add the explicit owner-run setup, inspection and rollback entry points.
   Keep policy observation separate from mutation, and serialize setup/rollback
   so two invocations cannot both claim the same rules or process. Exercise
   same-name collisions, stale adapter/profile information, effective Block and
   broader Allow rules, partial creation, process-launch failure, and incomplete
   rollback. Each new guard needs its own failing mutation. Synthetic policy
   fixtures do not establish that the machine's effective firewall is correct.
3. Integrate the verified remote profile with the existing launcher and browser
   authorization, then run the real-device procedure below. Bind its record to
   the candidate and actual device/software versions. Keep public evidence to
   outcomes, categories and counts, without credentials, addresses, device
   identities, transcript text or audio. A passing desktop or synthetic test
   cannot substitute for this remote transport acceptance.

The implementation must retain a bounded record of its exact owned resources
and make repeated setup and rollback safe. No service installer, automatic
startup, generalized firewall manager or additional remote-access mode is part
of these slices.

### Real-device procedure

From an authorized real device on the tailnet, with a candidate-specific record:

1. Confirm trusted HTTPS and `wss://` connections, and reject an unauthorized
   tailnet peer. Inspect the local listeners and the actual LiveKit ICE candidates;
   the selected media path must use the host Tailscale IPv4 on UDP 7882 or the
   scoped TCP 7881 fallback.
2. Connect, speak and hear a response; interrupt speech; disconnect and reconnect
   without duplicating a foreground turn. Record the actual outcomes and failures.
3. Probe the same ports from a non-tailnet device or interface and confirm denial.
   Re-read the rules after a network profile change and require setup to refuse
   stale scope until reconciled. Roll back and confirm only the two owned rules
   and remote process are gone while loopback Block rules remain.

This verifies one transport profile on the tested device and software versions.
It does not establish iPhone or WebKit support; that characterization belongs
to [issue #68](https://github.com/sushiHex/hermes-realtime/issues/68), outside
the initial desktop MVP's scope in
[issue #159](https://github.com/sushiHex/hermes-realtime/issues/159).
