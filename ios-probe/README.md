# Hermes iPhone VPIO feasibility probe

This is a deliberately small, disposable physical-device probe. It answers one question before production native protocol work begins: can LiveKit Swift **2.15.3** keep microphone capture and remote assistant playback in one Apple Voice Processing I/O graph with platform echo cancellation active, AGC inactive, and no WebRTC software-processing stack?

It does **not** contain a Hermes host, token, signing identity, device identifier, or API secret. The LiveKit URL, short-lived participant token, and expected assistant identity are pasted into memory at runtime and are never exported.

## Required build lane

- macOS with Xcode **16.3 (16E140)** (bundled Swift 6.1 toolchain; this probe selects
  Swift language mode 6.0)
- XcodeGen **2.42.0**
- An Apple development team authorized to install on the target iPhone
- iOS 17 or newer

```bash
cd ios-probe
xcodegen generate --spec project.yml
xcodebuild -resolvePackageDependencies \
  -project HermesVPIOProbe.xcodeproj \
  -scheme HermesVPIOProbe
open HermesVPIOProbe.xcodeproj
```

In Xcode, set a private unique bundle identifier and select the owner's development team. Do not commit those signing changes. Build directly to the target iPhone; no TestFlight or distribution upload is required.

## Bounded probe procedure

1. Issue a short-lived LiveKit participant credential whose grants allow microphone publication and assistant subscription in one test room.
2. Paste the `wss://` URL, token, and exact expected assistant identity; retain `microphone-1`, and leave **Request coupled platform noise suppression** off. Auto-subscription is disabled: the probe subscribes only to one microphone-source audio publication owned by that exact assistant identity.
3. Tap **Connect and publish**. The pre-publish request should normally report `stored`; the post-publish request must report `applied`.
4. Capture speaker and receiver preference states. Every ordinary route/interruption notification immediately mutes capture and latches a blocked state; use **Reapply processing request** to unmute, require a fresh `.applied`, and revalidate the complete live state. Toggle the NS request on and reapply. On Apple's coupled topology, the supported state may require platform NS active with platform EC.
5. Exercise built-in speaker, receiver, wired headset, Bluetooth HFP, route removal, and interruption. Media-services loss/reset disconnects instead of attempting recovery. Unsupported A2DP/AirPlay behavior is diagnostic only in this probe; production enforcement comes later.
6. Export sanitized evidence. Verify it contains no URL, token, host, participant identity, track SID, device name, or transcript.
7. Run far-end counting through the normal assistant path while remaining silent, then speak `Stop` during a second count. Record acoustic success separately from the processing-state JSON.

## Go/no-go

Proceed to Phase 0 only if a supported request yields all of the following during live capture and playback:

- VPIO enabled and not bypassed;
- platform echo cancellation active and effective;
- WebRTC software echo cancellation and noise suppression inactive;
- AGC inactive in both engine and platform state;
- one microphone track with the canonical publication name;
- built-in-speaker far-end speech does not create an admitted user turn;
- genuine `Stop` remains audible and interrupts promptly.

A state report without the acoustic checks is not a pass.
