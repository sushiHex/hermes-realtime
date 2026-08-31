@preconcurrency import AVFAudio
import Combine
import Foundation
@preconcurrency import LiveKit

struct ProbeEvidenceSample: Codable, Sendable {
    let timestamp: String
    let reason: String
    let sdkVersion: String
    let connected: Bool
    let roomConnectionState: String
    let processingVerdict: String
    let authorizedRemoteAudioSubscribed: Bool
    let requestedPlatformNoiseSuppression: Bool
    let outputPreference: String
    let applyOutcome: String
    let sessionCategory: String
    let sessionMode: String
    let routeInputs: [String]
    let routeOutputs: [String]
    let sampleRate: Double
    let ioBufferDuration: Double
    let echoCancellation: String
    let noiseSuppression: String
    let autoGainControl: String
    let highpassFilter: String
    let platformTopology: String
    let platformEchoCancellation: String
    let platformNoiseSuppression: String
    let platformAutoGainControl: String
    let voiceProcessingEnabled: String
    let voiceProcessingBypassed: String
    let voiceProcessingAGCEnabled: String
}

@MainActor
final class ProbeModel: ObservableObject {
    @Published var serverURL = ""
    @Published var token = ""
    @Published var expectedWorkerIdentity = ""
    @Published var trackName = "microphone-1"
    @Published var requestPlatformNoiseSuppression = false
    @Published private(set) var isConnected = false
    @Published private(set) var isBusy = false
    @Published private(set) var prefersSpeaker = true
    @Published private(set) var status = "Disconnected"
    @Published private(set) var samples: [ProbeEvidenceSample] = []
    @Published private(set) var latestSampleText = "No samples"
    @Published private(set) var exportURL: URL?

    private var room: Room?
    private var localTrack: LocalAudioTrack?
    private var publication: LocalTrackPublication?
    private var applyOutcome = "not-applied"
    private var samplingTask: Task<Void, Never>?
    private var observers: [NSObjectProtocol] = []
    private var selectedRemotePublication: ObjectIdentifier?
    private var remoteWaitStartedAt: Date?
    private var qualificationFailureLatched = false

    init() {
        prefersSpeaker = AudioManager.shared.audioSession.isSpeakerOutputPreferred
        installObservers()
        captureSample(reason: "launch")
    }

    func connect() async {
        guard !isBusy, !isConnected else { return }
        guard isValidCanonicalTrackName(trackName) else {
            status = "Blocked: track name must match microphone-N"
            return
        }
        guard let url = URL(string: serverURL), url.scheme?.lowercased() == "wss", url.host != nil else {
            status = "Blocked: enter a valid wss:// URL"
            return
        }
        guard !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            status = "Blocked: enter a short-lived token"
            return
        }
        let normalizedWorkerIdentity = expectedWorkerIdentity.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedWorkerIdentity.isEmpty else {
            status = "Blocked: enter the expected assistant identity"
            return
        }
        expectedWorkerIdentity = normalizedWorkerIdentity

        isBusy = true
        status = "Configuring"
        exportURL = nil
        let shortLivedToken = token
        defer {
            token = ""
            isBusy = false
        }

        do {
            AudioManager.shared.audioSession.isSpeakerOutputPreferred = prefersSpeaker
            AudioManager.shared.isVoiceProcessingAGCEnabled = false
            let captureOptions = makeCaptureOptions()
            let processingOptions = makeProcessingOptions()
            let track = LocalAudioTrack.createTrack(name: trackName, options: captureOptions)
            localTrack = track

            let prePublishResult = try track.setAudioProcessingOptions(processingOptions)
            applyOutcome = "pre-publish:\(describe(prePublishResult))"
            captureSample(reason: "pre-publish-processing")

            let roomOptions = RoomOptions(
                defaultAudioCaptureOptions: captureOptions,
                defaultAudioPublishOptions: AudioPublishOptions(name: trackName)
            )
            let newRoom = Room()
            room = newRoom
            status = "Connecting"
            try await newRoom.connect(
                url: url.absoluteString,
                token: shortLivedToken,
                connectOptions: ConnectOptions(autoSubscribe: false, enableMicrophone: false),
                roomOptions: roomOptions
            )

            status = "Publishing microphone"
            let published = try await newRoom.localParticipant.publish(
                audioTrack: track,
                options: AudioPublishOptions(name: trackName)
            )
            guard published.name == trackName else {
                throw ProbeError.publicationNameMismatch
            }
            publication = published

            let postPublishResult = try track.setAudioProcessingOptions(processingOptions)
            applyOutcome += ";post-publish:\(describe(postPublishResult))"
            guard case .applied = postPublishResult else {
                throw ProbeError.postPublishProcessingNotApplied
            }
            AudioManager.shared.isVoiceProcessingAGCEnabled = false
            let verdict = validateLiveState()
            guard verdict.accepted else {
                throw ProbeBlockedError(reason: verdict.message)
            }
            isConnected = true
            qualificationFailureLatched = false
            selectedRemotePublication = nil
            remoteWaitStartedAt = Date()
            status = "Live processing accepted; waiting for authorized assistant audio"
            captureSample(reason: "post-publish-processing")
            startPeriodicSampling()
        } catch {
            applyOutcome = describe(error)
            status = "Blocked: \(describe(error))"
            captureSample(reason: "connect-failed")
            await disconnectInternal(preserveStatus: true)
        }
    }

    func disconnect() async {
        guard !isBusy else { return }
        isBusy = true
        status = "Disconnecting"
        await disconnectInternal(preserveStatus: false)
        isBusy = false
    }

    func reapplyProcessing() async {
        guard let localTrack, isConnected else { return }
        do {
            try await localTrack.unmute()
            AudioManager.shared.isVoiceProcessingAGCEnabled = false
            let result = try localTrack.setAudioProcessingOptions(makeProcessingOptions())
            applyOutcome = "runtime:\(describe(result))"
            guard case .applied = result else {
                throw ProbeError.runtimeProcessingNotApplied
            }
            AudioManager.shared.isVoiceProcessingAGCEnabled = false
            let verdict = validateLiveState()
            guard verdict.accepted else {
                throw ProbeBlockedError(reason: verdict.message)
            }
            qualificationFailureLatched = false
            status = selectedRemotePublication == nil
                ? "Live processing accepted; waiting for authorized assistant audio"
                : "Ready for acoustic proof"
            captureSample(reason: "runtime-reapply")
        } catch {
            applyOutcome = describe(error)
            status = "Blocked: \(describe(error))"
            qualificationFailureLatched = true
            captureSample(reason: "runtime-reapply-failed")
            await disconnectInternal(preserveStatus: true)
        }
    }

    func toggleOutputPreference() async {
        if isConnected {
            qualificationFailureLatched = true
            status = "Blocked: output transition requires reapply"
            guard let localTrack else {
                status = "Blocked: microphone ownership was lost"
                await disconnectInternal(preserveStatus: true)
                return
            }
            do {
                try await localTrack.mute()
            } catch {
                status = "Blocked: mute failed during output transition"
                await disconnectInternal(preserveStatus: true)
                return
            }
        }
        prefersSpeaker.toggle()
        AudioManager.shared.audioSession.isSpeakerOutputPreferred = prefersSpeaker
        captureSample(reason: prefersSpeaker ? "speaker-preferred" : "receiver-preferred")
    }

    func captureSample(reason: String) {
        let session = AVAudioSession.sharedInstance()
        let engine = AudioManager.shared.audioProcessingState
        let platform = AudioManager.shared.platformVoiceProcessingState
        let route = session.currentRoute
        let verdict = validateLiveState()
        let sample = ProbeEvidenceSample(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            reason: reason,
            sdkVersion: "2.15.3",
            connected: roomIsConnected(),
            roomConnectionState: room.map { String(describing: $0.connectionState) } ?? "none",
            processingVerdict: qualificationFailureLatched ? "latched-failure:\(verdict.message)" : verdict.message,
            authorizedRemoteAudioSubscribed: authorizedRemoteAudioIsSubscribed(),
            requestedPlatformNoiseSuppression: requestPlatformNoiseSuppression,
            outputPreference: prefersSpeaker ? "speaker" : "receiver",
            applyOutcome: applyOutcome,
            sessionCategory: session.category.rawValue,
            sessionMode: session.mode.rawValue,
            routeInputs: route.inputs.map { $0.portType.rawValue }.sorted(),
            routeOutputs: route.outputs.map { $0.portType.rawValue }.sorted(),
            sampleRate: session.sampleRate,
            ioBufferDuration: session.ioBufferDuration,
            echoCancellation: describe(engine.echoCancellation),
            noiseSuppression: describe(engine.noiseSuppression),
            autoGainControl: describe(engine.autoGainControl),
            highpassFilter: describe(engine.highpassFilter),
            platformTopology: String(describing: platform.topology),
            platformEchoCancellation: describe(platform.echoCancellation),
            platformNoiseSuppression: describe(platform.noiseSuppression),
            platformAutoGainControl: describe(platform.autoGainControl),
            voiceProcessingEnabled: describe(platform.voiceProcessingEnabled),
            voiceProcessingBypassed: describe(platform.voiceProcessingBypassed),
            voiceProcessingAGCEnabled: describe(platform.voiceProcessingAGCEnabled)
        )
        samples.append(sample)
        if samples.count > 2_000 {
            samples.removeFirst(samples.count - 2_000)
        }
        latestSampleText = encode(sample) ?? "Encoding failed"
    }

    func exportEvidence() {
        guard let data = try? JSONEncoder.pretty.encode(samples) else {
            status = "Evidence export failed"
            return
        }
        do {
            let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let url = directory.appendingPathComponent("vpio-probe-evidence.json")
            try data.write(to: url, options: .atomic)
            exportURL = url
            status = "Sanitized evidence ready"
        } catch {
            status = "Evidence export failed"
        }
    }

    private func disconnectInternal(preserveStatus: Bool) async {
        samplingTask?.cancel()
        samplingTask = nil
        isConnected = false
        let ownedRoom = room
        if let ownedRoom {
            await ownedRoom.disconnect()
        }
        publication = nil
        localTrack = nil
        self.room = nil
        selectedRemotePublication = nil
        remoteWaitStartedAt = nil
        captureSample(reason: "disconnected")
        if !preserveStatus {
            status = "Disconnected"
        }
    }

    private func startPeriodicSampling() {
        samplingTask?.cancel()
        samplingTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled, let self else { return }
                guard self.roomIsConnected() else {
                    self.qualificationFailureLatched = true
                    self.status = "Blocked: room is not connected"
                    self.captureSample(reason: "room-not-connected")
                    await self.disconnectInternal(preserveStatus: true)
                    return
                }
                if self.qualificationFailureLatched {
                    self.captureSample(reason: "periodic-blocked")
                    continue
                }
                let verdict = self.validateLiveState()
                guard verdict.accepted else {
                    self.qualificationFailureLatched = true
                    self.status = "Blocked: \(verdict.message)"
                    self.captureSample(reason: "processing-state-invalid")
                    await self.disconnectInternal(preserveStatus: true)
                    return
                }
                do {
                    let ready = try await self.ensureAuthorizedRemoteAudio()
                    if ready {
                        self.status = "Ready for acoustic proof"
                    }
                } catch {
                    self.qualificationFailureLatched = true
                    self.status = "Blocked: \(self.describe(error))"
                    self.captureSample(reason: "remote-audio-invalid")
                    await self.disconnectInternal(preserveStatus: true)
                    return
                }
                self.captureSample(reason: "periodic")
            }
        }
    }

    private func installObservers() {
        let names: [(Notification.Name, String, Bool)] = [
            (AVAudioSession.routeChangeNotification, "route-change", false),
            (AVAudioSession.interruptionNotification, "interruption", false),
            (AVAudioSession.mediaServicesWereResetNotification, "media-services-reset", true),
            (AVAudioSession.mediaServicesWereLostNotification, "media-services-lost", true),
        ]
        for (name, reason, requiresDisconnect) in names {
            let observer = NotificationCenter.default.addObserver(
                forName: name,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.isConnected else {
                        self.captureSample(reason: reason)
                        return
                    }
                    self.qualificationFailureLatched = true
                    if requiresDisconnect {
                        self.status = "Blocked: media services require reconnect"
                        self.captureSample(reason: reason)
                        await self.disconnectInternal(preserveStatus: true)
                    } else {
                        self.status = "Blocked: audio transition requires processing reapply"
                        guard let localTrack = self.localTrack else {
                            self.status = "Blocked: microphone ownership was lost"
                            await self.disconnectInternal(preserveStatus: true)
                            return
                        }
                        do {
                            try await localTrack.mute()
                            self.captureSample(reason: reason)
                        } catch {
                            self.status = "Blocked: mute failed during audio transition"
                            self.captureSample(reason: "\(reason)-mute-failed")
                            await self.disconnectInternal(preserveStatus: true)
                        }
                    }
                }
            }
            observers.append(observer)
        }
    }

    private func makeCaptureOptions() -> AudioCaptureOptions {
        AudioCaptureOptions(
            echoCancellation: true,
            autoGainControl: false,
            noiseSuppression: requestPlatformNoiseSuppression,
            highpassFilter: false,
            typingNoiseDetection: false,
            echoCancellationMode: .platform,
            autoGainControlMode: .software,
            noiseSuppressionMode: .platform,
            highpassFilterMode: .automatic
        )
    }

    private func makeProcessingOptions() -> AudioProcessingOptions {
        AudioProcessingOptions(
            echoCancellation: true,
            autoGainControl: false,
            noiseSuppression: requestPlatformNoiseSuppression,
            highpassFilter: false,
            echoCancellationMode: .platform,
            autoGainControlMode: .software,
            noiseSuppressionMode: .platform,
            highpassFilterMode: .automatic
        )
    }

    private func roomIsConnected() -> Bool {
        guard let room else { return false }
        if case .connected = room.connectionState { return true }
        return false
    }

    private func ensureAuthorizedRemoteAudio() async throws -> Bool {
        guard let room else { throw ProbeError.roomUnavailable }
        let remoteAudio: [(RemoteParticipant, RemoteTrackPublication)] = room.remoteParticipants.values.flatMap { participant in
            participant.audioTracks.compactMap { publication in
                guard let remote = publication as? RemoteTrackPublication else { return nil }
                return (participant, remote)
            }
        }
        let authorized = remoteAudio.filter { participant, publication in
            participant.identity?.stringValue == expectedWorkerIdentity
                && publication.kind == .audio
                && publication.source == .microphone
        }
        if remoteAudio.contains(where: { participant, publication in
            let isAuthorized = participant.identity?.stringValue == expectedWorkerIdentity
                && publication.kind == .audio
                && publication.source == .microphone
            return publication.isSubscribed && !isAuthorized
        }) {
            throw ProbeError.unauthorizedRemoteAudioSubscribed
        }
        guard authorized.count <= 1 else {
            throw ProbeError.multipleAssistantAudioPublications
        }
        guard let (_, candidate) = authorized.first else {
            if selectedRemotePublication != nil {
                throw ProbeError.assistantAudioPublicationChanged
            }
            if let started = remoteWaitStartedAt, Date().timeIntervalSince(started) > 10 {
                throw ProbeError.assistantAudioTimeout
            }
            return false
        }
        let identity = ObjectIdentifier(candidate)
        if let selectedRemotePublication, selectedRemotePublication != identity {
            throw ProbeError.assistantAudioPublicationChanged
        }
        if selectedRemotePublication == nil {
            try await candidate.set(subscribed: true)
            selectedRemotePublication = identity
        }
        if !candidate.isSubscribed,
           let started = remoteWaitStartedAt,
           Date().timeIntervalSince(started) > 10 {
            throw ProbeError.assistantAudioTimeout
        }
        return candidate.isSubscribed
    }

    private func authorizedRemoteAudioIsSubscribed() -> Bool {
        guard let room, let selectedRemotePublication else { return false }
        return room.remoteParticipants.values
            .flatMap(\.audioTracks)
            .compactMap { $0 as? RemoteTrackPublication }
            .contains { ObjectIdentifier($0) == selectedRemotePublication && $0.isSubscribed }
    }

    private func validateLiveState() -> ProbeVerdict {
        let engine = AudioManager.shared.audioProcessingState
        let platform = AudioManager.shared.platformVoiceProcessingState
        guard platform.voiceProcessingEnabled.isRequested,
              platform.voiceProcessingEnabled.isActive else {
            return .blocked("VPIO was not requested and activated")
        }
        guard !platform.voiceProcessingBypassed.isRequested,
              !platform.voiceProcessingBypassed.isActive else {
            return .blocked("VPIO bypass is requested or active")
        }

        guard let ecRequest = engine.echoCancellation.requested,
              ecRequest.isEnabled,
              ecRequest.mode == .platform,
              !engine.echoCancellation.software.isResolved,
              !engine.echoCancellation.software.isActive,
              engine.echoCancellation.platform?.isResolved == true,
              engine.echoCancellation.platform?.isActive == true,
              engine.echoCancellation.effective == .platform,
              platform.echoCancellation.isRequested,
              platform.echoCancellation.isActive else {
            return .blocked("echo cancellation is not platform-only and active")
        }

        guard let agcRequest = engine.autoGainControl.requested,
              !agcRequest.isEnabled,
              agcRequest.mode == .software,
              !engine.autoGainControl.software.isResolved,
              !engine.autoGainControl.software.isActive,
              engine.autoGainControl.platform?.isResolved != true,
              engine.autoGainControl.platform?.isActive != true,
              engine.autoGainControl.effective == .disabled,
              !platform.autoGainControl.isRequested,
              !platform.autoGainControl.isActive,
              !platform.voiceProcessingAGCEnabled.isRequested,
              !platform.voiceProcessingAGCEnabled.isActive,
              !AudioManager.shared.isVoiceProcessingAGCEnabled else {
            return .blocked("AGC is not fully disabled")
        }

        guard let nsRequest = engine.noiseSuppression.requested,
              nsRequest.isEnabled == requestPlatformNoiseSuppression,
              nsRequest.mode == .platform,
              !engine.noiseSuppression.software.isResolved,
              !engine.noiseSuppression.software.isActive else {
            return .blocked("noise suppression request drifted or software NS is active")
        }
        if requestPlatformNoiseSuppression {
            guard engine.noiseSuppression.platform?.isResolved == true,
                  engine.noiseSuppression.platform?.isActive == true,
                  engine.noiseSuppression.effective == .platform,
                  platform.noiseSuppression.isRequested,
                  platform.noiseSuppression.isActive else {
                return .blocked("requested platform noise suppression is not active")
            }
        } else {
            let disabled = engine.noiseSuppression.effective == .disabled
                && engine.noiseSuppression.platform?.isResolved != true
                && engine.noiseSuppression.platform?.isActive != true
                && !platform.noiseSuppression.isRequested
                && !platform.noiseSuppression.isActive
            let coupledPlatform = platform.topology == .echoCancellationAndNoiseSuppressionCoupled
                && engine.noiseSuppression.effective == .platform
                && engine.noiseSuppression.platform?.isResolved == true
                && engine.noiseSuppression.platform?.isActive == true
                && platform.noiseSuppression.isRequested
                && platform.noiseSuppression.isActive
            guard disabled || coupledPlatform else {
                return .blocked("noise suppression is neither disabled nor topology-coupled platform NS")
            }
        }

        guard let hpfRequest = engine.highpassFilter.requested,
              !hpfRequest.isEnabled,
              hpfRequest.mode == .automatic,
              !engine.highpassFilter.software.isResolved,
              !engine.highpassFilter.software.isActive,
              engine.highpassFilter.platform == nil,
              engine.highpassFilter.effective == .disabled else {
            return .blocked("software high-pass filtering is active or unknown")
        }
        return .accepted("live processing accepted; acoustic proof still required")
    }

    private func isValidCanonicalTrackName(_ value: String) -> Bool {
        value.range(of: #"^microphone-[1-9][0-9]*$"#, options: .regularExpression) != nil
    }

    private func describe(_ result: AudioProcessingOptionsResult) -> String {
        switch result {
        case .applied: "applied"
        case .stored: "stored"
        @unknown default: "unknown"
        }
    }

    private func describe(_ error: Error) -> String {
        if let typed = error as? AudioProcessingOptionsError {
            return "audio-processing-\(String(describing: typed.code))"
        }
        if let probe = error as? ProbeError {
            return probe.rawValue
        }
        if let blocked = error as? ProbeBlockedError {
            return blocked.reason
        }
        return String(describing: type(of: error))
    }

    private func describe<Mode: Sendable>(_ state: AudioProcessingComponentState<Mode>) -> String {
        let requested = state.requested.map {
            "enabled=\($0.isEnabled),mode=\(String(describing: $0.mode))"
        } ?? "none"
        let platform = state.platform.map {
            "resolved=\($0.isResolved),active=\($0.isActive)"
        } ?? "unavailable"
        return "requested{\(requested)};software{resolved=\(state.software.isResolved),active=\(state.software.isActive)};platform{\(platform)};effective=\(state.effective)"
    }

    private func describe(_ state: PlatformVoiceProcessingComponentState) -> String {
        "available=\(state.isAvailable),requested=\(state.isRequested),active=\(state.isActive)"
    }

    private func describe(_ state: PlatformVoiceProcessingFlagState) -> String {
        "requested=\(state.isRequested),active=\(state.isActive)"
    }

    private func encode<T: Encodable>(_ value: T) -> String? {
        guard let data = try? JSONEncoder.pretty.encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

private enum ProbeError: String, Error {
    case publicationNameMismatch = "publication-name-mismatch"
    case postPublishProcessingNotApplied = "post-publish-processing-not-applied"
    case runtimeProcessingNotApplied = "runtime-processing-not-applied"
    case roomUnavailable = "room-unavailable"
    case unauthorizedRemoteAudioSubscribed = "unauthorized-remote-audio-subscribed"
    case multipleAssistantAudioPublications = "multiple-assistant-audio-publications"
    case assistantAudioTimeout = "assistant-audio-timeout"
    case assistantAudioPublicationChanged = "assistant-audio-publication-changed"
}

private struct ProbeBlockedError: Error {
    let reason: String
}

private struct ProbeVerdict {
    let accepted: Bool
    let message: String

    static func accepted(_ message: String) -> ProbeVerdict {
        ProbeVerdict(accepted: true, message: message)
    }

    static func blocked(_ message: String) -> ProbeVerdict {
        ProbeVerdict(accepted: false, message: message)
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}
