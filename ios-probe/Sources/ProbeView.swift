import SwiftUI

struct ProbeView: View {
    @ObservedObject var model: ProbeModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Ephemeral connection") {
                    TextField("wss:// private LiveKit URL", text: $model.serverURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Short-lived participant token", text: $model.token)
                    SecureField("Expected assistant identity", text: $model.expectedWorkerIdentity)
                    TextField("Canonical track name", text: $model.trackName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Toggle("Request coupled platform noise suppression", isOn: $model.requestPlatformNoiseSuppression)
                }

                Section("Controls") {
                    Button(model.isConnected ? "Disconnect" : "Connect and publish") {
                        Task {
                            if model.isConnected {
                                await model.disconnect()
                            } else {
                                await model.connect()
                            }
                        }
                    }
                    .disabled(model.isBusy)

                    Button("Reapply processing request") {
                        Task { await model.reapplyProcessing() }
                    }
                    .disabled(!model.isConnected || model.isBusy)

                    Button(model.prefersSpeaker ? "Switch to receiver preference" : "Switch to speaker preference") {
                        Task { await model.toggleOutputPreference() }
                    }

                    Button("Capture state now") {
                        model.captureSample(reason: "manual")
                    }

                    Button("Export sanitized evidence") {
                        model.exportEvidence()
                    }
                    .disabled(model.samples.isEmpty)
                }

                Section("State") {
                    LabeledContent("Status", value: model.status)
                    LabeledContent("Output preference", value: model.prefersSpeaker ? "speaker" : "receiver")
                    LabeledContent("Samples", value: String(model.samples.count))
                    if let exportURL = model.exportURL {
                        ShareLink(item: exportURL) {
                            Label("Share evidence JSON", systemImage: "square.and.arrow.up")
                        }
                    }
                }

                Section("Latest sanitized sample") {
                    Text(model.latestSampleText)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("VPIO Probe")
        }
    }
}
