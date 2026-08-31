import SwiftUI

@main
struct VPIOProbeApp: App {
    @StateObject private var model = ProbeModel()

    var body: some Scene {
        WindowGroup {
            ProbeView(model: model)
        }
    }
}
