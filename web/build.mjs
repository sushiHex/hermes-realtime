import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = dirname(fileURLToPath(import.meta.url));
const output = resolve(root, "../src/hermes_realtime/client/static");
await mkdir(resolve(output, "assets"), { recursive: true });

const result = await build({
  entryPoints: [resolve(root, "src/main.ts")],
  bundle: true,
  metafile: true,
  minify: true,
  legalComments: "external",
  outfile: resolve(output, "assets/app.js"),
  platform: "browser",
  sourcemap: false,
  target: ["safari16", "chrome120", "firefox120"],
});

function bundledPackageName(input) {
  const normalized = input.replaceAll("\\", "/");
  const marker = "node_modules/";
  const markerIndex = normalized.lastIndexOf(marker);
  if (markerIndex < 0) return null;
  const parts = normalized.slice(markerIndex + marker.length).split("/");
  return parts[0].startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
}

const productionPackages = new Set(
  Object.keys(result.metafile.inputs).map(bundledPackageName).filter(Boolean),
);
const packageLock = JSON.parse(await readFile(resolve(root, "package-lock.json"), "utf8"));
for (const [packagePath, packageRecord] of Object.entries(packageLock.packages ?? {})) {
  if (packagePath.startsWith("node_modules/") && !packageRecord.dev) {
    productionPackages.add(bundledPackageName(packagePath));
  }
}
const packageNames = [...productionPackages].filter(Boolean).sort();
const noticeSections = [
  "Hermes Realtime browser bundle third-party notices",
  "",
  "Generated deterministically from esbuild's production input graph and the non-development package-lock closure.",
];

const fallbackLicensePackages = {
  "0BSD": "tslib",
  "Apache-2.0": "livekit-client",
  "BSD-3-Clause": "webrtc-adapter",
  MIT: "jose",
};

async function rootLicenseEntries(packageRoot, labelPrefix = "") {
  const licenseFiles = (await readdir(packageRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && /^licen[cs]e(?:\.|$)/i.test(entry.name))
    .map((entry) => entry.name)
    .sort();
  return Promise.all(
    licenseFiles.map(async (licenseFile) => ({
      label: `${labelPrefix}${licenseFile}`,
      text: (await readFile(resolve(packageRoot, licenseFile), "utf8")).replaceAll("\r\n", "\n"),
    })),
  );
}

for (const packageName of packageNames) {
  const packageRoot = resolve(root, "node_modules", ...packageName.split("/"));
  const manifest = JSON.parse(await readFile(resolve(packageRoot, "package.json"), "utf8"));
  const source =
    (typeof manifest.repository === "string" ? manifest.repository : manifest.repository?.url) ??
    manifest.homepage ??
    "unspecified";
  let licenseEntries = await rootLicenseEntries(packageRoot);
  if (licenseEntries.length === 0) {
    const identifiers = [...new Set((manifest.license ?? "").match(/[A-Za-z0-9.-]+/g) ?? [])]
      .filter((identifier) => fallbackLicensePackages[identifier]);
    if (identifiers.length === 0) {
      throw new Error(`Bundled package ${packageName} has no usable license source`);
    }
    licenseEntries = [];
    for (const identifier of identifiers) {
      const fallbackPackage = fallbackLicensePackages[identifier];
      const fallbackRoot = resolve(root, "node_modules", ...fallbackPackage.split("/"));
      const fallbackEntries = await rootLicenseEntries(
        fallbackRoot,
        `SPDX ${identifier} via ${fallbackPackage}: `,
      );
      if (fallbackEntries.length === 0) {
        throw new Error(`Fallback package ${fallbackPackage} has no root license file`);
      }
      licenseEntries.push(...fallbackEntries);
    }
  }

  noticeSections.push(
    "",
    "=".repeat(72),
    `${manifest.name}@${manifest.version}`,
    `Declared license: ${manifest.license ?? "unspecified"}`,
    `Source: ${source}`,
  );
  for (const licenseEntry of licenseEntries) {
    noticeSections.push("", `--- ${licenseEntry.label} ---`, licenseEntry.text.trimEnd());
  }
}

await writeFile(
  resolve(output, "assets/app.js.LEGAL.txt"),
  `${noticeSections.join("\n")}\n`,
  "utf8",
);
await cp(resolve(root, "index.html"), resolve(output, "index.html"));
await cp(resolve(root, "src/styles.css"), resolve(output, "assets/styles.css"));
