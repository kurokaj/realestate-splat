# ReScan source and application security audit plan

## Purpose

This document defines a thorough, repeatable review of the ReScan iOS application before using it to capture private indoor spaces or extending it for Buildvision3D.

The primary concern is that the application or its build/install chain could leave its intended boundaries—for example, exfiltrate unrelated phone data, install malware, execute unexpected code, or communicate with an untrusted service. Capture privacy, dependency risk, build integrity, and data handling are also in scope.

This is an engineering audit checklist, not a guarantee of absolute safety. A future agent must record evidence for every check, distinguish verified facts from assumptions, and stop rather than silently skip checks that cannot be performed.

## Scope and threat model

### Assets to protect

- iPhone files and personal data outside the app's capture directory
- captured RGB, depth, confidence, pose, mesh, and metadata files
- Apple ID/developer signing credentials
- Buildvision3D credentials, source code, and local files on the Mac
- local network and other devices reachable from the iPhone
- integrity of the ReScan source, dependencies, build tools, and produced app

### Threats to test

- hidden upload or telemetry of captures or unrelated device data;
- unexpected network listeners, peer discovery, or remote-control channels;
- reading files, photos, contacts, location, microphone, motion, or other data beyond the stated purpose;
- malicious or obfuscated code, backdoors, downloader behavior, persistence, or exploit attempts;
- compromised GitHub release artifacts, build scripts, package dependencies, or sideload tools;
- unsafe handling of paths, archives, URLs, camera data, or imported files;
- code that changes behavior based on environment, date, debug/release mode, or network response;
- leakage through logs, crash reports, analytics, pasteboard, temporary files, or backups;
- supply-chain changes after the audited commit.

## Audit rules

1. Audit a pinned commit, not a moving branch. Record repository URL, commit SHA, date, and local clone hash.
2. Prefer building from source locally over installing a prebuilt IPA.
3. Do not run installation scripts, binaries, or Xcode build phases before reviewing them.
4. Perform the review in an isolated macOS user account or disposable VM where practical.
5. Use a test iPhone or a device with no sensitive personal data when possible.
6. Use a separate Apple Developer signing identity. Never give source code or scripts access to production credentials.
7. Keep the first test device disconnected from sensitive local networks; use a monitored test network.
8. Preserve all audit outputs and hashes. Do not rely on screenshots alone.
9. Treat downloaded releases, IPA files, side-loading tools, GitHub Actions, and dependencies as separate trust boundaries.
10. If a check is not possible, mark it `NOT VERIFIED` with the reason and risk implication.

## Phase 0 — Establish the audit record

Record:

- repository URL and owner;
- commit SHA, branch, tags, and release used;
- clone date and source archive hash;
- macOS version, Xcode version, Swift version, iOS version, and iPhone model;
- exact build command and signing configuration;
- all tools and versions used during analysis;
- network topology and monitoring tools;
- auditor and review date.

Create an audit directory outside the application source containing command output, hashes, screenshots, packet captures, and conclusions. Do not commit secrets or private capture data.

Suggested initial commands after reviewing any scripts:

```shell
git rev-parse HEAD
git status --short
git log --show-signature -n 10
git ls-files > audit-tracked-files.txt
shasum -a 256 audit-tracked-files.txt
```

Also record repository-level facts: commit count, contributors, open issues, pull requests, releases, tags, branch protection, CI workflows, and dependency update activity.

## Phase 1 — Repository inventory

Enumerate every tracked and untracked file. Pay particular attention to:

- `.github/workflows/`
- shell, Python, Ruby, JavaScript, Make, XcodeGen, and Swift scripts;
- `project.yml`, `.xcodeproj`, `.xcworkspace`, `Package.swift`, `Package.resolved`;
- `Podfile`, `Cartfile`, npm/pnpm/yarn files, binary frameworks, vendored source, and archives;
- entitlements, provisioning, plist files, URL schemes, app extensions, widgets, and services;
- prebuilt `.ipa`, `.app`, `.framework`, `.xcframework`, `.dylib`, `.a`, `.so`, and executable files;
- test fixtures, sample payloads, credentials, API keys, certificates, and configuration files;
- generated code and files downloaded during a build.

Search for suspicious or high-risk constructs. Review every match in context rather than treating a keyword as proof of malicious behavior:

```shell
rg -n -i "URLSession|NWConnection|Network.framework|CFNetwork|WebSocket|socket|Bonjour|Multipeer|MCSession|CoreBluetooth|FTP|HTTP|HTTPS|upload|telemetry|analytics|crash|Sentry|Firebase|AdMob|tracking|pasteboard|UIPasteboard|NSPasteboard" .
rg -n -i "Process\(|NSTask|system\(|popen|Shell|dlopen|dlsym|objc_msgSend|performSelector|NSClassFromString|JavaScriptCore|WKWebView|eval\(" .
rg -n -i "FileManager|PHPhotoLibrary|CNContact|ABAddressBook|CLLocation|CoreLocation|AVAudio|Microphone|UIDevice|identifierForVendor|keychain|SecItem|UserDefaults|AppStorage" .
rg -n -i "base64|AES|RSA|CryptoKit|CommonCrypto|Keychain|private key|api[_-]?key|token|secret|password|credential" .
rg -n -i "curl|wget|python|ruby|bash|zsh|sh -c|chmod|xattr|osascript|defaults write|launchctl|installer|curl.*pipe|download" .
```

Inspect obfuscation, encoded strings, generated blobs, unusually large files, and code that is never referenced by the normal capture flow.

## Phase 2 — Source-code review by subsystem

Review every source file, not only the capture UI. Create a table with file, responsibility, data accessed, external effects, and finding.

### Camera and LiDAR capture

- Verify only the intended camera, depth, confidence, motion, and ARKit APIs are used.
- Confirm buffers are copied and released safely.
- Check that timestamps and frame associations cannot cause unrelated memory or file access.
- Check all camera-format and device-model branches, including iPhone 17 Pro-specific paths.
- Verify no hidden photo-library access is used to obtain camera frames.
- Verify microphone capture is not enabled unless explicitly required.
- Verify location, contacts, Bluetooth, local network, and motion permissions are not requested unnecessarily.
- Check whether camera frames, depth, or poses are retained after export or capture deletion.

### File and storage handling

- Enumerate every read and write path.
- Confirm writes remain inside the app sandbox or an explicitly selected export directory.
- Check URL/bookmark/security-scoped resource handling.
- Test path traversal (`../`), absolute paths, symlinks, unusual Unicode names, null bytes, and archive extraction.
- Check file permissions, temporary files, caches, logs, thumbnails, crash dumps, and backups.
- Confirm deletion actually removes private capture data or document the recovery behavior.
- Verify no arbitrary path supplied by a file, QR code, URL, or network message is executed or used as a destination.

### Export and import formats

- Review parsers for PNG, MOV/MP4, EXR, CSV, OBJ, ZIP, JSON, and any custom format.
- Check decompression limits, integer overflow, malformed metadata, huge dimensions, recursion, and denial-of-service cases.
- Check whether imported filenames or metadata are inserted into shell commands, URLs, logs, or HTML without escaping.
- Verify exports contain only intended RGB/depth/confidence/pose/intrinsic data.

### Networking and remote communication

- Locate all networking code and list every host, port, protocol, DNS lookup, and certificate policy.
- Determine whether networking is part of ReScan or only an optional developer feature.
- Verify no network code runs during ordinary offline capture.
- Check default settings, startup hooks, background tasks, notification handlers, URL handlers, and app extensions for automatic communication.
- Check TLS certificate validation, redirects, hostname validation, authentication, replay protection, and logging of request bodies.
- Confirm no camera/depth/pose data is sent without an explicit user action.
- Confirm no arbitrary incoming network command can trigger file reads, shell execution, or app behavior.
- Test behavior with Wi-Fi, cellular data, VPN, captive portal, DNS failure, and hostile local network conditions.

### Permissions and privacy

- Inspect `Info.plist` usage descriptions and compare every permission to code usage.
- Inspect entitlements and provisioning capabilities.
- Check camera, microphone, photo library, location, Bluetooth, local-network, motion, iCloud, push notifications, background modes, associated domains, and keychain access groups.
- Verify the app has no unexpected App Groups, shared containers, document providers, or extensions.
- Confirm permission prompts are truthful and least-privilege.
- Check whether denial of an optional permission causes graceful failure rather than unsafe fallback.

### Process, code loading, and persistence

- Search for process creation, shell execution, dynamic library loading, JavaScript execution, downloaded code, and runtime class lookup.
- Verify the app cannot install profiles, certificates, launch agents, daemons, keyboard extensions, or other persistent components.
- Inspect background modes and lifecycle handlers for unexpected persistence.
- Check custom URL schemes and universal links for command injection or unauthorized actions.
- Check use of `WKWebView`, JavaScript bridges, remote HTML, and local web servers.

### Logging and diagnostics

- Search every log statement and debug screen for RGB paths, depth values, poses, device identifiers, tokens, or personal data.
- Verify release builds disable verbose logs and debug endpoints.
- Check `os_log` privacy annotations and crash reporting behavior.
- Check pasteboard, screenshots, previews, Quick Look, share sheets, and recent-document lists for unintended capture exposure.

## Phase 3 — Dependency and supply-chain audit

For every dependency:

- record source URL, version, commit/tag, license, maintainer, release date, and transitive dependencies;
- prefer immutable commit hashes or verified tags;
- inspect source and build scripts, not only package names;
- check whether it contains networking, code generation, binary downloads, shell commands, or post-install hooks;
- check GitHub security advisories, CVEs, issue history, and abandoned/unmaintained status;
- run the ecosystem's dependency audit tools where applicable;
- compare resolved dependencies before and after the audit.

Inspect GitHub Actions and build automation for:

- unpinned third-party actions;
- secret exposure in logs;
- downloading and executing unverified artifacts;
- release or signing permissions;
- use of pull-request code with write access;
- workflows triggered by untrusted forks;
- scripts that modify the developer machine.

Do not install XcodeGen, SideStore, iloader, or other sideload/build tools without auditing those tools independently. Build from source whenever possible and hash the resulting artifacts.

## Phase 4 — Build and artifact integrity

1. Build on a clean macOS account from the pinned source commit.
2. Review Xcode build phases before running them.
3. Disable network access during the build where practical after dependencies are present.
4. Record compiler warnings and errors.
5. Compare a reproducible second build if possible.
6. Inspect the generated `.app`, entitlements, embedded frameworks, Info.plist, URL schemes, and linked libraries.
7. Verify code signatures and provisioning profile:

```shell
codesign -dvvv --entitlements :- path/to/ReScan.app
codesign --verify --deep --strict --verbose=4 path/to/ReScan.app
otool -L path/to/ReScan.app/ReScan
```

8. Search the final bundle for unexpected executables, scripts, URLs, certificates, private keys, or large encoded payloads.
9. Compare source-to-binary symbols and resources. Explain every embedded framework and asset.
10. Never install an IPA whose hash, source commit, or signing identity cannot be identified.

## Phase 5 — Static analysis

Use multiple independent checks:

- Xcode compiler warnings with warnings treated seriously;
- SwiftLint or equivalent style/static analysis;
- Semgrep rules for Swift/Objective-C security patterns;
- dependency vulnerability scanners;
- secret scanners such as Gitleaks or TruffleHog against history and current files;
- malware scanners on downloaded archives and build artifacts;
- `otool`, `strings`, `nm`, and `codesign` inspection of the final app;
- review of exported symbols and linked libraries;
- archive and parser fuzzing for import paths.

False positives must be investigated and documented. A clean scanner result is not proof of safety.

## Phase 6 — Dynamic and network testing

Use a dedicated test iPhone and a monitored isolated network. Establish a baseline before installing ReScan.

### Offline test

- Disable Wi-Fi and cellular data.
- Capture RGB-D data, browse the app, export files, and delete a session.
- Confirm all intended operations work offline.
- Confirm no crash, log, or permission behavior indicates an attempted network dependency.

### Monitored-network test

- Route traffic through a controlled DNS resolver and proxy such as mitmproxy or Charles.
- Capture DNS, TCP, TLS, UDP, Bonjour/mDNS, and IPv6 traffic.
- Run the app through launch, permission grant/denial, idle, capture, preview, export, share, deletion, backgrounding, and update flows.
- Repeat with an empty capture and with a synthetic capture.
- Identify every destination and payload. Treat unexplained traffic as a finding until resolved.
- Check for certificate pinning and whether any endpoint receives device identifiers, location, media, depth, or pose data.

### Host/device observation

- Inspect app sandbox files before, during, and after capture.
- Monitor CPU, memory, file descriptors, processes, open ports, and background activity.
- Verify no new profiles, certificates, launch agents, extensions, or persistent services appear.
- Check iOS Settings for newly granted capabilities.
- Inspect macOS build-machine changes after running build/install scripts.

Do not bypass TLS security or install unknown root certificates on a personal device. Use a disposable test environment for interception.

## Phase 7 — Adversarial testing

Test the app with:

- malformed images, videos, depth maps, confidence maps, CSV, JSON, OBJ, ZIP, and custom files;
- huge dimensions and oversized archives;
- invalid timestamps, NaNs, infinities, negative depths, and extreme coordinates;
- filenames containing traversal sequences, shell metacharacters, Unicode confusables, and very long paths;
- repeated start/stop, low storage, thermal pressure, interrupted exports, denied permissions, and device reboot;
- hostile local-network packets and unexpected server responses;
- malformed custom URLs, QR codes, document-provider URLs, and share-sheet inputs;
- corrupted or partially written captures.

Record crashes, hangs, data corruption, unsafe writes, excessive resource use, or unexpected external effects.

## Phase 8 — Data-leakage validation

Create a test phone with unique canary files and identifiers in locations the app should not access. Examples:

- uniquely named files outside the app sandbox;
- synthetic contacts and calendar entries;
- a canary photo library item;
- a test location and Wi-Fi network name;
- a fake token that must never leave the device.

After each test, inspect network captures, logs, exported files, crash reports, pasteboard, shared containers, and cloud backups. Confirm that only explicitly selected capture data is exported.

Check whether metadata includes:

- exact device identifiers;
- location, compass, or room coordinates;
- timestamps and timezone;
- Apple account or developer information;
- filenames revealing private paths;
- unrelated photo or document metadata.

## Phase 9 — Source maintenance and trust assessment

Review:

- commit signing and author consistency;
- sudden ownership, maintainer, or dependency changes;
- release/tag provenance;
- security policy and vulnerability reporting process;
- issue and pull-request responsiveness;
- stale branches and abandoned dependencies;
- whether releases are reproducible from public source;
- whether documentation matches the actual source.

Small community size is not a vulnerability by itself, but it means fewer independent reviewers and less evidence of real-world use. Record this as a confidence limitation.

## Required audit outputs

The future agent must produce:

```text
audit/
  scope-and-versions.md
  repository-inventory.txt
  source-review.md
  permissions-and-entitlements.md
  dependencies-and-licenses.md
  build-integrity.md
  static-analysis.md
  network-observations.md
  dynamic-test-results.md
  adversarial-test-results.md
  data-leakage-results.md
  artifact-hashes.txt
  findings.json
  final-recommendation.md
```

Each finding should include:

- ID;
- severity: critical/high/medium/low/informational;
- affected file, symbol, dependency, or build step;
- evidence and reproduction steps;
- impact;
- confidence;
- remediation;
- whether it blocks personal testing, internal use, or production use.

## Go/no-go criteria

### Automatic no-go

- unexplained outbound transmission of capture or unrelated personal data;
- arbitrary code execution, shell execution, persistence, or downloader behavior;
- unexplained privileged entitlement or permission;
- untrusted or unverifiable binary dependency in the production path;
- inability to identify the source commit or signing/build provenance;
- confirmed malicious behavior or serious exploitable vulnerability.

### Conditional approval for controlled testing

Controlled personal testing may be acceptable when:

- all network destinations are understood or the app is proven to work offline;
- no unrelated data access is found;
- the app is built locally from a pinned commit;
- the test device contains no sensitive personal data;
- medium-risk findings are documented and isolated;
- raw captures remain under the operator's control.

### Production-readiness requirements

Before production or commercial use, additionally require:

- reviewed license and commercial-use position;
- repeatable builds and pinned dependencies;
- documented update and vulnerability-response process;
- tested data deletion and retention behavior;
- resolved high/critical findings;
- regression tests for permissions, network silence, export scope, and file safety;
- review of any modifications made to ReScan or copied into Buildvision3D.

## Recommended first pass

Do not begin by reading every SwiftUI view equally. Start with this order:

1. Pin and hash the repository.
2. Read the README, license, project configuration, entitlements, Info.plist, build scripts, and workflows.
3. Inventory dependencies and binary artifacts.
4. Search all networking, process-execution, file-access, dynamic-loading, and permission APIs.
5. Review capture, export, and lifecycle code end to end.
6. Build locally without installing a downloaded IPA.
7. Run offline, monitored-network, and canary-data tests.
8. Inspect the final signed app and all generated artifacts.
9. Write the findings and recommendation before using the app for real indoor captures.

