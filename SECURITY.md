# Security policy

## Scope

`spin_based_neural_network` is a research framework for classical spin
models, topological quantum computing primitives, and neural-network
quantum states. It is intended for scientific computation on trusted
local data and **not** for serving untrusted input over a network.

Threat model:

- The framework reads configuration and lattice data from local files
  and CLI arguments. Those inputs are assumed to come from the user
  operating the binary.
- No network interfaces are exposed. No data is transmitted off-host.
- The visualization binary (`visualization`) opens an SDL2 window on
  the local display; it does not listen on any socket.

If you use the library in a setting that exposes it to untrusted input
(e.g. a web service accepting configurations), that is outside the
supported threat model and additional hardening is required.

## Reporting a vulnerability

If you discover a security issue — for instance:

- A path that allows a crafted input file to overwrite host files
  outside the working directory;
- A buffer overflow or use-after-free reachable from public API;
- A supply-chain concern in how the framework pulls in optional bridges
  (libirrep, moonlab, eshkol, etc.);
- Any other issue that materially affects confidentiality, integrity,
  or availability of a deployment;

please report it privately by emailing the maintainer (see
`CONTRIBUTING.md` for contact). Do **not** open a public GitHub issue
for security-sensitive reports.

I will acknowledge receipt within seven days and communicate an
expected remediation timeline, typically within 30 days for issues
affecting the public API and within 90 days for issues in optional
bridges or build tooling.

## Disclosure policy

Coordinated disclosure preferred. After a fix is merged and released,
the security advisory will be published via GitHub Security Advisories
with credit to the reporter (unless anonymity is requested).

## Out of scope

- Numerical instability at extreme parameters (e.g. entropy
  calculations beyond `N > 20`) is a research / accuracy issue, not a
  security issue. File a regular bug report.
- Optional bridges gated by `#ifdef SPIN_NN_HAS_*` are not exercised
  in the default build. Their security status is the responsibility
  of the upstream project.
- Performance / DoS concerns on hostile input are out of scope; the
  framework is CPU-bound and single-threaded by default and does not
  claim resource guarantees against adversarial workloads.

## Third-party code

The framework does not vendor third-party C source. Optional bridges
link against external libraries (`libirrep`, `libquantumsim`,
`libspin_engine`) which ship from their own repositories with their
own licenses and security policies. The framework compiles cleanly
without any of them.

## Build-time hardening suggestions

If you are deploying builds of this framework into any environment
where hardening matters:

- Build with `-D_FORTIFY_SOURCE=2 -fstack-protector-strong` in
  `CFLAGS`.
- Enable AddressSanitizer during development and CI: pass
  `CFLAGS_COMMON+=" -fsanitize=address,undefined"`.
- Pin dependency versions via `VERSION_PINS` (planned for v0.5; see
  `docs/architecture_v0.4.md`).

## Version history of this policy

- 2026-04-18: initial policy for v0.4.
