# Pi development environment

Fresh cases have physical backing under `~/.cache/development/work-<opaque>/state`
and a logical symlink below the requested ignored `out/` directory. Preparation
resolves the physical path before creating editable environments. Pi's SDK,
minimal worker and rg binary are mirrored into a read-only development runtime
cache keyed by their current content hashes. Kernel `/proc/self/mountinfo` and
`/proc/self/mounts` therefore describe neutral physical sources as well as neutral
mount targets; their output is not rewritten or hidden. These names do not conceal
that Linux namespaces are in use.

Fresh preparations run tools at `/home/developer/project`, with a task-owned
Python environment at `/home/developer/.venv`, Node dependencies at
`/home/developer/.local/share/node`, Go modules at `/home/developer/go/pkg/mod`,
and shared read-only runtimes under `/opt/`. Pi's system context and actual tool
cwd use these same paths. Editable Python metadata, script shebangs and Python
runtime configuration are projected into the corresponding visible paths before
execution. Bytecode is compiled into the task home so tracebacks do not reuse
host-built filenames. The original pinned project instructions are unchanged.

`rg`, `uv`, Node, npm, Python, pytest, pip and Ruff are available; Go and Bun use
their recorded task runtimes. LangChain retains the test versions in its upstream
lock, with pinned pip/Ruff added. Other test-tool pins and the complete Python
freeze are recorded. Python runtime version, physical locator, executable and
shared-library hashes are recorded too. New managed interpreter downloads use
neutral shared runtime storage; previously prepared interpreter locations are
preserved as read-only dependencies and must accompany source evidence. `UV_LINK_MODE=copy` keeps writable task packages independent
of the host dependency cache. Go module, Python, Node and user caches are writable
and retained. Node dependency symlinks and Expo's scoped Jest configuration point
to their actual neutral mount paths; `JEST_CONFIG` identifies Expo's configuration.
Prepared Git authorship is `Developer <developer@localhost>`. Preparation removes
only pre-collection clone/setup reflogs, which otherwise retain private cache
paths and the host clone identity. Ordinary reflogs are enabled for subsequent
assistant Git work; checkpoint archives preserve those logs without modification.
An unstarted preparation can refresh its collector identity while retaining the
earlier plan by hash. Started trajectories require a fresh output directory.

Tools can use public development networks. Each subprocess receives a private
user/network namespace with slirp4netns NAT and working DNS. Host loopback is
explicitly disabled. Before the inner Bubblewrap process starts, nftables in the
outer namespace rejects private, link-local, reserved and metadata destination
addresses plus all current host interface IPv4 addresses. Only the internal DNS
proxy is excepted. IPv6 egress is disabled. Inner tools can use their own
loopback services but cannot change the outer firewall or reach host listeners.
No host firewall rules are changed. DNS/HTTPS availability can still fail for
ordinary upstream reasons; those failures must remain in any recorded trajectory.

The tool environment does not inherit provider keys, proxy credentials, SSH keys,
personal settings, host home files or collector artifacts. Only Pi's SDK modules
and the minimal tool worker are mounted, with neutral SDK installation metadata.
Collector source, baseline helpers and `PI_*` environment variables are absent
from live tools. Baseline helpers are mounted separately during preparation only,
using a separate `setup-tool-state` home/tmp/cache. The model starts with its own
`tool-state`; preparation probe bytecode and temporary fixtures are not shared.
Shared runtimes are read-only. Network access is not a claim that arbitrary remote
services are trustworthy or that every upstream monorepo has been bootstrapped.

Linux requirements include Bubblewrap, `unshare`, slirp4netns, nftables, Git,
Node.js 22+, npm, rg, curl, uv and Bun 1.3.12 for the Bun case. Namespace/network
setup fails closed if routing or firewall initialization fails. The offline test
suite warms the content-addressed SDK mirror before starting test timers, then
checks host-secret isolation, host-loopback/private/metadata rejection,
protected firewall state, ordinary tools, neutral prompt paths and writable
cache mounts. It makes no provider requests. A separate public HTTPS smoke check
can verify actual development connectivity without invoking a model.

## Checkpoints

Before the initial prompt and after every user turn, the collector records
`turns/NNN.workspace.tar.gz` and its SHA-256/size/root/Git-state receipt. These
archives retain the complete independent Git directory, tracked/untracked/ignored
workspace files, mutable Python/Node/Go dependencies and tool-state home/tmp.
Symlinks are stored without dereferencing them; dependency installations and
ignored task fixtures are not silently omitted. Shared immutable runtime versions
remain bound by provenance. The initial checkpoint is staged before the inference
timer starts; later checkpoint work is included in the active-turn bound, and
archive commands have their own ten-minute timeout.

`restoreCheckpoint(archive, destination, {sha256, sourceCaseDir, prepared})` in
`checkpoints.mjs` verifies the SHA-256 and extracts into a fresh absolute private
directory, using a neutral physical case and a logical symlink at the requested
output path. The return value identifies both actual `destination` and logical
`alias`. The Python helper rejects traversal, duplicate paths, special files
and writes through symlink parents; it retains literal symlink targets without
following them. The returned `{workspace, prepared}` relocates task-owned host
mounts and adds source-path aliases for editable Python metadata. It accepts original logical or physical source paths when relocating metadata.
It does not modify native messages or mount the old case. Persist the returned prepared
metadata before using the restored task with the collector.

Restore the matching native message prefix, source state, runtime and dependency
mounts together. Shared immutable runtime paths still need their recorded runtime
identities; moving the archive alone to another machine does not install them.
Processes, sockets and external remote-service state are not checkpointed. A
tracked `.patch` alone is insufficient for continuation.

This environment applies only to fresh plans made with the new collector identity.
Previous plans, captures and path-normalized review derivatives remain historical
evidence. They must not be rewritten to pretend their failed commands succeeded.
