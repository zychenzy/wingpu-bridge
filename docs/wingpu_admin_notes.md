# Admin Notes

This note covers the small amount of privileged setup that still exists around `wingpu`.

## Goal

Keep daily model serving unprivileged, and allow passwordless sudo only for a very narrow set of maintenance tasks inside WSL.

## Current Rule Shape

The WSL host uses root-owned wrapper scripts plus a tight sudoers allowlist.

Installed wrappers:

- `/usr/local/sbin/wingpu-install-build-prereqs`
- `/usr/local/sbin/wingpu-install-cuda-toolkit`
- `/usr/local/sbin/wingpu-install-experiment-prereqs`
- `/usr/local/sbin/wingpu-install-llamacpp-system`

Installed sudoers pattern:

```sudoers
Defaults:<wsl-user> env_reset
<wsl-user> ALL=(root) NOPASSWD: /usr/local/sbin/wingpu-install-build-prereqs, /usr/local/sbin/wingpu-install-cuda-toolkit, /usr/local/sbin/wingpu-install-experiment-prereqs, /usr/local/sbin/wingpu-install-llamacpp-system
```

## Why This Shape Is Preferred

Good:

- daily runtime stays unprivileged
- automation can perform narrow maintenance steps
- risk is much smaller than blanket `NOPASSWD: ALL`

Avoid:

- passwordless arbitrary shells
- wildcard sudo rules that can execute arbitrary code
- storing sudo passwords in scripts

## What Still Needs Privilege

These are the main privileged tasks:

- installing build prerequisites
- installing CUDA toolkit packages in Ubuntu
- installing extra experiment packages
- copying built `llama.cpp` binaries into `/usr/local/bin`

The serving path itself does not require sudo.

## Operational Advice

- keep a normal strong sudo password for the WSL user
- review any new wrapper before adding it to sudoers
- treat each `NOPASSWD` entry like a root API surface
