# Remote-Only Artifacts

This directory is reserved for generated mirrors of the live remote WSL environment.

It is intentionally not tracked in git by default because the generated output can contain:

- host-specific paths
- usernames and machine names
- runtime state and logs
- benchmark artifacts
- local inventory snapshots

To regenerate a local mirror for your own use:

```bash
./bridge/mac/mirror_remote_wsl_llama.sh gpu-host Ubuntu
```

The generated output will be written under `bridge/remote-only/wsl-llama/` and ignored by git.
