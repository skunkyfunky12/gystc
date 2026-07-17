# GYSTC

Get Your Shit Together, Claude — persistent long-term memory via MCP.

See `INSTALL.html` for the full installation guide.

## Windows / SmartScreen

The Windows release binary is unsigned, so SmartScreen will show "Windows protected your PC"
when you run it. This is expected — click **More info** → **Run anyway**.

To verify the download hasn't been tampered with, check its hash against `SHA256SUMS.txt`
from the same release:

```powershell
Get-FileHash <datei> -Algorithm SHA256
```

Compare the resulting hash against the matching line in `SHA256SUMS.txt`.
