# Public deployment templates

The service units and Nginx configuration in `deploy/` are examples. Paths, users,
certificate locations and hostnames must be adapted to your environment. Addresses
in historical operational notes were replaced by documentation-only placeholders.
Those notes do not establish that any public deployment is currently running.

No Freqtrade runtime configuration or trading database is included. Start from
`freqtrade/config_dryrun.example.json` and create an ignored local runtime file.
It is deliberately dry-run-only, initially stopped, and has empty credentials.
You must generate compatible model signals and choose a valid market universe before
starting a simulation; the two example pairs are not a strategy recommendation.

Bind the read-only dashboard API to `127.0.0.1:8080`. Set
`ALPHA_DASHBOARD_API_TARGET` in an ignored `dashboard/.env.local` if needed.
Configure TLS and authentication in your reverse proxy before granting remote access.
The example Nginx template alone does not supply authentication.

Do not expose Freqtrade's trading RPC API as an unauthenticated public service.
Do not change `dry_run` to false as a deployment shortcut. The separate testnet path
has additional gates; none of these examples certify readiness for funded trading.
