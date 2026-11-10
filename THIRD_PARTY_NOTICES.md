# Third-party licensing

The MIT license in this repository applies to the original project code and
documentation. It does not relicense dependencies, market data, or third-party
assets. Dependency source distributions and installed environments are not
included in this repository.

## Freqtrade

Freqtrade is an optional, separately installed GPLv3 dependency. See its
[license](https://github.com/freqtrade/freqtrade/blob/develop/LICENSE).
The files in `freqtrade/strategies/` implement this project's strategies against
Freqtrade's API; they are not a vendored copy of the Freqtrade engine.

MIT is GPL-compatible, but compatibility does not remove GPL obligations. If you
distribute a combined or derivative work with Freqtrade, comply with the applicable
GPL requirements, including source availability and license notices. Do not describe
such a combined distribution as MIT-only. This repository's MIT grant is not a
license to relicense Freqtrade or make a GPL-covered combination proprietary.
See the GNU project's [compatibility explanation](https://www.gnu.org/licenses/gpl-faq.html#WhatDoesCompatMean).

## Other dependencies and data

Dependencies are declared in `pyproject.toml`, `uv.lock`, `dashboard/package.json`
and `dashboard/package-lock.json`. Consult each package's license before redistribution.
Historical exchange data, model weights, trading databases and private deployment
artifacts are not distributed. Fetch data under its provider's applicable terms.
Synthetic test fixtures are not historical performance claims.
