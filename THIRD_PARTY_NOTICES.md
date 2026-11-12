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
The explicitly published runtime assets include the Tokyo five-model LightGBM
ensemble, the selected XGB15 weights and historical acceptance inputs, and a recent
market-cap bootstrap snapshot. Other private datasets, weights, trading databases
and deployment credentials remain excluded. Provider data remains subject to its applicable terms;
the project license does not relicense third-party data.
Synthetic test fixtures are not historical performance claims.
