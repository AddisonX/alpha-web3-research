from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_web3_research import __version__
from alpha_web3_research.backtest.dashboard import FreqtradeDashboardExporter
from alpha_web3_research.backtest.freqtrade import FreqtradeOHLCVExporter
from alpha_web3_research.backtest.hourly_rolling import HourlyPortfolioRules
from alpha_web3_research.config import Settings, load_settings
from alpha_web3_research.data.exchange import create_exchange
from alpha_web3_research.data.providers import create_ohlcv_fetcher
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.evaluation.decorrelation import FactorDecorrelator
from alpha_web3_research.evaluation.single_factor import SingleFactorEvaluator
from alpha_web3_research.factors.builder import FactorBuilder
from alpha_web3_research.factors.hourly import HourlyMultiTimeframeFactorBuilder
from alpha_web3_research.labels.builder import LabelBuilder
from alpha_web3_research.labels.hourly import HourlyMultiHorizonLabelBuilder
from alpha_web3_research.live.config import write_live_long_only_config
from alpha_web3_research.live.reconciliation import audit_dry_run, publish_audit
from alpha_web3_research.live.signals import LiveLongOnlySignalProducer
from alpha_web3_research.live.testnet import write_testnet_config
from alpha_web3_research.models.hourly_rolling import (
    HourlyRollingResearchConfig,
    HourlyRollingResearchRunner,
)
from alpha_web3_research.models.production import ProductionModelPromoter
from alpha_web3_research.strategies.registry import StrategyRegistry
from alpha_web3_research.universe.builder import UniverseBuilder

console = Console()

app = typer.Typer(help="Alpha Web3 Research command line interface.")
config_app = typer.Typer(help="Inspect project configuration.")
data_app = typer.Typer(help="Fetch, store, catalog, and inspect market data.")
universe_app = typer.Typer(help="Build and inspect tradable universes.")
labels_app = typer.Typer(help="Build supervised learning labels.")
factors_app = typer.Typer(help="Build candidate alpha factors.")
evaluation_app = typer.Typer(help="Evaluate factors and models.")
models_app = typer.Typer(help="Train and inspect machine learning models.")
backtest_app = typer.Typer(help="Export signals and run trading-system validation.")
live_app = typer.Typer(help="Promote validated models and generate live dry-run signals.")
strategies_app = typer.Typer(help="Inspect versioned strategies and their historical research runs.")
research_app = typer.Typer(help="Run isolated point-in-time research without changing legacy production artifacts.")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")
app.add_typer(labels_app, name="labels")
app.add_typer(factors_app, name="factors")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(models_app, name="models")
app.add_typer(backtest_app, name="backtest")
app.add_typer(live_app, name="live")
app.add_typer(strategies_app, name="strategies")
app.add_typer(research_app, name="research")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _settings(config_file: Optional[Path]) -> Settings:
    return load_settings(config_file)


@research_app.command("prepare-pit-rank")
def prepare_pit_rank(
    research_config: Path = typer.Option(Path("strategies/configs/pit_rank64_202501_202610.v1.json"), "--research-config"),
    workers: int = typer.Option(12, "--workers", min=1, max=32),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    from alpha_web3_research.research.pit_rank import PITRankResearchConfig, prepare_pit_rank_data

    settings = _settings(config_file)
    path = research_config if research_config.is_absolute() else settings.project_root / research_config
    result = prepare_pit_rank_data(settings, PITRankResearchConfig.load(path), workers, console.print)
    console.print(f"[green]PIT data prepared[/green]: {result}")


@research_app.command("run-pit-rank")
def run_pit_rank(
    research_config: Path = typer.Option(Path("strategies/configs/pit_rank64_202501_202610.v1.json"), "--research-config"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    from alpha_web3_research.models.pit_rank import PITRankResearchRunner
    from alpha_web3_research.research.pit_rank import PITRankResearchConfig

    settings = _settings(config_file)
    path = research_config if research_config.is_absolute() else settings.project_root / research_config
    result = PITRankResearchRunner(settings, PITRankResearchConfig.load(path)).run(console.print)
    console.print(f"[green]PIT rank research complete[/green]: {result}")


@research_app.command("run-pit-xgboost")
def run_pit_xgboost(
    research_config: Path = typer.Option(Path("strategies/configs/pit_rank64_xgboost_2025_train_2026_test.v1.json"), "--research-config"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    from alpha_web3_research.models.pit_xgboost import PITXGBoostConfig, PITXGBoostResearchRunner

    settings = _settings(config_file)
    path = research_config if research_config.is_absolute() else settings.project_root / research_config
    result = PITXGBoostResearchRunner(settings, PITXGBoostConfig.load(path)).run(console.print)
    console.print(f"[green]2026 XGBoost backtest complete[/green]: {result}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show package version and exit."),
) -> None:
    if version:
        console.print(f"alpha-web3-research {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print("Alpha Web3 Research: CCXT data, factor mining, LightGBM models, Freqtrade validation.")
        console.print("Run [bold]alpha-web3-research --help[/bold] to see available commands.")


@config_app.command("show")
def show_config(
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    table = Table(title="Alpha Web3 Research Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in settings.public_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@strategies_app.command("list")
def list_strategies(
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    registry = StrategyRegistry(_settings(config_file))
    definitions = registry.definitions()
    table = Table(title="Strategy Registry")
    for column in ["Strategy key", "Status", "Direction", "Frequency", "Selection", "Name"]:
        table.add_column(column)
    for definition in definitions:
        table.add_row(
            definition.strategy_key,
            definition.status,
            definition.direction,
            definition.decision_frequency,
            definition.selection_method,
            definition.name,
        )
    console.print(table)


@strategies_app.command("show")
def show_strategy(
    strategy: str = typer.Argument(..., help="Strategy id or versioned strategy key."),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    definition = StrategyRegistry(_settings(config_file)).get(strategy)
    console.print_json(data=definition.model_dump(mode="json"))


@strategies_app.command("sync")
def sync_strategy_registry(
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    settings = _settings(config_file)
    registry = StrategyRegistry(settings)
    definitions, runs = registry.sync()
    ParquetOHLCVStore(settings=settings).create_duckdb_database()
    console.print(f"[green]strategy registry synced[/green]: {len(definitions)} definitions, {len(runs)} runs")
    console.print(f"definitions: [bold]{registry.registry_path}[/bold]")
    console.print(f"definition index: [bold]{registry.definitions_parquet_path}[/bold]")
    console.print(f"runs:        [bold]{registry.runs_path}[/bold]")
    console.print(f"run index:   [bold]{registry.runs_parquet_path}[/bold]")


@strategies_app.command("runs")
def list_strategy_runs(
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Strategy id or versioned key."),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    registry = StrategyRegistry(_settings(config_file))
    runs = registry.discover_runs(strategy)
    table = Table(title="Strategy Research Runs")
    for column in ["Strategy key", "Kind", "Run id", "Created at", "Artifact"]:
        table.add_column(column)
    for run in runs:
        table.add_row(run.strategy_key, run.kind, run.run_id, run.created_at, run.primary_artifact)
    console.print(table)


@data_app.command("list-symbols")
def list_symbols(
    quote: str = typer.Option("USDT", "--quote", help="Quote asset to filter."),
    limit: int = typer.Option(30, "--limit", min=1, help="Maximum number of symbols to display."),
    sort_by_quote_volume: bool = typer.Option(False, "--sort-by-quote-volume", help="Sort symbols by current ticker quote volume."),
    include_non_crypto: bool = typer.Option(False, "--include-non-crypto", help="Include non-crypto underlyings such as equity perpetuals."),
    include_stable: bool = typer.Option(False, "--include-stable", help="Include stablecoin base assets."),
    include_leveraged: bool = typer.Option(False, "--include-leveraged", help="Include leveraged token style base assets."),
    sandbox: bool = typer.Option(False, "--sandbox", help="Use exchange sandbox/testnet endpoints."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    exchange = create_exchange(settings, sandbox=sandbox)
    fetcher = create_ohlcv_fetcher(exchange=exchange, settings=settings)
    symbols = fetcher.list_usdt_perpetual_symbols(
        quote=quote,
        limit=limit,
        exclude_non_crypto=not include_non_crypto,
        exclude_stable=not include_stable,
        exclude_leveraged=not include_leveraged,
        sort_by_quote_volume=sort_by_quote_volume,
    )

    table = Table(title=f"{settings.exchange_id} {quote} perpetual symbols")
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    for index, symbol in enumerate(symbols, start=1):
        table.add_row(str(index), symbol)
    console.print(table)


@data_app.command("rank-symbols")
def rank_symbols(
    quote: str = typer.Option("USDT", "--quote", help="Quote asset to filter."),
    limit: int = typer.Option(120, "--limit", min=1, help="Maximum ranked symbols to display."),
    include_non_crypto: bool = typer.Option(False, "--include-non-crypto", help="Include non-crypto underlyings such as equity perpetuals."),
    include_stable: bool = typer.Option(False, "--include-stable", help="Include stablecoin base assets."),
    include_leveraged: bool = typer.Option(False, "--include-leveraged", help="Include leveraged token style base assets."),
    comma: bool = typer.Option(False, "--comma", help="Print a comma-separated symbol list after the table."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    exchange = create_exchange(settings)
    fetcher = create_ohlcv_fetcher(exchange=exchange, settings=settings)
    rows = fetcher.ranked_usdt_perpetual_symbols(
        quote=quote,
        limit=limit,
        exclude_non_crypto=not include_non_crypto,
        exclude_stable=not include_stable,
        exclude_leveraged=not include_leveraged,
    )
    table = Table(title=f"Top {len(rows)} {settings.exchange_id} {quote} perpetual symbols by quote volume")
    for column in ["rank", "symbol", "quote_volume", "last", "percentage"]:
        table.add_column(column)
    for index, row in enumerate(rows, start=1):
        table.add_row(
            str(index),
            str(row["symbol"]),
            f"{row['quote_volume']:.2f}" if row.get("quote_volume") is not None else "",
            f"{row['last']:.8g}" if row.get("last") is not None else "",
            f"{row['percentage']:.4f}" if row.get("percentage") is not None else "",
        )
    console.print(table)
    if comma:
        console.print(",".join(str(row["symbol"]) for row in rows))


@data_app.command("fetch-ohlcv")
def fetch_ohlcv(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated CCXT symbols, e.g. BTC/USDT:USDT,ETH/USDT:USDT."),
    timeframes: str = typer.Option("1h,4h", "--timeframes", help="Comma-separated timeframes."),
    since: str = typer.Option(..., "--since", help="Inclusive start time, e.g. 2026-07-01 or 2026-07-01T00:00:00Z."),
    until: Optional[str] = typer.Option(None, "--until", help="Exclusive end time. Defaults to now."),
    limit_per_request: int = typer.Option(1000, "--limit-per-request", min=1, max=1500, help="Exchange page size."),
    include_unclosed: bool = typer.Option(False, "--include-unclosed", help="Keep the currently forming candle."),
    sandbox: bool = typer.Option(False, "--sandbox", help="Use exchange sandbox/testnet endpoints."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    exchange = create_exchange(settings, sandbox=sandbox)
    fetcher = create_ohlcv_fetcher(exchange=exchange, settings=settings)
    store = ParquetOHLCVStore(settings=settings)

    requested_symbols = _split_csv(symbols)
    requested_timeframes = _split_csv(timeframes)
    total_rows = 0
    for symbol in requested_symbols:
        for timeframe in requested_timeframes:
            frame = fetcher.fetch_ohlcv_frame(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                limit_per_request=limit_per_request,
                include_unclosed=include_unclosed,
            )
            written = store.upsert_ohlcv(frame)
            total_rows += written
            console.print(f"[green]stored[/green] {written} rows for {symbol} {timeframe}")

    catalog = store.build_catalog()
    store.write_catalog(catalog)
    store.write_quality_report()
    console.print(f"[bold green]done[/bold green]: processed {total_rows} rows")


@data_app.command("fetch-ranked-ohlcv")
def fetch_ranked_ohlcv(
    candidate_limit: int = typer.Option(120, "--candidate-limit", min=1, help="Number of top quote-volume symbols to fetch."),
    quote: str = typer.Option("USDT", "--quote", help="Quote asset to filter."),
    timeframes: str = typer.Option("1h,4h", "--timeframes", help="Comma-separated timeframes."),
    since: str = typer.Option(..., "--since", help="Inclusive start time, e.g. 2026-01-05."),
    until: Optional[str] = typer.Option(None, "--until", help="Exclusive end time. Defaults to now."),
    limit_per_request: int = typer.Option(1000, "--limit-per-request", min=1, max=1500, help="Exchange page size."),
    include_unclosed: bool = typer.Option(False, "--include-unclosed", help="Keep the currently forming candle."),
    include_non_crypto: bool = typer.Option(False, "--include-non-crypto", help="Include non-crypto underlyings such as equity perpetuals."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    exchange = create_exchange(settings)
    fetcher = create_ohlcv_fetcher(exchange=exchange, settings=settings)
    store = ParquetOHLCVStore(settings=settings)
    ranked = fetcher.ranked_usdt_perpetual_symbols(
        quote=quote,
        limit=candidate_limit,
        exclude_non_crypto=not include_non_crypto,
        exclude_stable=True,
        exclude_leveraged=True,
    )
    requested_symbols = [str(row["symbol"]) for row in ranked]
    requested_timeframes = _split_csv(timeframes)
    total_rows = 0
    console.print(f"Fetching {len(requested_symbols)} ranked symbols across {len(requested_timeframes)} timeframes.")
    for symbol_index, symbol in enumerate(requested_symbols, start=1):
        for timeframe in requested_timeframes:
            frame = fetcher.fetch_ohlcv_frame(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                limit_per_request=limit_per_request,
                include_unclosed=include_unclosed,
            )
            written = store.upsert_ohlcv(frame)
            total_rows += written
            console.print(f"[{symbol_index}/{len(requested_symbols)}] [green]stored[/green] {written} rows for {symbol} {timeframe}")

    catalog = store.build_catalog()
    store.write_catalog(catalog)
    store.write_quality_report()
    store.create_duckdb_database()
    console.print(f"[bold green]done[/bold green]: processed {total_rows} rows")


@data_app.command("fetch-universe-ohlcv")
def fetch_universe_ohlcv(
    since: str = typer.Option(..., "--since", help="Inclusive start time."),
    until: Optional[str] = typer.Option(None, "--until", help="Exclusive end time. Defaults to now."),
    timeframes: str = typer.Option("1h,4h", "--timeframes", help="Stored market timeframes."),
    limit_per_request: int = typer.Option(1000, "--limit-per-request", min=1, max=1500),
    include_unclosed: bool = typer.Option(False, "--include-unclosed"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    settings = _settings(config_file)
    store = ParquetOHLCVStore(settings=settings)
    universe = UniverseBuilder(settings=settings, store=store).read_latest()
    if universe.is_empty():
        raise FileNotFoundError("latest universe is missing; run `universe build` first")
    symbols = universe.filter(
        pl.col("included")
        & (pl.col("exchange") == settings.exchange_id)
        & (pl.col("market_type") == settings.market_type)
    ).get_column("symbol").unique().sort().to_list()
    if not symbols:
        raise ValueError("latest universe has no included symbols")
    requested_timeframes = _split_csv(timeframes)
    exchange = create_exchange(settings)
    fetcher = create_ohlcv_fetcher(exchange=exchange, settings=settings)
    total_rows = 0
    for symbol_index, symbol in enumerate(symbols, start=1):
        for timeframe in requested_timeframes:
            frame = fetcher.fetch_ohlcv_frame(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                limit_per_request=limit_per_request,
                include_unclosed=include_unclosed,
            )
            written = store.upsert_ohlcv(frame)
            total_rows += written
            console.print(f"[{symbol_index}/{len(symbols)}] [green]stored[/green] {written} rows for {symbol} {timeframe}")
    catalog = store.build_catalog()
    store.write_catalog(catalog)
    store.write_quality_report()
    store.create_duckdb_database()
    console.print(
        f"[bold green]universe OHLCV done[/bold green]: {len(symbols)} symbols, "
        f"timeframes={','.join(requested_timeframes)}, processed={total_rows} rows"
    )


@data_app.command("catalog")
def catalog(
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Refresh catalog before showing it."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    store = ParquetOHLCVStore(settings=settings)
    frame = store.build_catalog() if refresh else store.read_catalog()
    if refresh:
        store.write_catalog(frame)
        store.create_duckdb_database()

    table = Table(title="OHLCV Catalog")
    for column in frame.columns:
        table.add_column(column)
    for row in frame.iter_rows(named=True):
        table.add_row(*(str(row[column]) for column in frame.columns))
    console.print(table)


@data_app.command("quality")
def quality(
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    store = ParquetOHLCVStore(settings=settings)
    report_path = store.write_quality_report()
    console.print(f"Quality report written to [bold]{report_path}[/bold]")


@universe_app.command("build")
def build_universe(
    timeframes: str = typer.Option("1h,4h", "--timeframes", help="Comma-separated required timeframes."),
    tier: str = typer.Option("research", "--tier", help="Universe tier name: core, research, broad, or custom."),
    min_rows: int = typer.Option(2, "--min-rows", min=1, help="Fallback minimum rows required per timeframe."),
    min_days: int = typer.Option(180, "--min-days", min=1, help="Minimum history days required per timeframe."),
    liquidity_lookback_days: int = typer.Option(30, "--liquidity-lookback-days", min=1, help="Daily quote-volume lookback window."),
    liquidity_timeframe: str = typer.Option("1h", "--liquidity-timeframe", help="Timeframe used to compute daily quote volume."),
    min_median_daily_quote_volume: float = typer.Option(5_000_000.0, "--min-median-daily-quote-volume", min=0.0, help="Minimum median daily quote volume."),
    min_mean_daily_quote_volume: float = typer.Option(0.0, "--min-mean-daily-quote-volume", min=0.0, help="Minimum mean daily quote volume."),
    max_symbols: Optional[int] = typer.Option(120, "--max-symbols", min=1, help="Maximum included symbols by liquidity rank."),
    min_quote_volume: float = typer.Option(0.0, "--min-quote-volume", min=0.0, help="Legacy minimum average candle quote volume."),
    include_non_crypto: bool = typer.Option(False, "--include-non-crypto", help="Allow non-crypto underlyings into the universe."),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Optional timestamp cutoff for the snapshot."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    builder = UniverseBuilder(settings=settings)
    result = builder.build(
        timeframes=_split_csv(timeframes),
        min_rows=min_rows,
        min_quote_volume=min_quote_volume,
        min_days=min_days,
        liquidity_lookback_days=liquidity_lookback_days,
        liquidity_timeframe=liquidity_timeframe,
        min_median_daily_quote_volume=min_median_daily_quote_volume,
        min_mean_daily_quote_volume=min_mean_daily_quote_volume,
        max_symbols=max_symbols,
        tier=tier,
        exclude_non_crypto=not include_non_crypto,
        as_of=as_of,
    )
    ParquetOHLCVStore(settings=settings).create_duckdb_database()
    console.print(
        f"[green]universe built[/green]: {result.included_count}/{result.total_count} symbols included, "
        f"tier={result.tier}, snapshot={result.snapshot_id}"
    )
    console.print(f"snapshot: [bold]{result.path}[/bold]")
    console.print(f"latest:   [bold]{result.latest_path}[/bold]")


@universe_app.command("show")
def show_universe(
    included_only: bool = typer.Option(True, "--included-only/--all", help="Only show included symbols."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    frame = UniverseBuilder(settings=settings).read_latest()
    if included_only and not frame.is_empty():
        frame = frame.filter(frame["included"])

    table = Table(title="Latest Universe")
    columns = ["symbol", "included", "exclusion_reason", "available_timeframes", "rows_total", "mean_quote_volume"]
    for column in columns:
        table.add_column(column)
    for row in frame.select(columns).iter_rows(named=True):
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)


@labels_app.command("build")
def build_labels(
    timeframe: str = typer.Option("1h", "--timeframe", help="Source OHLCV timeframe."),
    horizons: str = typer.Option("1,4,8,24", "--horizons", help="Comma-separated forward horizons in bars."),
    volatility_lookback: int = typer.Option(24, "--volatility-lookback", min=2, help="Historical bars used for realized volatility."),
    universe_only: bool = typer.Option(True, "--universe-only/--all-symbols", help="Filter to latest included universe symbols."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    parsed_horizons = [int(item) for item in _split_csv(horizons)]
    result = LabelBuilder(settings=settings).build(
        timeframe=timeframe,
        horizons=parsed_horizons,
        volatility_lookback=volatility_lookback,
        universe_only=universe_only,
    )
    ParquetOHLCVStore(settings=settings).create_duckdb_database()
    console.print(
        f"[green]labels built[/green]: {result.rows} rows for timeframe={result.timeframe}, "
        f"horizons={','.join(str(item) for item in result.horizons)}"
    )
    console.print(f"labels: [bold]{result.path}[/bold]")
    console.print(f"latest: [bold]{result.latest_path}[/bold]")


@labels_app.command("build-hourly")
def build_hourly_labels(
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    _, result = HourlyMultiHorizonLabelBuilder(settings=settings).build()
    console.print(
        f"[green]hourly labels built[/green]: {result.rows} rows, {result.symbol_count} symbols, "
        f"horizons={','.join(str(item) for item in result.horizons)}h"
    )
    console.print(f"labels: [bold]{result.path}[/bold]")


@factors_app.command("build")
def build_factors(
    timeframes: str = typer.Option("1h,4h", "--timeframes", help="Comma-separated source OHLCV timeframes."),
    universe_only: bool = typer.Option(True, "--universe-only/--all-symbols", help="Filter to latest included universe symbols."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    parsed_timeframes = _split_csv(timeframes)
    result = FactorBuilder(settings=settings).build(timeframes=parsed_timeframes, universe_only=universe_only)
    ParquetOHLCVStore(settings=settings).create_duckdb_database()
    console.print(
        f"[green]factors built[/green]: {result.rows} rows, {result.factor_count} factors, "
        f"timeframes={','.join(result.timeframes)}"
    )
    console.print(f"factors: [bold]{result.path}[/bold]")
    console.print(f"latest:  [bold]{result.latest_path}[/bold]")


@factors_app.command("build-hourly")
def build_hourly_factors(
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    _, result = HourlyMultiTimeframeFactorBuilder(settings=settings).build()
    console.print(
        f"[green]hourly multi-timeframe factors built[/green]: {result.rows} rows, "
        f"{result.feature_count} features, {result.symbol_count} symbols"
    )
    console.print(f"features: [bold]{result.path}[/bold]")


@evaluation_app.command("single-factor")
def evaluate_single_factors(
    label_column: str = typer.Option("forward_return", "--label-column", help="Label column to evaluate against."),
    quantiles: int = typer.Option(5, "--quantiles", min=2, help="Number of factor groups for grouped return analysis."),
    min_pairs: int = typer.Option(2, "--min-pairs", min=2, help="Minimum cross-section size per timestamp."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    result = SingleFactorEvaluator(settings=settings).evaluate(
        label_column=label_column,
        quantiles=quantiles,
        min_pairs=min_pairs,
    )
    console.print(
        f"[green]single-factor evaluation done[/green]: {result.evaluated_factors} factors, "
        f"{result.evaluated_rows} joined rows"
    )
    console.print(f"summary:       [bold]{result.summary_path}[/bold]")
    console.print(f"timeseries:    [bold]{result.timeseries_path}[/bold]")
    console.print(f"group returns: [bold]{result.group_returns_path}[/bold]")
    console.print(f"report:        [bold]{result.report_path}[/bold]")


@evaluation_app.command("decorrelate")
def decorrelate_factors(
    max_value_corr: float = typer.Option(0.85, "--max-value-corr", min=0.0, max=1.0, help="Maximum allowed factor value correlation to selected factors."),
    max_ic_corr: float = typer.Option(0.85, "--max-ic-corr", min=0.0, max=1.0, help="Maximum allowed RankIC series correlation to selected factors."),
    min_abs_rank_ic: float = typer.Option(0.0, "--min-abs-rank-ic", min=0.0, help="Minimum absolute mean RankIC required before decorrelation."),
    min_samples: int = typer.Option(1, "--min-samples", min=1, help="Minimum sample count required before decorrelation."),
    max_factors_per_group: Optional[int] = typer.Option(None, "--max-factors-per-group", min=1, help="Optional cap per timeframe/horizon group."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    result = FactorDecorrelator(settings=settings).run(
        max_value_corr=max_value_corr,
        max_ic_corr=max_ic_corr,
        min_abs_rank_ic=min_abs_rank_ic,
        min_samples=min_samples,
        max_factors_per_group=max_factors_per_group,
    )
    console.print(
        f"[green]decorrelation done[/green]: {result.selected_count}/{result.candidate_count} candidates selected"
    )
    console.print(f"selected:  [bold]{result.selected_path}[/bold]")
    console.print(f"decisions: [bold]{result.decisions_path}[/bold]")
    console.print(f"report:    [bold]{result.report_path}[/bold]")


@models_app.command("train-hourly-rolling")
def train_hourly_rolling_strategy(
    train_fraction: float = typer.Option(0.70, "--train-fraction", min=0.1, max=0.9),
    valid_fraction: float = typer.Option(0.15, "--valid-fraction", min=0.05, max=0.5),
    embargo_hours: int = typer.Option(24, "--embargo-hours", min=24),
    round_trip_cost_rate: float = typer.Option(0.0014, "--round-trip-cost-rate", min=0.0),
    disagreement_penalty: float = typer.Option(0.25, "--disagreement-penalty", min=0.0),
    thresholds: str = typer.Option("0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.075,0.09,0.10,0.12,0.15", "--thresholds"),
    gross_exposure: float = typer.Option(0.90, "--gross-exposure", min=0.01, max=1.0),
    max_asset_weight: float = typer.Option(0.12, "--max-asset-weight", min=0.001, max=1.0),
    exit_threshold_ratio: float = typer.Option(0.70, "--exit-threshold-ratio", min=0.0, max=1.0),
    minimum_trade_weight: float = typer.Option(0.02, "--minimum-trade-weight", min=0.0, max=1.0),
    max_weight_step: float = typer.Option(0.04, "--max-weight-step", min=0.001, max=1.0),
    max_hourly_turnover: float = typer.Option(0.25, "--max-hourly-turnover", min=0.001, max=2.0),
    one_way_cost_rate: float = typer.Option(0.0007, "--one-way-cost-rate", min=0.0),
    rebuild_dataset: bool = typer.Option(True, "--rebuild-dataset/--reuse-dataset"),
    seed: int = typer.Option(42, "--seed"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    settings = _settings(config_file)
    research_config = HourlyRollingResearchConfig(
        train_fraction=train_fraction,
        valid_fraction=valid_fraction,
        embargo_hours=embargo_hours,
        round_trip_cost_rate=round_trip_cost_rate,
        disagreement_penalty=disagreement_penalty,
        threshold_grid=tuple(float(item) for item in _split_csv(thresholds)),
        seed=seed,
    )
    portfolio_rules = HourlyPortfolioRules(
        gross_exposure=gross_exposure,
        max_asset_weight=max_asset_weight,
        exit_threshold_ratio=exit_threshold_ratio,
        minimum_trade_weight=minimum_trade_weight,
        max_weight_step=max_weight_step,
        max_hourly_turnover=max_hourly_turnover,
        one_way_cost_rate=one_way_cost_rate,
    )
    result = HourlyRollingResearchRunner(
        settings=settings,
        config=research_config,
        portfolio_rules=portfolio_rules,
    ).run(rebuild_dataset=rebuild_dataset)
    console.print(f"[green]hourly rolling research done[/green]: {result.run_id}")
    console.print(f"chosen Score threshold: [bold]{result.chosen_entry_threshold:.4f}[/bold]")
    console.print(f"test total return: [bold]{result.test_metrics['total_return']:.2%}[/bold]")
    console.print(f"test max drawdown: [bold]{result.test_metrics['max_drawdown']:.2%}[/bold]")
    console.print(f"predictions: [bold]{result.predictions_path}[/bold]")
    console.print(f"portfolio:   [bold]{result.hourly_backtest_path}[/bold]")
    console.print(f"positions:   [bold]{result.positions_path}[/bold]")
    console.print(f"metrics:     [bold]{result.metrics_path}[/bold]")
    console.print(f"report:      [bold]{result.report_path}[/bold]")


@backtest_app.command("export-freqtrade-data")
def export_freqtrade_data(
    pairs_path: Optional[Path] = typer.Option(
        None,
        "--pairs-path",
        help="JSON pair list generated by export-freqtrade-signals.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Freqtrade user_data directory. Defaults to freqtrade/user_data.",
    ),
    timeframe: str = typer.Option("4h", "--timeframe", help="Trading candle timeframe."),
    mark_timeframe: str = typer.Option("1h", "--mark-timeframe", help="Local OHLCV timeframe used as mark-price proxy."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    result = FreqtradeOHLCVExporter(settings=settings).export(
        pairs_path=pairs_path,
        output_dir=output_dir,
        timeframe=timeframe,
        mark_timeframe=mark_timeframe,
    )
    console.print(
        f"[green]Freqtrade OHLCV exported[/green]: {result.pair_count} pairs, "
        f"{result.futures_rows} futures rows, {result.mark_rows} mark-proxy rows"
    )
    console.print(f"data: [bold]{result.data_dir}[/bold]")


@backtest_app.command("export-dashboard-data")
def export_dashboard_data(
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Freqtrade backtest result directory. Defaults to freqtrade/user_data/backtest_results.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Dashboard JSON path. Defaults to dashboard/public/data/backtests.json.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    result = FreqtradeDashboardExporter(settings=_settings(config_file)).export(
        results_dir=results_dir,
        output_path=output_path,
    )
    console.print(
        f"[green]dashboard data exported[/green]: {result.strategy_count} strategies, "
        f"{result.run_count} runs, {result.trade_count} trades"
    )
    console.print(f"output: [bold]{result.output_path}[/bold]")


@live_app.command("promote-model")
def promote_live_model(
    experiment_dir: Optional[Path] = typer.Option(
        None,
        "--experiment-dir",
        help="Validated walk-forward experiment directory. Defaults to the latest 4h/h6/forward_return run.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    source = experiment_dir
    if source is None:
        candidates = sorted(
            (settings.resolved_data_path() / "models" / "walk_forward").glob(
                "walk_forward_4h_h6_forward_return_*"
            )
        )
        if not candidates:
            raise FileNotFoundError("no validated 4h/h6/forward_return walk-forward experiment found")
        source = candidates[-1]
    result = ProductionModelPromoter(settings=settings).promote(source)
    console.print(
        f"[green]production model promoted[/green]: {result.model_count} ensemble members"
    )
    console.print(f"bundle:   [bold]{result.bundle_dir}[/bold]")
    console.print(f"manifest: [bold]{result.manifest_path}[/bold]")
    console.print(f"latest:   [bold]{result.latest_path}[/bold]")


@live_app.command("write-dry-run-config")
def write_live_dry_run_config(
    pairs_path: Optional[Path] = typer.Option(
        None,
        "--pairs-path",
        help="JSON pair list. Defaults to freqtrade/user_data/pairs_live.json.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Config output. Defaults to freqtrade/config_live_long_only_dryrun.json.",
    ),
    max_open_trades: int = typer.Option(9, "--max-open-trades", min=1),
    testnet_market_data: bool = typer.Option(
        True,
        "--testnet-market-data/--mainnet-market-data",
        help="Use Binance Futures Testnet public endpoints or Binance mainnet public endpoints.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    pairs = (
        pairs_path or settings.project_root / "freqtrade" / "user_data" / "pairs_live.json"
    ).resolve()
    output = (
        output_path
        or settings.project_root / "freqtrade" / "config_live_long_only_dryrun.json"
    ).resolve()
    result = write_live_long_only_config(
        pairs_path=pairs,
        output_path=output,
        max_open_trades=max_open_trades,
        use_testnet_market_data=testnet_market_data,
    )
    console.print(f"[green]live dry-run config written[/green]: [bold]{result}[/bold]")


@live_app.command("write-testnet-config")
def write_live_testnet_config(
    pairs_path: Path = typer.Option(..., "--pairs-path", help="JSON pair list for Testnet market metadata."),
    output_path: Path = typer.Option(..., "--output-path", help="Testnet-only public config path."),
    database_path: Path = typer.Option(..., "--database-path", help="Separate Testnet SQLite path."),
) -> None:
    result = write_testnet_config(
        pairs_path=pairs_path.resolve(),
        output_path=output_path.resolve(),
        database_path=database_path.resolve(),
    )
    console.print(f"[green]Testnet config written[/green]: [bold]{result}[/bold]")


@live_app.command("prepare-pairs")
def prepare_live_pairs(
    source_path: Optional[Path] = typer.Option(
        None,
        "--source-path",
        help="Research pair list. Defaults to freqtrade/user_data/pairs.json.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Live pair list output. Defaults to freqtrade/user_data/pairs_live.json.",
    ),
    sandbox: bool = typer.Option(
        True,
        "--sandbox/--mainnet",
        help="Validate pairs against Binance Futures Testnet or mainnet.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    source = (
        source_path or settings.project_root / "freqtrade" / "user_data" / "pairs.json"
    ).resolve()
    output = (
        output_path
        or settings.project_root / "freqtrade" / "user_data" / "pairs_live.json"
    ).resolve()
    requested = [str(pair) for pair in json.loads(source.read_text(encoding="utf-8"))]
    exchange = create_exchange(settings, sandbox=sandbox)
    markets = exchange.load_markets()
    active = sorted(
        pair
        for pair in requested
        if pair in markets and markets[pair].get("active") is not False
    )
    excluded = sorted(set(requested) - set(active))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[green]live pairs prepared[/green]: {len(active)} active / {len(requested)} requested"
    )
    if excluded:
        console.print(f"excluded: [yellow]{', '.join(excluded)}[/yellow]")
    console.print(f"output: [bold]{output}[/bold]")


@live_app.command("generate-signals")
def generate_live_signals(
    pairs_path: Optional[Path] = typer.Option(
        None,
        "--pairs-path",
        help="JSON pair list. Defaults to freqtrade/user_data/pairs_live.json.",
    ),
    signal_path: Optional[Path] = typer.Option(
        None,
        "--signal-path",
        help="Live signal parquet output.",
    ),
    status_path: Optional[Path] = typer.Option(
        None,
        "--status-path",
        help="Live signal health JSON output.",
    ),
    min_cross_section_count: int = typer.Option(
        60,
        "--min-cross-section-count",
        min=2,
    ),
    sandbox: bool = typer.Option(
        True,
        "--sandbox/--mainnet",
        help="Use Binance Futures Testnet or mainnet public market data.",
    ),
    force: bool = typer.Option(False, "--force", help="Regenerate an already healthy decision."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    signals_dir = settings.project_root / "freqtrade" / "user_data" / "signals"
    producer = LiveLongOnlySignalProducer(
        settings=settings,
        exchange=create_exchange(settings, sandbox=sandbox),
        pairs_path=(
            pairs_path or settings.project_root / "freqtrade" / "user_data" / "pairs_live.json"
        ).resolve(),
        signal_path=(
            signal_path or signals_dir / "alpha_live_long_only_signals.parquet"
        ).resolve(),
        status_path=(
            status_path or signals_dir / "alpha_live_long_only_status.json"
        ).resolve(),
        min_cross_section_count=min_cross_section_count,
    )
    result = producer.run(force=force)
    action = "already healthy" if result.skipped else "generated"
    console.print(
        f"[green]live signals {action}[/green]: {result.cross_section_count} pairs, "
        f"{result.selected_count} long"
    )
    console.print(f"selected: [bold]{', '.join(result.selected_pairs)}[/bold]")
    console.print(f"signals:  [bold]{result.signal_path}[/bold]")
    console.print(f"status:   [bold]{result.status_path}[/bold]")


@live_app.command("audit-dry-run")
def audit_live_dry_run(
    status_path: Optional[Path] = typer.Option(None, "--status-path", help="Live signal status JSON."),
    signal_path: Optional[Path] = typer.Option(None, "--signal-path", help="Live signal Parquet."),
    database_path: Optional[Path] = typer.Option(None, "--database-path", help="Freqtrade dry-run SQLite database."),
    report_path: Optional[Path] = typer.Option(None, "--report-path", help="Atomic safety audit JSON output."),
    grace_minutes: int = typer.Option(15, "--grace-minutes", min=0),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    user_data = settings.project_root / "freqtrade" / "user_data"
    report = audit_dry_run(
        status_path=(status_path or user_data / "signals" / "alpha_live_long_only_status.json").resolve(),
        signal_path=(signal_path or user_data / "signals" / "alpha_live_long_only_signals.parquet").resolve(),
        database_path=(database_path or user_data / "tradesv3.live-long-only.dryrun.sqlite").resolve(),
        grace_minutes=grace_minutes,
    )
    output = (report_path or user_data / "signals" / "alpha_live_safety_audit.json").resolve()
    report = publish_audit(report, output, webhook_url=os.environ.get("ALPHA_SAFETY_WEBHOOK_URL"))
    console.print(
        f"Safety audit: [bold]{report['state']}[/bold], "
        f"entry allowed: {report['entry_allowed']}, "
        f"issues: {', '.join(issue['code'] for issue in report['issues']) or 'none'}"
    )
    console.print(f"report: [bold]{output}[/bold]")
    if not report["entry_allowed"]:
        raise typer.Exit(code=2)
