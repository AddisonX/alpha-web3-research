from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from src import __version__
from src.config import Settings, load_settings
from src.data.exchange import create_exchange
from src.live.config import write_live_long_only_config
from src.live.reconciliation import audit_dry_run, publish_audit
from src.live.signals import LiveLongOnlySignalProducer
from src.live.paths import instance_directory
console = Console()
app = typer.Typer(help="Tokyo long-only dry-run runtime.")
config_app = typer.Typer(help="Inspect runtime settings.")
live_app = typer.Typer(help="Generate signals, configure dry-run and audit safety.")
app.add_typer(config_app, name="config")
app.add_typer(live_app, name="live")

def _settings(config_file: Optional[Path]) -> Settings:
    return load_settings(config_file)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show package version and exit."),
) -> None:
    if version:
        console.print(f"alpha-web3-research {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print("Alpha: Tokyo LightGBM long-only dry-run runtime.")
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


@live_app.command("write-dry-run-config")
def write_live_dry_run_config(
    pairs_path: Optional[Path] = typer.Option(
        None,
        "--pairs-path",
        help="JSON pair list. Defaults to the Tokyo local instance pairs.json.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Config output. Defaults to the Tokyo local instance config.json.",
    ),
    max_open_trades: int = typer.Option(9, "--max-open-trades", min=1),
    testnet_market_data: bool = typer.Option(
        False,
        "--testnet-market-data/--mainnet-market-data",
        help="Use Binance Futures Testnet public endpoints or Binance mainnet public endpoints.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    pairs = (
        pairs_path or instance_directory(settings.project_root) / "pairs.json"
    ).resolve()
    output = (
        output_path
        or instance_directory(settings.project_root) / "config.json"
    ).resolve()
    result = write_live_long_only_config(
        pairs_path=pairs,
        output_path=output,
        max_open_trades=max_open_trades,
        use_testnet_market_data=testnet_market_data,
    )
    console.print(f"[green]live dry-run config written[/green]: [bold]{result}[/bold]")


@live_app.command("prepare-pairs")
def prepare_live_pairs(
    source_path: Optional[Path] = typer.Option(
        None,
        "--source-path",
        help="Pair list to validate. Defaults to the Tokyo local instance pairs.json.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output-path",
        help="Validated pair list output. Defaults to the Tokyo local instance pairs.json.",
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox/--mainnet",
        help="Validate pairs against Binance Futures Testnet or mainnet.",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    source = (
        source_path or instance_directory(settings.project_root) / "pairs.json"
    ).resolve()
    output = (
        output_path
        or instance_directory(settings.project_root) / "pairs.json"
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
    strategy_id: Optional[str] = typer.Option(None, "--strategy", help="Registered strategy ID; defaults to ALPHA_LIVE_STRATEGY."),
    pairs_path: Optional[Path] = typer.Option(
        None,
        "--pairs-path",
        help="JSON pair list. Defaults to the selected strategy instance pairs.json.",
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
        False,
        "--sandbox/--mainnet",
        help="Use Binance Futures Testnet or mainnet public market data.",
    ),
    force: bool = typer.Option(False, "--force", help="Regenerate an already healthy decision."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional .env-style config file."),
) -> None:
    settings = _settings(config_file)
    import re

    selected_strategy = strategy_id or settings.live_strategy
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", selected_strategy):
        raise typer.BadParameter("strategy ID must be lowercase letters, digits, underscores or hyphens")
    selected_instance = instance_directory(settings.project_root, selected_strategy)
    signals_dir = selected_instance / "signals"
    producer = LiveLongOnlySignalProducer(
        settings=settings,
        exchange=create_exchange(settings, sandbox=sandbox),
        pairs_path=(
            pairs_path or selected_instance / "pairs.json"
        ).resolve(),
        signal_path=(
            signal_path or signals_dir / "alpha_live_long_only_signals.parquet"
        ).resolve(),
        status_path=(
            status_path or signals_dir / "alpha_live_long_only_status.json"
        ).resolve(),
        min_cross_section_count=min_cross_section_count,
        strategy_id=selected_strategy,
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
    user_data = instance_directory(settings.project_root, settings.live_strategy)
    report = audit_dry_run(
        status_path=(status_path or user_data / "signals" / "alpha_live_long_only_status.json").resolve(),
        signal_path=(signal_path or user_data / "signals" / "alpha_live_long_only_signals.parquet").resolve(),
        database_path=(database_path or user_data / "database" / "trades.dryrun.sqlite").resolve(),
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

