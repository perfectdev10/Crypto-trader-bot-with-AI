# Crypto Trader Bot with AI (v1.05)

## Overview
This project combines machine-learning price forecasting with automated trade execution and conversational control surfaces. It supports:
- Crypto trading with Binance market data, Capital.com execution, and a Telegram assistant.
- Index option trading on Zerodha Kite via a modular, agent-driven orchestrator.

Both tracks share a focus on explainable automation, configurable risk controls, and paper/live mode flexibility.

## Key Features
- **AI Forecasting**: PyTorch LSTM predicts directional moves for ETH/SOL/BTC using recent Binance OHLC data.
- **Human-in-the-loop Execution**: Telegram inline keyboards require explicit confirmation before every order; auto-close prompts are issued when counter signals appear.
- **Broker Integrations**: Capital.com (crypto) and Zerodha Kite (index options) connectors with dry-run fallbacks.
- **Strategy Intelligence Layer**: Sentiment analysis, market-context tagging, and optional RAG-augmented LangGraph agent choose the intraday playbook.
- **Trade Logging & Reporting**: SQLite persistence, backtesting hooks, and scheduled reporting stubs keep performance auditable.

## Architecture Snapshot
- `main.py` — End-to-end crypto bot: loads model, fetches Binance data, produces signals, mediates Telegram UX, and dispatches to Capital.com.
- `trading_bot.py` — Zerodha orchestrator that authenticates, selects strategies, and coordinates order and position agents.
- `agents.py` — Order execution (isolated worker pattern) and position management (stop-loss, trailing-stop logic).
- `strategy_factory.py`, `indicators.py`, `indicator_calculator.py` — Technical indicator calculators and strategy registry.
- `langgraph_agent.py`, `sentiment_agent.py`, `market_context.py` — Market-intel inputs for the LangGraph-driven strategy selector.
- `backtester.py`, `reporting.py` — Offline evaluation and reporting utilities.
- `state.json`, `output/` (if present) — Persisted runtime state and generated reports/backtests.

> Some modules referenced in the codebase (e.g., `rag_service.py`) are optional or may be supplied privately. Stub them if you plan to run the full workflow.

## Setup
1. **Python Environment**
   - `pip install -r requirements.txt`
   - GPU support is optional; CPU works for inference.
2. **Secrets**
   - Create a `.env` file holding Binance, Capital.com, and Telegram credentials for `main.py`.
   - Populate `config.yaml` with Zerodha keys, trading flags, and strategy settings for `trading_bot.py`.
3. **Model Artifact**
   - Place `price_predictor.pt` in the project root or retrain/export using your own pipeline.
4. **Database**
   - The crypto bot autogenerates `trades.db`; ensure the process has write access.

## Telegram Workflow (Crypto Bot)
- `/show` — Run AI analysis and receive BUY/SELL/HOLD suggestions with inline actions.
- `/status` — Summarize the most recent open position and P&L.
- `/close` — Close the active trade (requires stored `deal_id`).
- `/capital` — Display Capital.com connection diagnostics.
- `/symbols` — List common ETH epic formats for Capital.com.
- `/test` — Run a health check across data, model, broker, and Telegram subsystems.

## Configuration Highlights
Default trading flags (editable via `.env` or `config.yaml`):
- Underlying instrument, timeframe, and lot sizing.
- Risk per trade, stop-loss thresholds, and maximum trades per day.
- Paper trading toggles and natural-language prompt overrides.
- RAG controls (minimum trading days, enable/disable retrieval augmentation).

## Safety & Disclaimer
Algorithmic trading carries significant financial risk. Use paper/demo modes until you validate strategies, confirm broker credentials, and stress test failure paths. The authors assume no responsibility for trading outcomes.

Developer: [@Manokil](https://t.me/draviener)
