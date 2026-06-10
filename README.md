# AgentEval

LLM エージェントの出力品質を自動評価するフレームワークです。エージェントを N 回実行し、LLM をジャッジとして各実行結果を評価・集計します。

## 概要

```
Dataset (入力 + 評価基準) → Iterator (N回実行) → LLMEvaluator (判定) → DatasetStats (集計)
```

- **Coverage (カバレッジ)**: 評価基準の達成率（重み付き）
- **Soundness (健全性)**: 事実誤り・フォーマットエラーの数
- **pass@k**: k 回中少なくとも 1 回成功する確率（統計的推定）

## インストール

Python 3.11+ が必要です。

```bash
pip install -e .
```

開発用依存関係（ruff, mypy, pytest など）:

```bash
pip install -e ".[dev]"
```

オプション機能（`.docx` / `.pdf` の読み取り）:

```bash
pip install -e ".[docx,pdf]"
```

## クイックスタート

`example.py` に動作可能なサンプルがあります。

```bash
ANTHROPIC_API_KEY=your_key python example.py
```

実行には `ANTHROPIC_API_KEY` 環境変数が必要です。

## 使い方

### 1. エージェントを実装する

`Agent` プロトコルを満たすクラスを作成します。

```python
from pathlib import Path
from agent_eval import Agent, AgentMetadata, RunResult, RunStatus

class MyAgent:
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(model="claude-sonnet-4-6", sdk_version="1.0.0")

    def run(self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float) -> RunResult:
        # エージェントの処理をここに実装
        output_file = output_dir / "report.txt"
        output_file.write_text("...")
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[output_file],
            cost_usd=0.05,
            latency_s=3.0,
            tokens=500,
            tool_calls=10,
        )
```

### 2. データセットと評価基準を定義する

```python
from pathlib import Path
from agent_eval import Criterion, Dataset

dataset = Dataset(
    id="my-task-001",
    input_dir=Path("data/input"),
    reference_files=[Path("data/reference/expected.txt")],
    criteria=[
        Criterion(id="c1", description="必須条件の説明", importance="must", weight=2.0, tags=["category"]),
        Criterion(id="c2", description="推奨条件の説明", importance="should", weight=1.0),
    ],
    timeout_s=300,
    budget_usd=1.0,
)
```

### 3. 評価を実行する

```python
from agent_eval import Iterator, LLMEvaluator, SuiteReport, aggregate_dataset

evaluator = LLMEvaluator(model="claude-opus-4-7", n_samples=1)

# キャリブレーション（リファレンス自身が基準を満たすか確認）
cal = evaluator.calibrate(dataset)
if not cal.passed:
    raise RuntimeError(f"Calibration failed: {cal.message}")

runner = Iterator(output_root=Path("runs"))
report = runner.run(dataset=dataset, agent=MyAgent(), evaluator=evaluator, n_repeats=5)

stats = aggregate_dataset(report, dataset, coverage_threshold=0.7)
SuiteReport(dataset_stats=[stats]).print_summary()
```

## リポジトリ構成

```
agent_eval/
  models.py      # データクラス (Criterion, Dataset, RunResult, EvaluationResult など)
  agent.py       # Agent プロトコル定義
  llm.py         # LLMClient プロトコルと AnthropicClient 実装
  evaluator.py   # LLMEvaluator (ジャッジ + キャリブレーション)
  runner.py      # Iterator (N回実行のオーケストレーション)
  scoring.py     # メトリクス集計 (coverage, pass@k, bootstrap CI)
tests/           # pytest テスト
example.py       # 動作サンプル
```

## カスタム LLM クライアント

`LLMClient` プロトコルを実装することで任意の LLM を利用できます。

```python
class MyLLMClient:
    def complete(self, prompt: str) -> str:
        # 独自の LLM 呼び出しを実装
        return call_my_llm(prompt)

evaluator = LLMEvaluator(client=MyLLMClient())
```

構造化出力（tool use）をサポートする場合は `StructuredLLMClient` プロトコルも実装してください。

## 開発

```bash
# フォーマット
ruff format .

# リント
ruff check .

# 型チェック
mypy agent_eval

# セキュリティスキャン
bandit -c pyproject.toml -r agent_eval

# テスト
pytest --cov=agent_eval -v
```

CI は GitHub Actions で実行されます（`.github/workflows/ci.yml`）。

## ライセンス

[LICENSE](LICENSE) を参照してください。
