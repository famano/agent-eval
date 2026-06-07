# LLMエージェント評価基盤 設計書

## 1. 目的と背景

LLMエージェントは「ツール使用による外部状態の変更」が目的であることがあり、最終的なテキスト出力はその報告にすぎない場合がある。この場合、出力テキストだけを見ても達成度を測れず、評価が難しい。

一方で、**ツールを使ってファイルの作成・変更を行うエージェント**については、生成された出力ファイルを成果物として直接評価できる。本基盤は、この「出力ファイルを成果物とみなす」エージェント群を対象に、再現性のある自動評価を行う仕組みを提供する。

特定ドメインに限定しない汎用基盤として設計するが、契約レビュー・デューデリジェンス等の文書生成エージェントを典型的なユースケースとして想定する。

## 2. 設計思想・前提

- **入力の不変性**: 評価用データセットは read-only ディレクトリとして与え、エージェントが入力を破壊できないようにする。
- **出力ファイルベース評価**: 評価は出力ディレクトリに生成されたファイルに対して行う。
- **リファレンス＋観点（criterion）併用**: 完全一致ではなく、正解ファイルから抽出した観点リストへの充足度で採点する。これによりタスクに複数の正解があってもよい。
- **評価器自身がノイズ源であることを前提にする**: 評価器もLLMであり非決定的なので、評価器の分散をエージェントの分散と区別し、評価器の妥当性を担保する仕組み（キャリブレーション・観点ごと独立採点）を組み込む。

### 既知の限界（明示しておく前提）

- 出力ファイルのみを見るため、出力が正しくても**到達プロセスが誤り／偶然**だった場合を検出できない。デバッグのため軌跡（trajectory）は別途ログするが、採点には用いない。
- 外部ツール（Web検索等）を使うエージェントは run 間で完全再現できない。再現性は「環境・パラメータの固定」と「統計的な反復」で担保する。
- LLM評価器はドメインの正誤を独力で判断しきれない。誤り検出は「正解ファイルとの矛盾」「正解にもデータにも無い断定」を基準にすることで、判断を裏付け可能な範囲に限定する。

## 3. 全体アーキテクチャ

```
Dataset (read-only input + reference + criteria)
        │
        ▼
   ┌─────────┐   run(input_dir, output_dir)   ┌──────────────┐
   │ Iterator │ ─────────────────────────────▶ │    Agent     │
   │ /Runner  │ ◀───────── RunResult ────────── │ (具象実装)    │
   └─────────┘                                  └──────────────┘
        │   output_files
        ▼
   ┌──────────────┐  evaluate(output, reference, criteria)
   │  Evaluator   │  （観点ごと独立採点 + 誤り分類）
   │ (LLM judge)  │ ─────────▶ EvaluationResult
   └──────────────┘
        │
        ▼
   生データ（観点別判定・誤り一覧） + 集計結果（統計）
```

データフロー:
1. Iterator が Dataset を1件取り、隔離した出力ディレクトリを用意して Agent.run を呼ぶ。
2. RunResult の status を見て、インフラ失敗ならリトライ／除外、正常終了なら Evaluator へ。
3. Evaluator が出力ファイル・正解ファイル・観点リストを入力に EvaluationResult を返す。
4. これを規定回数 N 反復し、生データと統計集計を出力する。

## 4. データモデル

```python
# ---- 観点（評価基準）----
@dataclass
class Criterion:
    id: str
    description: str                     # 「xxという論点に言及し、xxと検討、xxと結論」
    importance: Literal["must", "should"]
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)   # 任意。分野等のスライス用ラベル

# ---- データセット ----
@dataclass
class Dataset:
    id: str
    input_dir: Path                      # read-only
    reference_files: list[Path]          # 人手作成。複数許容（複数ファイル成果物）
    criteria: list[Criterion]            # LLM生成 → 人手キュレーション済
    timeout_s: int = 600
    budget_usd: float = 1.0

# ---- 実行結果 ----
class RunStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    CRASHED = "crashed"                  # インフラ起因。中身の0点とは区別

@dataclass
class RunResult:
    status: RunStatus
    output_files: list[Path]
    cost_usd: float
    latency_s: float
    tokens: int
    tool_calls: int
    trajectory_ref: Path | None = None   # デバッグ用ログへの参照（採点には未使用）

# ---- 評価結果 ----
class Verdict(Enum):
    MET = "met"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    CONTRADICTED = "contradicted"        # 正解と矛盾＝充足の逆方向

@dataclass
class CriterionResult:
    criterion_id: str
    verdict: Verdict
    rationale: str                       # judge の根拠（監査用）

@dataclass
class EvalError:
    type: Literal["contradiction", "unsupported", "format", "extra_neutral"]
    severity: Literal["critical", "major", "minor"]
    description: str

@dataclass
class EvaluationResult:
    dataset_id: str
    run_index: int
    criterion_results: list[CriterionResult]
    errors: list[EvalError]
    # 派生指標は集計時に算出（coverage, critical_error_count 等）
```

### 誤り（error）の型と方針

| type | 意味 | 採点上の扱い |
|------|------|--------------|
| `contradiction` | 正解ファイルと矛盾する記述 | 重大。severity に応じ減点 |
| `unsupported` | 正解にもデータにも裏付けの無い断定（ハルシネーション） | 重大度に応じ減点 |
| `format` | 出力先・形式が指定外、ファイル欠落 | 場合により0点扱い |
| `extra_neutral` | 観点外だが正しい／無害な追加記述 | **原則ペナルティなし**（網羅性を罰しない） |

「不要項目」を一律に罰すると thoroughness を不当に減点するため、無害な追加は中立扱いとし、減点対象は矛盾・無裏付け・形式違反に限定する。

## 5. コンポーネント仕様

### 5.1 Agent

```python
class Agent(Protocol):
    def metadata(self) -> AgentMetadata: ...
    def run(self, input_dir: Path, output_dir: Path,
            timeout_s: int, budget_usd: float) -> RunResult: ...
```

- `AgentMetadata`: model, tools, skills, temperature, seed, sdk_version 等。**再現性のため version/seed/temperature を必ず固定・記録**する。
- `run` は read-only の `input_dir` を読み、`output_dir`（run ごとに新規作成・隔離）へ出力する。
- 戻り値は単なる終了サインではなく `RunStatus` を含む `RunResult`。タイムアウト・予算超過・クラッシュを区別する。
- モデルやツール構成で実行方法が異なるため `run` は具象クラスに委ねるインターフェイス。

### 5.2 Evaluator（LLM judge）

評価器の信頼性が基盤全体の妥当性を決めるため、以下を必須仕様とする。

```python
class Evaluator(Protocol):
    def evaluate(self, output_files: list[Path],
                 reference_files: list[Path],
                 criteria: list[Criterion]) -> EvaluationResult: ...
    def calibrate(self, dataset: Dataset) -> CalibrationResult: ...
```

**(a) 観点ごとの独立採点**
「N個中いくつ満たすか」を1回で数えさせない。criterion 1個ごとに judge を呼び、構造化出力で `verdict` と `rationale` を返させる。一括カウントより安定し、根拠が監査可能になる。

**(b) リファレンスを judge に渡す**
誤り（特に `contradiction`）の検出には正解ファイルが必要。judge には「出力ファイル＋正解ファイル＋当該 criterion」を渡す。

**(c) キャリブレーション（ゲート）**
`calibrate` は judge に**正解ファイルそのもの**を採点させ、全 criterion がほぼ MET・critical error が 0 になることを確認する。満点近くにならなければ、criterion が曖昧か judge が壊れているサインなので、本番評価の前段ゲートとして失敗させる／警告する。

**(d) judge の選定とサンプリング**
- judge は被験エージェントより強い（できれば別系列の）モデルを用い、self-enhancement bias を避ける。
- 重要 criterion は judge を複数サンプルして多数決を取り、評価器分散を低減する。

**(e) 出力ファイルの正規化**
docx / pdf 等はテキスト抽出を経て judge に渡す。複数ファイルは結合または役割別に提示。ファイル探索ロジック（どの拡張子・どの場所を成果物とみなすか）を明示する。形式・場所が指定外なら `format` error として扱う。

### 5.3 データセットと観点生成

- 各データセット = 入力ディレクトリ ＋ 正解ファイル（人手作成）＋ 観点リスト。
- 観点は LLM であらかじめ生成するが、**生成しっぱなしにせず人手で一度キュレーション**する。基準:
  - **原子性**: 1観点 = 1判定単位。
  - **独立性**: 観点同士が重複しない。
  - **検証可能性**: 出力を見て met/not_met を判定できる。
  - **解法非依存**: 特定の言い回し・構成ではなく「論点・検討・結論の有無」を問う（複数の正解を許容するため）。
- `tags` は任意。分野別・観点種別などのスライス集計が要るときだけ付与する（汎用基盤として固定タクソノミーは強制しない）。

### 5.4 Iterator / Runner

```python
class Iterator:
    def run(self, dataset: Dataset, agent: Agent,
            evaluator: Evaluator, n_repeats: int,
            max_infra_retries: int = 2) -> DatasetReport: ...
```

- データセットごとに N 回反復。各 run で出力ディレクトリを新規作成し、run 間のファイル汚染を防ぐ。
- `RunStatus` で分岐:
  - `CRASHED`（インフラ起因）→ `max_infra_retries` までリトライ。それでも失敗なら**集計から除外**し別枠で記録（0点として平均に混ぜない）。
  - `TIMEOUT` / `BUDGET_EXCEEDED` → タスク失敗として記録（中身の評価対象に含めるかは方針で選択）。
  - `SUCCESS` → Evaluator へ。
- 出力は「生データ（全 run の CriterionResult・EvalError）」と「統計集計」の両方。

## 6. スコアリングと集計

スコアを1次元に潰さず、**網羅性と健全性の2軸**で報告する。

### 6.1 1 run あたりの派生指標

- **Coverage（recall的）**: 重み付き充足率
  ```
  coverage = Σ_i w_i · s_i  /  Σ_i w_i
  s_i = 1.0 (MET), 0.5 (PARTIAL), 0.0 (NOT_MET), 0.0 (CONTRADICTED)
  ```
  `must` 観点には大きい weight を割り当て、結論の誤りと論点漏れを同列にしない。
- **健全性（precision的）**: severity 別 error 件数。特に `critical` error の有無を独立指標として出す（critical が1件でもあれば成果物として不可、という閾値運用が可能）。
- 安易に F値1個へ畳まない。畳む場合は重みを明示的に外出しする。

### 6.2 反復集計（統計）

平均値だけでは bimodal（時々満点・時々失敗）を隠すため、以下を出す。

- coverage の **平均・標準偏差・bootstrap 信頼区間**。
- **pass@k**（k 回中1回でも閾値超え）と、必要なら全 run 成功率。
- critical error 発生率。
- コスト・レイテンシ・tool_calls の分布（コストは独立の評価軸）。
- 反復回数 N は観測分散から正当化する（分散が大きいタスクほど N を増やす）。

### 6.3 報告の粒度

- まず **データセット別**に集計し、その後に全体を出す。構成が不均一なデータセットを単純平均すると誤導するため、データセット間の単純平均は補助指標に留める。
- `tags` がある場合はタグ別スライスを出し、systematic な弱点（特定種類の観点で繰り返し落とす等）を可視化する。

## 7. 実行・再現性・隔離

- 入力 read-only、出力 run ごと隔離。
- `AgentMetadata` に model version / seed / temperature / sdk version を固定記録。
- 全 run の trajectory・コスト・トークン・レイテンシをログ（採点外、デバッグ／分析用）。
- judge のモデル・プロンプト・サンプリング設定もメタデータとして記録し、評価器側の変更が結果に与えた影響を追えるようにする。

## 8. 実装優先度

評価値の信頼性を最も左右する順に着手する。

1. **Evaluator のキャリブレーション ＋ 観点ごと独立採点**（リファレンスを judge に渡す）。
2. **error の型・severity 分離**（無害な追加を罰しない）。
3. **RunStatus によるインフラ失敗と0点の分離**、run 隔離。
4. **2軸スコア（coverage / 健全性）と統計集計**（CI・pass@k・分散）。
5. trajectory・コスト等の観測ログ整備。

1・2 が固まる前に反復回数やスコア合成を作り込むと、ノイズの上に集計を重ねることになるため、評価器の妥当性を先に担保する。

## 9. 用語

- **観点 / criterion**: 正解ファイルに含まれるべき具体的項目（論点への言及・検討・結論）。採点の最小単位。
- **キャリブレーション**: 正解ファイル自身を採点させ、評価器・観点の妥当性を確認する前段ゲート。
- **網羅性 / coverage**: 重み付き充足率（recall 的指標）。
- **健全性**: 矛盾・無裏付け・形式違反など誤りの少なさ（precision 的指標）。
