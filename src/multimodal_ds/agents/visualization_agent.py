"""
Visualization Agent — generates a production-grade Plotly chart gallery.

Chart suite (auto-selected based on data shape):
  1. data_quality        — missing value bar chart (always generated)
  2. distributions       — histogram grid for all numeric columns
  3. correlation_heatmap — Pearson correlation matrix (≥2 numeric cols)
  4. target_analysis     — class balance + box plots (binary/categorical target)
  5. scatter_matrix      — pair plot coloured by target (≥50 rows)
  6. feature_importance  — bar chart from feature_importance.csv if present
  7. roc_curve           — Logistic Regression baseline ROC (binary target, sklearn)

Each chart:
  - Saved as .html (self-contained Plotly interactive file)
  - Gets an LLM-generated narrative paragraph via Ollama
  - Is registered in ChartManifest with type, filename, title, narrative, data_shape

Message bus integration:
  - Publishes VIZ_REQUEST  at the start of generate()
  - Publishes VIZ_COMPLETE at the end with chart_count in payload

Graceful degradation:
  - _PLOTLY_AVAILABLE flag — if plotly isn't installed, generate() returns
    an empty manifest without raising.
  - All individual chart methods are wrapped in try/except so one failing
    chart never aborts the entire gallery.
  - Ollama narrative fallback — if LLM is unreachable, a rule-based string
    is used instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

from multimodal_ds.config import OUTPUT_DIR, REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT

import pandas as pd


from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("[VizAgent] plotly not installed — visualization disabled")


# ══════════════════════════════════════════════════════════════════════════
#  ChartManifest
# ══════════════════════════════════════════════════════════════════════════

class ChartManifest:
    """Registry of all charts generated in a session.

    Charts are stored as plain dicts so the manifest is trivially JSON-serialisable
    and survives LangGraph checkpoint serialisation.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.charts: List[Dict[str, Any]] = []

    def add(self, chart_type: str, filename: str, title: str, narrative: str, data_shape: tuple) -> None:
        self.charts.append({
            "chart_type": chart_type,
            "filename":   filename,
            "title":      title,
            "narrative": narrative,
            "data_shape": list(data_shape),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "chart_count": len(self.charts),
            "charts":      self.charts,
        }

    def save(self, output_dir: Path) -> Path:
        path = Path(output_dir) / "chart_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ══════════════════════════════════════════════════════════════════════════
#  VisualizationAgent
# ══════════════════════════════════════════════════════════════════════════

class VisualizationAgent:
    """Generates a standard Plotly chart gallery for any tabular dataset.

    Usage:
        agent = VisualizationAgent(session_id="abc123")
        manifest = agent.generate(df=df, target_col="churn")
        print(manifest.to_dict())
    """

    def __init__(self, session_id: str, working_dir: Optional[str] = None):
        self.session_id = session_id
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, df: pd.DataFrame, target_col: Optional[str] = None) -> ChartManifest:
        manifest = ChartManifest(session_id=self.session_id)

        if not _PLOTLY_AVAILABLE:
            logger.warning("[VizAgent] plotly not available — returning empty manifest")
            return manifest

        if df is None or df.empty:
            logger.warning("[VizAgent] Empty dataframe — returning empty manifest")
            return manifest

        # Publish VIZ_REQUEST event
        try:
            bus = get_bus()
            bus.publish(AgentMessage(
                msg_type=MessageType.VIZ_REQUEST,
                payload={"session_id": self.session_id, "target_col": target_col},
                sender="visualization_agent",
                session_id=self.session_id,
            ))
        except Exception as e:
            logger.warning(f"[VizAgent] failed to publish VIZ_REQUEST: {e}")

        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        self._chart_missing_values(df, manifest)
        # 2. Distributions
        if numeric_cols:
            self._chart_distributions(df, numeric_cols, target_col, manifest)
        # 3. Correlation heatmap
        if len(numeric_cols) >= 2:
            self._chart_correlation_heatmap(df, numeric_cols, manifest)
        # 4. Target analysis
        if target_col and target_col in df.columns:
            self._chart_target_analysis(df, target_col, numeric_cols, manifest)
        # 5. Scatter matrix
        if len(df) >= 50 and len(numeric_cols) >= 2:
            self._chart_scatter_matrix(df, numeric_cols, target_col, manifest)
        # 6. Feature importance
        fi_path = self._find_feature_importance()
        if fi_path:
            self._chart_feature_importance(fi_path, manifest)
        # 7. ROC curve
        if target_col and target_col in df.columns:
            self._chart_roc_curve(df, target_col, numeric_cols, manifest)

        # Save manifest before publishing completion event
        manifest.save(self.working_dir)
        # Publish VIZ_COMPLETE event
        try:
            bus = get_bus()
            bus.publish(AgentMessage(
                msg_type=MessageType.VIZ_COMPLETE,
                payload={"session_id": self.session_id, "chart_count": len(manifest.charts)},
                sender="visualization_agent",
                session_id=self.session_id,
            ))
        except Exception as e:
            logger.warning(f"[VizAgent] failed to publish VIZ_COMPLETE: {e}")
        return manifest

    def _get_narrative(self, chart_type: str, stats_summary: str) -> str:
        """Generate a one-sentence LLM narrative for a chart.

        Uses a short connect timeout (5s) for fast-fail when Ollama is down,
        and a separate read timeout (25s) for the actual generation. The
        original bare integer timeout=15 set ALL four httpx timeout categories
        to 15 seconds total — on a loaded machine where Ollama takes 3s to
        connect and 13s to generate a 200-token response, the request times
        out at second 15 and falls back to the stats_summary string silently.

        Narratives are best-effort: any exception returns stats_summary as
        the fallback so chart generation never blocks on LLM availability.
        """
        try:
            model  = REVIEWER_MODEL.replace("ollama/", "")
            prompt = f"In one sentence, describe this chart: {chart_type}. Stats: {stats_summary}"
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Generate a concise chart description."},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": 200, "temperature": 0.1},
                },
                timeout=httpx.Timeout(
                    connect=5.0,    # fast-fail if Ollama is not running
                    read=25.0,      # 200 tokens at ~8 tok/s = ~25s worst case
                    write=10.0,
                    pool=5.0,
                ),
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return content or stats_summary
        except Exception:
            pass
        return stats_summary
    def _chart_missing_values(self, df: pd.DataFrame, manifest: ChartManifest) -> None:
        try:
            # Compute missing counts per column
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            # Create bar chart
            fig = px.bar(x=missing_counts.index, y=missing_counts.values,
                         labels={"x": "Column", "y": "Missing Count"})
            title = "Data Quality — Missing Values"
            fig.update_layout(template="plotly_dark", title=title)
            # Save chart
            path = self.working_dir / "data_quality.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            # Narrative
            narrative = f"{total_missing} total missing values across {len(df.columns)} columns"
            narrative = self._get_narrative("data_quality", narrative)
            manifest.add("data_quality", "data_quality.html", "Data Quality", narrative, df.shape)
        except Exception as e:
            logger.warning(f"[VizAgent] data_quality failed: {e}")

    def _chart_distributions(self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str], manifest: ChartManifest) -> None:
        try:
            # Determine number of columns (max 12)
            cols_to_plot = numeric_cols[:12]
            n = len(cols_to_plot)
            # Create subplots grid (2 rows, ceil(n/2) cols) for readability
            rows = 1 if n <= 1 else 2
            cols = n if n <= 2 else (n + 1) // 2
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=cols_to_plot)
            for idx, col in enumerate(cols_to_plot):
                row = idx // cols + 1
                col_pos = idx % cols + 1
                if target_col and target_col in df.columns and df[target_col].nunique() == 2:
                    hist = px.histogram(df, x=col, color=target_col, nbins=30)
                else:
                    hist = px.histogram(df, x=col, nbins=30)
                for trace in hist.data:
                    fig.add_trace(trace, row=row, col=col_pos)
            title = "Feature Distributions"
            fig.update_layout(template="plotly_dark", title=title, showlegend=False)
            path = self.working_dir / "distributions.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            narrative = f"Distributions of {len(numeric_cols)} numeric features"
            narrative = self._get_narrative("distributions", narrative)
            manifest.add("distributions", "distributions.html", "Feature Distributions", narrative, df.shape)
        except Exception as e:
            logger.warning(f"[VizAgent] distributions failed: {e}")

    def _chart_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str], manifest: ChartManifest) -> None:
        try:
            if len(numeric_cols) < 2:
                return
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu", zmin=-1, zmax=1)
            title = "Pearson Correlation Matrix"
            fig.update_layout(template="plotly_dark", title=title)
            path = self.working_dir / "correlation_heatmap.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            # Count strong correlations (|r| > 0.7) excluding diagonal
            import numpy as np
            upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
            strong = np.abs(corr.where(upper)).values > 0.7
            n_strong = int(strong.sum())
            narrative = f"{n_strong} strong correlations detected (|r| > 0.7)"
            narrative = self._get_narrative("correlation_heatmap", narrative)
            manifest.add("correlation_heatmap", "correlation_heatmap.html", "Correlation Heatmap", narrative, df.shape)
        except Exception as e:
            logger.warning(f"[VizAgent] correlation_heatmap failed: {e}")

    def _chart_target_analysis(self, df: pd.DataFrame, target_col: str, numeric_cols: List[str], manifest: ChartManifest) -> None:
        try:
            if target_col not in df.columns:
                return
            # Left: class balance bar chart
            balance_counts = df[target_col].value_counts().reset_index()
            balance_counts.columns = ["target", "count"]
            fig_left = px.bar(balance_counts, x="target", y="count",
                                labels={"target": target_col, "count": "Count"})
            # Right: box plot of first numeric column vs target
            if numeric_cols:
                first_num = numeric_cols[0]
                fig_right = px.box(df, x=target_col, y=first_num,
                                   labels={target_col: target_col, first_num: first_num})
            else:
                fig_right = go.Figure()
            # Combine subplots
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Class Balance", f"{first_num} by {target_col}" if numeric_cols else ""] )
            for trace in fig_left.data:
                fig.add_trace(trace, row=1, col=1)
            for trace in fig_right.data:
                fig.add_trace(trace, row=1, col=2)
            title = "Target Analysis"
            fig.update_layout(template="plotly_dark", title=title)
            path = self.working_dir / "target_analysis.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            narrative = f"Target '{target_col}' class distribution and feature separation"
            narrative = self._get_narrative("target_analysis", narrative)
            manifest.add("target_analysis", "target_analysis.html", "Target Analysis", narrative, df.shape)
        except Exception as e:
            logger.warning(f"[VizAgent] target_analysis failed: {e}")

    def _chart_scatter_matrix(self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str], manifest: ChartManifest) -> None:
        try:
            cols = numeric_cols[:5]
            if not cols:
                return
            color = df[target_col] if target_col and target_col in df.columns else None
            fig = px.scatter_matrix(df[cols], dimensions=cols, color=color)
            title = "Scatter Matrix"
            fig.update_layout(template="plotly_dark", title=title)
            path = self.working_dir / "scatter_matrix.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            narrative = f"Pairwise relationships across {len(cols)} features"
            narrative = self._get_narrative("scatter_matrix", narrative)
            manifest.add("scatter_matrix", "scatter_matrix.html", "Scatter Matrix", narrative, df.shape)
        except Exception as e:
            logger.warning(f"[VizAgent] scatter_matrix failed: {e}")

    def _find_feature_importance(self) -> Optional[Dict[str, float]]:
        candidate = self.working_dir / "feature_importance.csv"
        if not candidate.exists():
            return None
        try:
            df = pd.read_csv(candidate)
            if "feature" not in df.columns or "importance" not in df.columns:
                logger.warning("[VizAgent] feature_importance.csv missing required columns")
                return None
            return {row["feature"]: row["importance"] for _, row in df.iterrows()}
        except Exception as e:
            logger.warning(f"[VizAgent] failed to read feature_importance.csv: {e}")
            return None

    def _chart_feature_importance(self, fi_data: Dict[str, float], manifest: ChartManifest) -> None:
        try:
            if not fi_data:
                return
            # Sort and take top 15
            sorted_items = sorted(fi_data.items(), key=lambda kv: kv[1], reverse=True)[:15]
            features, scores = zip(*sorted_items) if sorted_items else ([], [])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=scores, y=features, orientation='h'))
            fig.update_layout(template="plotly_dark", title="Feature Importance")
            path = self.working_dir / "feature_importance.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            top_feature, top_score = sorted_items[0] if sorted_items else ("", 0)
            narrative = f"Top feature: {top_feature} with importance {top_score:.3f}"
            narrative = self._get_narrative("feature_importance", narrative)
            manifest.add("feature_importance", "feature_importance.html", "Feature Importance", narrative, (len(fi_data),))
        except Exception as e:
            logger.warning(f"[VizAgent] feature_importance failed: {e}")

    def _chart_roc_curve(
        self,
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: List[str],
        manifest: ChartManifest,
    ) -> None:
        try:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_curve, auc
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder
            except ImportError:
                logger.warning("[VizAgent] sklearn not installed — skipping ROC curve")
                return

            # ── Guard: need numeric features and a binary target ──────────
            if not numeric_cols:
                logger.debug("[VizAgent] No numeric columns — skipping ROC curve")
                return

            y_raw = df[target_col].dropna()
            if y_raw.nunique() != 2:
                logger.debug(f"[VizAgent] Target '{target_col}' is not binary ({y_raw.nunique()} classes) — skipping ROC")
                return

            # ── Align X and y on the same non-null index ──────────────────
            valid_idx = df[target_col].notna()
            X = df.loc[valid_idx, numeric_cols].fillna(0)
            y_raw = df.loc[valid_idx, target_col]

            # ── Encode string targets (e.g. 'Yes'/'No', 'True'/'False') ───
            # Original code passed string y directly to roc_curve which
            # crashes with "could not convert string to float".
            if y_raw.dtype == object or hasattr(y_raw, "cat"):
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values

            # ── Require minimum sample size for a meaningful split ─────────
            if len(X) < 20:
                logger.debug(f"[VizAgent] Too few samples ({len(X)}) for ROC train/test split — skipping")
                return

            # ── Train/test split — original code trained and evaluated on  ─
            # the full dataset (data leakage). AUC on training data is      ─
            # meaningless and misleading in a report.                        ─
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=42,
                stratify=y,   # preserve class balance in both splits
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)   # transform only — no fit on test

            model = LogisticRegression(max_iter=500, random_state=42)
            model.fit(X_train_sc, y_train)

            # Evaluate on held-out test set only
            probs = model.predict_proba(X_test_sc)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"Baseline LR (AUC={roc_auc:.3f})",
                line=dict(width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random classifier",
                showlegend=True,
            ))
            fig.update_layout(
                template="plotly_dark",
                title=f"ROC Curve — {target_col} (test set, n={len(X_test)})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.6, y=0.1),
            )
            path = self.working_dir / "roc_curve.html"
            fig.write_html(str(path), include_plotlyjs="cdn")

            narrative = (
                f"Logistic Regression baseline AUC on held-out test set "
                f"({len(X_test)} samples): {roc_auc:.3f}. "
                f"{'Strong' if roc_auc > 0.8 else 'Moderate' if roc_auc > 0.65 else 'Weak'} "
                f"baseline signal detected."
            )
            narrative = self._get_narrative("roc_curve", narrative)
            manifest.add("roc_curve", "roc_curve.html", "ROC Curve", narrative, df.shape)

        except Exception as e:
            logger.warning(f"[VizAgent] roc_curve failed: {e}")
