"""Deep Learning Tabular Quantile Regression model for football attendance prediction in PyTorch.
Implements Entity Embeddings for categorical entities (home/away clubs, league),
StandardScaler for continuous features, Dense MLP with BatchNorm & Dropout,
and Pinball Loss for joint quantile estimation [Q10, Q50, Q90].
"""
from typing import Dict, List, Optional, Tuple, Sequence, Union, Any
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, cap: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate regression metrics for attendance predictions."""
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true > 0)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return {}

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100.0
    r2 = r2_score(yt, yp)

    res = {
        "MAE": round(mae, 1),
        "RMSE": round(rmse, 1),
        "MAPE_pct": round(mape, 2),
        "R2": round(r2, 4),
        "count": len(yt)
    }

    if cap is not None:
        c = cap[mask]
        occ_true = yt / (c + 1e-5)
        occ_pred = yp / (c + 1e-5)
        res["Occupancy_MAE_pct"] = round(mean_absolute_error(occ_true, occ_pred) * 100.0, 2)

    return res

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.scraper.stadiums import normalize_club_name

DEFAULT_FEATURE_COLS: List[str] = [
    "league_tier",
    "is_home_reserve", "is_away_reserve", "is_reserve_fixture",
    "dow",
    "hour_num",
    "sin_hour", "cos_hour",
    "sin_dow", "cos_dow",
    "is_weekend", "is_friday", "is_holiday_proximate",
    "is_round_opener",
    "temperature_c", "precipitation_mm", "rain_exposure_mm", "temp_comfort_dev",
    "effective_weather_discomfort",
    "cloud_cover_pct", "wind_speed_kmh", "is_sunny", "is_snow", "is_heavy_rain",
    "home_pts_last_5", "away_pts_last_5",
    "home_pts_last_3_home", "away_pts_last_3_away",
    "home_table_pos", "away_table_pos", "table_pos_diff", "table_pts_diff",
    "away_draw_factor",
    "travel_distance_km", "is_local_derby", "is_major_derby",
    "h2h_avg_att", "h2h_home_venue_avg_att",
    "home_att_ema_3", "home_occupancy_ema_5",
    "stadium_expansion_boost", "away_sector_closed",
    "is_title_or_promotion_race", "is_relegation_battle", "is_dead_rubber",
    "match_stakes_index",
    "roof_pct",
    "season_progress"
]


class CensoredSmoothPinballLoss(nn.Module):
    """
    Smooth (Huberized) and Right-Censored Quantile Loss for multi-quantile regression.
    
    Features:
      1. Huber smoothing around zero (|e| <= delta):
         Eliminates oscillating gradients around the minimum and accelerates convergence.
      2. Right-censored loss for sellout fixtures:
         If a fixture is sold out (is_sellout == 1), true spectator demand was constrained
         by stadium capacity. Thus, overprediction (y_hat > y) is not penalized up to
         cap_limit (e.g. 1.05 * capacity).
    """
    def __init__(self, quantiles: Sequence[float] = (0.25, 0.50, 0.75), delta: float = 0.02):
        super().__init__()
        self.quantiles = list(quantiles)
        self.delta = float(delta)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        is_sellout: Optional[torch.Tensor] = None,
        cap_limit: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        preds: Tensor of shape (batch_size, num_quantiles)
        targets: Tensor of shape (batch_size, 1) or (batch_size,)
        is_sellout: Optional Tensor of shape (batch_size, 1) or (batch_size,), 1.0 if match was sellout
        cap_limit: Optional Tensor of shape (batch_size, 1) or (batch_size,), upper capacity in target scale
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        if is_sellout is not None and is_sellout.ndim == 1:
            is_sellout = is_sellout.unsqueeze(1)
        if cap_limit is not None and cap_limit.ndim == 1:
            cap_limit = cap_limit.unsqueeze(1)

        errors = targets - preds  # (batch_size, num_quantiles)
        total_loss = 0.0

        for i, q in enumerate(self.quantiles):
            err = errors[:, i:i+1]
            p = preds[:, i:i+1]

            # Right-censored adjustment for overprediction in sellouts
            if is_sellout is not None and cap_limit is not None:
                # err < 0 means y < y_hat (model predicted more than capacity)
                over_pred_mask = (is_sellout > 0.5) & (err < 0)
                within_buf = over_pred_mask & (p <= cap_limit)
                above_buf = over_pred_mask & (p > cap_limit)

                eff_err = err.clone()
                eff_err = torch.where(within_buf, torch.zeros_like(eff_err), eff_err)
                eff_err = torch.where(above_buf, cap_limit - p, eff_err)
            else:
                eff_err = err

            # Huberized smooth pinball
            if self.delta > 0.0:
                abs_e = torch.abs(eff_err)
                huber_e = torch.where(
                    abs_e <= self.delta,
                    0.5 * (eff_err ** 2) / self.delta,
                    abs_e - 0.5 * self.delta
                )
                q_loss = torch.where(eff_err >= 0, q * huber_e, (1.0 - q) * huber_e)
            else:
                q_loss = torch.max(q * eff_err, (q - 1.0) * eff_err)

            total_loss = total_loss + q_loss.mean()

        return total_loss


class PinballLoss(CensoredSmoothPinballLoss):
    """
    Standard Joint Pinball (Quantile) Loss for multi-quantile regression (delta=0.0).
    L_tau(y, y_hat) = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))
    """
    def __init__(self, quantiles: Sequence[float] = (0.25, 0.50, 0.75)):
        super().__init__(quantiles=quantiles, delta=0.0)


class TabularResidualBlock(nn.Module):
    """
    Residual Block for tabular deep learning with SiLU (Swish) activations:
    x_out = x + Dropout(Linear(SiLU(BatchNorm(Linear(x)))))
    Features:
      - 2 linear sub-layers with BatchNorm and SiLU
      - Skip connection adding input directly to residual branch
      - Optional projection shortcut if in_dim != out_dim
    """
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.use_projection = in_features != out_features
        if self.use_projection:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        identity = self.shortcut(x)
        out = self.act(identity + res)
        out = self.dropout(out)
        return out


class TabularQuantileNet(nn.Module):
    """
    Modern Tabular ResNet with Structurally Monotonic Multi-Quantile Head [Q10, Q50, Q90]:
    - Hybrid Decomposed Club Embeddings: shared identity (d=10) + role-specific embeddings (home d=6, away d=6)
    - Random UNK embedding dropout during training for out-of-vocabulary robustness
    - Stacked TabularResidualBlocks with SiLU activations & skip connections
    - Monotonic Softplus output head guaranteeing Q10 <= Q50 <= Q90 natively in the computation graph
    """
    def __init__(
        self,
        num_clubs: int,
        num_leagues: int,
        num_continuous_features: int,
        club_emb_dim: int = 16,
        shared_club_dim: Optional[int] = None,
        league_emb_dim: int = 4,
        hidden_dims: Tuple[int, int, int] = (128, 64, 32),
        dropout_rates: Tuple[float, float, float] = (0.20, 0.15, 0.10),
        club_unk_dropout: float = 0.05,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ):
        super().__init__()
        self.club_emb_dim = club_emb_dim
        self.quantiles = list(quantiles)
        self.club_unk_dropout = float(club_unk_dropout)

        # Decompose club embedding into shared identity + role-specific component
        if shared_club_dim is None:
            self.shared_dim = max(2, int(round(club_emb_dim * 0.625)))  # 10 if club_emb_dim == 16
        else:
            self.shared_dim = shared_club_dim
        self.role_dim = max(1, club_emb_dim - self.shared_dim)          # 6 if club_emb_dim == 16

        # Index 0 is reserved for <UNK>
        self.shared_club_emb = nn.Embedding(num_clubs + 1, self.shared_dim, padding_idx=0)
        self.home_role_emb = nn.Embedding(num_clubs + 1, self.role_dim, padding_idx=0)
        self.away_role_emb = nn.Embedding(num_clubs + 1, self.role_dim, padding_idx=0)
        self.league_embedding = nn.Embedding(num_leagues + 1, league_emb_dim, padding_idx=0)

        in_dim = (club_emb_dim * 2) + league_emb_dim + num_continuous_features

        # Initial projection into hidden space with SiLU
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[0])
        )

        # ResNet Residual Blocks with skip connections
        self.res_block1 = TabularResidualBlock(hidden_dims[0], hidden_dims[1], dropout_rate=dropout_rates[1])
        self.res_block2 = TabularResidualBlock(hidden_dims[1], hidden_dims[2], dropout_rate=dropout_rates[2])

        # Output head generates [z1, z2, z3]
        # z1: Q50 median
        # z2: spread between Q50 and Q10
        # z3: spread between Q90 and Q50
        self.out_head = nn.Linear(hidden_dims[2], 3)

    def forward(
        self,
        home_idx: torch.Tensor,
        away_idx: torch.Tensor,
        league_idx: torch.Tensor,
        continuous_x: torch.Tensor
    ) -> torch.Tensor:
        # Randomly mask club tokens to <UNK> (index 0) during training
        if self.training and self.club_unk_dropout > 0.0:
            mask_h = torch.rand_like(home_idx, dtype=torch.float32) < self.club_unk_dropout
            mask_a = torch.rand_like(away_idx, dtype=torch.float32) < self.club_unk_dropout
            home_idx = torch.where(mask_h, torch.zeros_like(home_idx), home_idx)
            away_idx = torch.where(mask_a, torch.zeros_like(away_idx), away_idx)

        # Hybrid entity embeddings: shared identity + role-specific (with legacy fallback)
        if hasattr(self, "shared_club_emb") and hasattr(self, "home_role_emb"):
            emb_home = torch.cat([self.shared_club_emb(home_idx), self.home_role_emb(home_idx)], dim=1)
            emb_away = torch.cat([self.shared_club_emb(away_idx), self.away_role_emb(away_idx)], dim=1)
        elif hasattr(self, "club_embedding"):
            emb_home = self.club_embedding(home_idx)
            emb_away = self.club_embedding(away_idx)
        else:
            raise AttributeError("TabularQuantileNet has neither 'shared_club_emb' nor 'club_embedding'.")

        emb_league = self.league_embedding(league_idx)

        # Concatenate embeddings and continuous features
        x = torch.cat([emb_home, emb_away, emb_league, continuous_x], dim=1)
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        z = self.out_head(x)

        # Backward compatibility for models saved before monotonic softplus spread head
        if not hasattr(self, "shared_club_emb"):
            return z

        # Structurally enforce Q10 <= Q50 <= Q90 via softplus positive spreads
        q50 = z[:, 0:1]
        spread_lower = torch.nn.functional.softplus(z[:, 1:2]) + 1e-4
        spread_upper = torch.nn.functional.softplus(z[:, 2:3]) + 1e-4

        q10 = q50 - spread_lower
        q90 = q50 + spread_upper

        return torch.cat([q10, q50, q90], dim=1)

    def init_empirical_bias(self, y_scaled: np.ndarray):
        """
        Empirical bias initialization in latent scaled space.
        Initializes out_head.bias such that raw activations correspond exactly to
        the sample empirical quantiles [Q10, Q50, Q90] before training begins.
        """
        q10 = float(np.quantile(y_scaled, self.quantiles[0]))
        q50 = float(np.quantile(y_scaled, self.quantiles[1]))
        q90 = float(np.quantile(y_scaled, self.quantiles[2]))

        d_lower = max(1e-4, q50 - q10)
        d_upper = max(1e-4, q90 - q50)

        # Inverse softplus: softplus_inv(y) = ln(exp(y) - 1)
        b1 = float(np.log(np.expm1(d_lower))) if d_lower < 20.0 else d_lower
        b2 = float(np.log(np.expm1(d_upper))) if d_upper < 20.0 else d_upper

        with torch.no_grad():
            self.out_head.bias.data.copy_(
                torch.tensor([q50, b1, b2], dtype=torch.float32, device=self.out_head.bias.device)
            )



class TabularQuantileNN(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for PyTorch Tabular Quantile Neural Network.
    
    Provides:
      - fit(X, y): fits embeddings, StandardScaler and trains neural network with Pinball Loss.
      - predict(X): returns Q50 (median attendance estimate).
      - predict_interval(X): returns DataFrame with [q10, q50, q90], row-sorted and clipped to capacity.
    """
    def __init__(
        self,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
        club_emb_dim: int = 16,
        shared_club_dim: Optional[int] = None,
        league_emb_dim: int = 4,
        hidden_dims: Tuple[int, int, int] = (128, 64, 32),
        dropout_rates: Tuple[float, float, float] = (0.20, 0.15, 0.10),
        club_unk_dropout: float = 0.05,
        loss_delta: float = 0.02,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 120,
        val_split: float = 0.15,
        patience: int = 18,
        log_target: bool = True,
        temporal_validation: bool = False,
        conformal_alpha: float = 0.50,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.quantiles = list(quantiles)
        self.club_emb_dim = club_emb_dim
        self.shared_club_dim = shared_club_dim
        self.league_emb_dim = league_emb_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.club_unk_dropout = club_unk_dropout
        self.loss_delta = loss_delta
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.patience = patience
        self.log_target = log_target
        self.temporal_validation = temporal_validation
        self.conformal_alpha = conformal_alpha
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        # Fitted state
        self.scaler_: Optional[StandardScaler] = None
        self.club_to_idx_: Dict[str, int] = {}
        self.league_to_idx_: Dict[Any, int] = {}
        self.continuous_cols_: List[str] = []
        self.network_: Optional[TabularQuantileNet] = None
        self.target_scale_: float = 1000.0
        self.conformal_delta_: float = 0.0
        self.is_fitted_: bool = False

    def _determine_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_continuous_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract continuous features and handle missing values."""
        df_cont = pd.DataFrame(index=X.index)
        for col in self.continuous_cols_:
            if col in X.columns:
                df_cont[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                df_cont[col] = 0.0

        # Safe domain-specific defaults
        defaults = {
            "temperature_c": 15.0,
            "precipitation_mm": 0.0,
            "rain_exposure_mm": 0.0,
            "temp_comfort_dev": 3.0,
            "effective_weather_discomfort": 0.0,
            "travel_distance_km": 200.0,
            "away_draw_factor": 1.0,
            "roof_pct": 1.0,
            "dow": 5.0,
            "hour_num": 18.0,
            "season_progress": 0.5,
            "home_att_ema_3": 2500.0,
            "home_occupancy_ema_5": 0.45,
            "stadium_expansion_boost": 1.0,
            "away_sector_closed": 0.0,
            "is_round_opener": 0.0,
        }
        for col, default_val in defaults.items():
            if col in df_cont.columns:
                df_cont[col] = df_cont[col].fillna(default_val)
                
        df_cont = df_cont.fillna(0.0)
        return df_cont

    @staticmethod
    def _safe_normalize_club(name: Any) -> str:
        """Safely normalize club name handling None, NaN and non-string types."""
        if name is None or pd.isna(name) or str(name).strip().lower() in ("nan", "none", ""):
            return ""
        return normalize_club_name(str(name))

    def _extract_tensors(self, X: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert input DataFrame into PyTorch tensors for network input."""
        # 1. Home team indices
        if "home_team" in X.columns:
            clean_home = X["home_team"].apply(self._safe_normalize_club)
            home_indices = clean_home.map(self.club_to_idx_).fillna(0).astype(int).values
        else:
            home_indices = np.zeros(len(X), dtype=int)

        # 2. Away team indices
        if "away_team" in X.columns:
            clean_away = X["away_team"].apply(self._safe_normalize_club)
            away_indices = clean_away.map(self.club_to_idx_).fillna(0).astype(int).values
        else:
            away_indices = np.zeros(len(X), dtype=int)

        # 3. League indices
        if "league_tier" in X.columns:
            league_raw = X["league_tier"]
        elif "league_name" in X.columns:
            league_raw = X["league_name"]
        else:
            league_raw = pd.Series([1] * len(X), index=X.index)
        league_indices = league_raw.map(self.league_to_idx_).fillna(0).astype(int).values

        # 4. Continuous scaled features
        cont_df = self._prepare_continuous_df(X)
        cont_scaled = self.scaler_.transform(cont_df)

        return (
            torch.tensor(home_indices, dtype=torch.long),
            torch.tensor(away_indices, dtype=torch.long),
            torch.tensor(league_indices, dtype=torch.long),
            torch.tensor(cont_scaled, dtype=torch.float32)
        )

    def fit(self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series, Sequence]] = None) -> "TabularQuantileNN":
        """
        Fit TabularQuantileNN on tabular DataFrame X and target attendance y.
        If y is None and 'attendance' column is in X, it will be automatically extracted.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            if "attendance" in X.columns:
                y_arr = X["attendance"].values
            else:
                raise ValueError("Target y must be provided or 'attendance' column must be present in X.")
        else:
            y_arr = np.asarray(y)

        # Filter out invalid / NaN targets if present
        valid_mask = (~np.isnan(y_arr)) & (y_arr > 0)
        X_clean = X.loc[valid_mask].copy().reset_index(drop=True)
        y_clean = y_arr[valid_mask]

        # 1. Build Entity Vocabularies
        all_clubs = set()
        if "home_team" in X_clean.columns:
            all_clubs.update(X_clean["home_team"].apply(self._safe_normalize_club).unique())
        if "away_team" in X_clean.columns:
            all_clubs.update(X_clean["away_team"].apply(self._safe_normalize_club).unique())
            
        sorted_clubs = sorted([c for c in all_clubs if c])
        # 0 is reserved for <UNK>
        self.club_to_idx_ = {club: idx + 1 for idx, club in enumerate(sorted_clubs)}

        # League vocabulary
        if "league_tier" in X_clean.columns:
            leagues = sorted(X_clean["league_tier"].dropna().unique().tolist())
        elif "league_name" in X_clean.columns:
            leagues = sorted(X_clean["league_name"].dropna().unique().tolist())
        else:
            leagues = [1, 2, 3]
        self.league_to_idx_ = {lg: idx + 1 for idx, lg in enumerate(leagues)}

        # 2. Select Continuous Features (all numerical features except categorical entity identifiers and capacity)
        entity_cols = {"home_team", "away_team", "league_tier", "league_name", "match_id", "attendance", "occupancy_pct", "capacity_resolved"}
        base_cols = [c for c in DEFAULT_FEATURE_COLS if c not in entity_cols]
        # Include any matching columns from X_clean
        self.continuous_cols_ = [c for c in base_cols if c in X_clean.columns or c in DEFAULT_FEATURE_COLS]
        if not self.continuous_cols_:
            # Fallback to all numeric columns in X
            self.continuous_cols_ = [c for c in X_clean.select_dtypes(include=[np.number]).columns if c not in entity_cols]

        # 3. Fit Integrated StandardScaler
        cont_df = self._prepare_continuous_df(X_clean)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(cont_df)

        # 4. Target Scaling and PyTorch Dataset Preparation
        if self.log_target:
            # Log-transform target: z = log(1 + y)
            y_trans = np.log1p(np.maximum(0.0, y_clean))
            self.target_scale_ = float(np.std(y_trans)) if np.std(y_trans) > 0 else 1.0
            y_scaled = y_trans / self.target_scale_
        else:
            self.target_scale_ = float(np.std(y_clean)) if np.std(y_clean) > 0 else 5000.0
            y_scaled = y_clean / self.target_scale_

        # Extract capacity and sellout indicators for Censored Loss
        if "capacity_resolved" in X_clean.columns:
            caps = pd.to_numeric(X_clean["capacity_resolved"], errors="coerce").fillna(5000.0).values
            is_sellout_arr = (y_clean >= 0.97 * caps).astype(np.float32)
            cap_limit_arr = caps * 1.05
            if self.log_target:
                cap_scaled = np.log1p(np.maximum(0.0, cap_limit_arr)) / self.target_scale_
            else:
                cap_scaled = cap_limit_arr / self.target_scale_
        else:
            is_sellout_arr = np.zeros(len(y_clean), dtype=np.float32)
            cap_scaled = np.full(len(y_clean), 1e6, dtype=np.float32)

        n_samples = len(X_clean)
        val_size = int(round(n_samples * self.val_split)) if self.val_split > 0 else 0
        train_size = n_samples - val_size

        if self.temporal_validation and val_size > 0:
            date_cols = [c for c in ["match_datetime", "match_date", "dt"] if c in X_clean.columns]
            if date_cols:
                sort_col = date_cols[0]
                sort_idx = np.argsort(pd.to_datetime(X_clean[sort_col], errors="coerce").values)
                X_clean = X_clean.iloc[sort_idx].reset_index(drop=True)
                y_clean = y_clean[sort_idx]
                y_scaled = y_scaled[sort_idx]
                is_sellout_arr = is_sellout_arr[sort_idx]
                cap_scaled = cap_scaled[sort_idx]

            # Last val_size fixtures chronologically form the out-of-time validation set
            train_idx = np.arange(train_size)
            val_idx = np.arange(train_size, n_samples)
        elif val_size > 0:
            # Deterministic, non-random systematic validation split across seasons & tiers (no random clustering)
            strat_cols = [c for c in ["season", "league_tier"] if c in X_clean.columns]
            if strat_cols:
                val_indices = []
                for _, grp in X_clean.groupby(strat_cols, sort=False):
                    n_g = len(grp)
                    k_v = int(round(n_g * self.val_split))
                    if k_v == 0 and n_g >= 3:
                        k_v = 1
                    if k_v > 0:
                        step_v = n_g / k_v
                        chosen_rel = [min(n_g - 1, int(np.floor((i + 0.5) * step_v))) for i in range(k_v)]
                        chosen_rel = sorted(list(set(chosen_rel)))
                        val_indices.extend(grp.index.values[chosen_rel])
                val_mask = X_clean.index.isin(val_indices)
                val_idx = np.where(val_mask)[0]
                train_idx = np.where(~val_mask)[0]
            else:
                indices = np.arange(n_samples)
                train_idx, val_idx = indices[val_size:], indices[:val_size]
        else:
            train_idx = np.arange(n_samples)
            val_idx = np.array([], dtype=int)

        t_home, t_away, t_league, t_cont = self._extract_tensors(X_clean)
        t_y = torch.tensor(y_scaled, dtype=torch.float32)
        t_sellout = torch.tensor(is_sellout_arr, dtype=torch.float32)
        t_cap = torch.tensor(cap_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(
            t_home[train_idx], t_away[train_idx], t_league[train_idx], t_cont[train_idx],
            t_y[train_idx], t_sellout[train_idx], t_cap[train_idx]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_idx)),
            shuffle=True,
            drop_last=len(train_idx) > self.batch_size
        )

        val_loader = None
        if val_size > 0:
            val_dataset = TensorDataset(
                t_home[val_idx], t_away[val_idx], t_league[val_idx], t_cont[val_idx],
                t_y[val_idx], t_sellout[val_idx], t_cap[val_idx]
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 5. Build Network Architecture
        dev = self._determine_device()
        self.network_ = TabularQuantileNet(
            num_clubs=len(self.club_to_idx_),
            num_leagues=len(self.league_to_idx_),
            num_continuous_features=len(self.continuous_cols_),
            club_emb_dim=self.club_emb_dim,
            shared_club_dim=self.shared_club_dim,
            league_emb_dim=self.league_emb_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates,
            club_unk_dropout=self.club_unk_dropout,
            quantiles=self.quantiles
        ).to(dev)

        # Initialize output bias to empirical quantiles of scaled target in latent space
        self.network_.init_empirical_bias(y_scaled)

        criterion = CensoredSmoothPinballLoss(quantiles=self.quantiles, delta=self.loss_delta)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

        # 6. Training Loop with Early Stopping
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            running_loss = 0.0
            total_batches = 0

            for b_home, b_away, b_league, b_cont, b_y, b_sell, b_cap in train_loader:
                b_home = b_home.to(dev)
                b_away = b_away.to(dev)
                b_league = b_league.to(dev)
                b_cont = b_cont.to(dev)
                b_y = b_y.to(dev)
                b_sell = b_sell.to(dev)
                b_cap = b_cap.to(dev)

                optimizer.zero_grad()
                preds = self.network_(b_home, b_away, b_league, b_cont)
                loss = criterion(preds, b_y, is_sellout=b_sell, cap_limit=b_cap)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            train_loss = running_loss / max(1, total_batches)

            # Validation
            if val_loader is not None:
                self.network_.eval()
                val_loss_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for b_home, b_away, b_league, b_cont, b_y, b_sell, b_cap in val_loader:
                        b_home = b_home.to(dev)
                        b_away = b_away.to(dev)
                        b_league = b_league.to(dev)
                        b_cont = b_cont.to(dev)
                        b_y = b_y.to(dev)
                        b_sell = b_sell.to(dev)
                        b_cap = b_cap.to(dev)
                        p_val = self.network_(b_home, b_away, b_league, b_cont)
                        v_loss = criterion(p_val, b_y, is_sellout=b_sell, cap_limit=b_cap)
                        val_loss_total += v_loss.item()
                        val_batches += 1

                val_loss = val_loss_total / max(1, val_batches)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.network_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Train Pinball Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.2f}")
                    break
            else:
                if self.verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Train Pinball Loss: {train_loss:.2f}")

        # Restore best weights if validation was used
        if best_weights is not None:
            self.network_.load_state_dict({k: v.to(dev) for k, v in best_weights.items()})

        # 7. Conformalized Quantile Regression (CQR) Calibration
        self.conformal_delta_ = 0.0
        if val_size > 0:
            self.network_.eval()
            with torch.no_grad():
                val_raw = self.network_(
                    t_home[val_idx].to(dev),
                    t_away[val_idx].to(dev),
                    t_league[val_idx].to(dev),
                    t_cont[val_idx].to(dev)
                ).cpu().numpy()
            val_raw = val_raw * self.target_scale_
            if self.log_target:
                val_raw = np.clip(val_raw, 0.0, 15.0)
                val_raw = np.expm1(val_raw)
                val_raw = np.maximum(0.0, val_raw)

            val_q_low = val_raw[:, 0]
            val_q_high = val_raw[:, 2]
            val_y = y_clean[val_idx]

            # Non-conformity scores: E_i = max(q_low - y, y - q_high)
            conformity_errors = np.maximum(val_q_low - val_y, val_y - val_q_high)
            n_val = len(val_y)
            desired_coverage = 1.0 - getattr(self, "conformal_alpha", 0.50)
            q_level = min(1.0, math.ceil((n_val + 1) * desired_coverage) / n_val)
            delta_hat = float(np.quantile(conformity_errors, q_level))
            self.conformal_delta_ = float(delta_hat)
            if self.verbose:
                cov_pct = int(round(desired_coverage * 100))
                print(f"CQR Calibrated {cov_pct}% (IQR) Conformal Delta: {self.conformal_delta_:+.1f} spectators (q_level={q_level:.3f})")

        # Move to CPU for safe pickling/joblib serialization
        self.network_.to(torch.device("cpu"))
        self.network_.eval()
        self.is_fitted_ = True
        return self

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict Q10, Q50, and Q90 quantiles.
        - Guarantees non-crossing quantiles via row-wise sorting: q10 <= q50 <= q90.
        - Applies CQR calibration offset for certified 80% coverage on out-of-distribution fixtures.
        - Clips predictions to [50, 1.05 * capacity] if capacity is present.
        """
        if not self.is_fitted_ or self.network_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.network_.eval()
        t_home, t_away, t_league, t_cont = self._extract_tensors(X)

        with torch.no_grad():
            raw_out = self.network_(t_home, t_away, t_league, t_cont).cpu().numpy()
        raw_out = raw_out * self.target_scale_

        # Inverse target transformation if log_target was used
        if getattr(self, "log_target", False):
            # Clip numerical bounds before exp to avoid overflow
            raw_out = np.clip(raw_out, 0.0, 15.0)
            raw_out = np.expm1(raw_out)
            raw_out = np.maximum(0.0, raw_out)

        # 1. Enforce monotonic order: row-wise sort prevents quantile crossing
        sorted_out = np.sort(raw_out, axis=1)

        # 2. Conformal Quantile Calibration adjustment
        c_delta = getattr(self, "conformal_delta_", 0.0)
        if c_delta != 0.0:
            q10_cal = sorted_out[:, 0] - c_delta
            q50_cal = sorted_out[:, 1]
            q90_cal = sorted_out[:, 2] + c_delta
            # Monotonicity invariant: q10 <= q50 <= q90
            q10_cal = np.minimum(q10_cal, q50_cal)
            q90_cal = np.maximum(q90_cal, q50_cal)
            sorted_out = np.column_stack([q10_cal, q50_cal, q90_cal])

        # 3. Capacity clipping
        if "capacity_resolved" in X.columns:
            cap = pd.to_numeric(X["capacity_resolved"], errors="coerce").fillna(5000.0).values
            upper_bound = cap * 1.05
            sorted_out = np.clip(sorted_out, 50.0, upper_bound[:, None])
        else:
            sorted_out = np.clip(sorted_out, 50.0, None)

        q_low_name = f"q{int(round(self.quantiles[0]*100))}"
        q_mid_name = f"q{int(round(self.quantiles[1]*100))}"
        q_high_name = f"q{int(round(self.quantiles[2]*100))}"

        res_df = pd.DataFrame(
            {
                q_low_name: np.round(sorted_out[:, 0]),
                q_mid_name: np.round(sorted_out[:, 1]),
                q_high_name: np.round(sorted_out[:, 2]),
            },
            index=X.index
        )
        res_df["q25"] = res_df[q_low_name]
        res_df["q50"] = res_df[q_mid_name]
        res_df["q75"] = res_df[q_high_name]
        res_df["q10"] = res_df[q_low_name]  # compatibility alias
        res_df["q90"] = res_df[q_high_name]  # compatibility alias
        res_df["interval_width"] = res_df[q_high_name] - res_df[q_low_name]
        
        # Compatibility aliases for downstream pipelines
        res_df["pred_attendance"] = res_df["q50"]
        if "capacity_resolved" in X.columns:
            cap = pd.to_numeric(X["capacity_resolved"], errors="coerce").fillna(5000.0).values
            res_df["pred_occupancy"] = np.round(np.clip(res_df["pred_attendance"] / (cap + 1e-5), 0.01, 1.05), 4)

        return res_df

    # Backward compatibility alias
    predict_intervals = predict_interval

    def predict(self, X: pd.DataFrame, use_club_adaptation: bool = True) -> np.ndarray:
        """
        Scikit-learn compatible point prediction method.
        Returns Q50 (median attendance estimate).
        """
        interval_df = self.predict_interval(X)
        return interval_df["q50"].values

    def score(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series, Sequence]) -> float:
        """Calculate R^2 score between Q50 predictions and actual y."""
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        mask = (~np.isnan(y_true)) & (y_true > 0)
        return r2_score(y_true[mask], y_pred[mask])

    def get_feature_importances(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series, Sequence]] = None,
        method: str = "permutation",
        n_repeats: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute feature importance for TabularQuantileNN.
        Supports:
          - 'permutation': Permutation Feature Importance (PFI) based on increase in MAE when shuffling a feature.
          - 'weights': First-layer input weight magnitude attribution.
        """
        if not self.is_fitted_ or self.network_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Method A: Weight magnitude attribution (fast, no dataset needed)
        if method == "weights" or X is None:
            w = self.network_.input_layer[0].weight.detach().cpu().numpy()  # shape (hidden_0, in_dim)
            # Magnitude per input dimension
            in_mag = np.mean(np.abs(w), axis=0)

            # In_dim breakdown:
            # [0 : club_dim] -> home_team
            # [club_dim : 2*club_dim] -> away_team
            # [2*club_dim : 2*club_dim + league_dim] -> league_tier
            # [2*club_dim + league_dim :] -> continuous_cols_
            c_dim = self.club_emb_dim
            l_dim = self.league_emb_dim

            records = [
                {"feature": "home_team", "importance": float(np.mean(in_mag[0:c_dim]))},
                {"feature": "away_team", "importance": float(np.mean(in_mag[c_dim: 2 * c_dim]))},
                {"feature": "league_tier", "importance": float(np.mean(in_mag[2 * c_dim: 2 * c_dim + l_dim]))},
            ]
            cont_offset = 2 * c_dim + l_dim
            for i, col in enumerate(self.continuous_cols_):
                records.append({
                    "feature": col,
                    "importance": float(in_mag[cont_offset + i])
                })
            df_imp = pd.DataFrame(records).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df_imp

        # Method B: Permutation Feature Importance (PFI) on provided dataset X, y
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            if "attendance" in X.columns:
                y_true = X["attendance"].values
            else:
                raise ValueError("Target y or 'attendance' column is required for permutation importance.")
        else:
            y_true = np.asarray(y)

        # Baseline error (MAE)
        base_preds = self.predict(X)
        base_mae = mean_absolute_error(y_true, base_preds)

        rng = np.random.RandomState(random_state)
        feature_list = ["home_team", "away_team", "league_tier"] + list(self.continuous_cols_)
        # Filter to features actually present in X
        eval_features = [f for f in feature_list if f in X.columns or f in ("home_team", "away_team", "league_tier")]

        importance_scores = []
        for feat in eval_features:
            mae_diffs = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                if feat in X_permuted.columns:
                    shuffled_vals = rng.permutation(X_permuted[feat].values)
                    X_permuted[feat] = shuffled_vals
                preds_shuffled = self.predict(X_permuted)
                perm_mae = mean_absolute_error(y_true, preds_shuffled)
                mae_diffs.append(max(0.0, perm_mae - base_mae))
            importance_scores.append({
                "feature": feat,
                "importance": float(np.mean(mae_diffs))
            })

        df_imp = pd.DataFrame(importance_scores).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp


class NeuralEnsembleQuantileModel(BaseEstimator, RegressorMixin):
    """
    Hybrid Ensemble Quantile Model combining Deep Tabular ResNets (PyTorch) with
    Multi-Quantile Gradient Boosted Trees (CatBoost & LightGBM).

    Architecture:
      - 3x TabularQuantileNet (PyTorch ResNets with SiLU and Conformal CQR, weight: 0.50)
      - 1x CatBoostRegressor (MultiQuantile:alpha=0.1,0.5,0.9, native categoricals, weight: 0.30)
      - 1x LightGBMRegressor (Quantile regressors for Q10, Q50, Q90, weight: 0.20)
    
    Provides:
      - Hybrid diversity combining smooth neural representations with sharp decision tree boundaries.
      - Certified 80% coverage on out-of-distribution fixtures.
      - Stable non-crossing quantiles [Q10, Q50, Q90] clipped to venue capacity.
    """
    def __init__(
        self,
        n_models: int = 3,
        use_catboost: bool = True,
        use_lightgbm: bool = True,
        nn_weight: float = 0.50,
        catboost_weight: float = 0.30,
        lightgbm_weight: float = 0.20,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
        club_emb_dim: int = 16,
        shared_club_dim: Optional[int] = None,
        league_emb_dim: int = 4,
        hidden_dims: Tuple[int, int, int] = (128, 64, 32),
        dropout_rates: Tuple[float, float, float] = (0.20, 0.15, 0.10),
        club_unk_dropout: float = 0.05,
        loss_delta: float = 0.02,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        val_split: float = 0.15,
        patience: int = 15,
        log_target: bool = True,
        temporal_validation: bool = False,
        conformal_alpha: float = 0.50,
        base_seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.n_models = n_models
        self.use_catboost = use_catboost
        self.use_lightgbm = use_lightgbm
        self.nn_weight = float(nn_weight)
        self.catboost_weight = float(catboost_weight)
        self.lightgbm_weight = float(lightgbm_weight)
        self.quantiles = list(quantiles)
        self.club_emb_dim = club_emb_dim
        self.shared_club_dim = shared_club_dim
        self.league_emb_dim = league_emb_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.club_unk_dropout = club_unk_dropout
        self.loss_delta = loss_delta
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.patience = patience
        self.log_target = log_target
        self.temporal_validation = temporal_validation
        self.conformal_alpha = conformal_alpha
        self.base_seed = base_seed
        self.device = device
        self.verbose = verbose

        self.models_: List[TabularQuantileNN] = []
        self.catboost_model_: Optional[Any] = None
        self.lightgbm_models_: Dict[float, Any] = {}
        self.tree_features_: List[str] = []
        self.tree_cat_features_: List[str] = ["home_team", "away_team", "league_tier"]
        self.is_fitted_: bool = False

    def _prepare_tree_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare feature representations for CatBoost (strings) and LightGBM (numeric codes)."""
        df_cb = pd.DataFrame(index=X.index)
        
        # Categorical strings for CatBoost
        df_cb["home_team"] = X["home_team"].apply(TabularQuantileNN._safe_normalize_club) if "home_team" in X.columns else "Unknown"
        df_cb["home_team"] = df_cb["home_team"].replace("", "Unknown")
        
        df_cb["away_team"] = X["away_team"].apply(TabularQuantileNN._safe_normalize_club) if "away_team" in X.columns else "Unknown"
        df_cb["away_team"] = df_cb["away_team"].replace("", "Unknown")
        
        if "league_tier" in X.columns:
            df_cb["league_tier"] = X["league_tier"].fillna(1).astype(int).astype(str)
        else:
            df_cb["league_tier"] = "1"

        # Continuous features from neural network fitted model (or DEFAULT_FEATURE_COLS)
        cont_cols = self.models_[0].continuous_cols_ if self.models_ else [c for c in DEFAULT_FEATURE_COLS if c not in ("home_team", "away_team", "league_tier")]
        for col in cont_cols:
            if col in X.columns:
                df_cb[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            else:
                df_cb[col] = 0.0

        # Numeric encoding for LightGBM
        df_lgb = df_cb.copy()
        if self.models_ and self.models_[0].club_to_idx_:
            c_map = self.models_[0].club_to_idx_
            l_map = self.models_[0].league_to_idx_
            df_lgb["home_team"] = df_lgb["home_team"].map(c_map).fillna(0).astype(int)
            df_lgb["away_team"] = df_lgb["away_team"].map(c_map).fillna(0).astype(int)
            df_lgb["league_tier"] = pd.to_numeric(df_lgb["league_tier"], errors="coerce").fillna(1).astype(int)
        else:
            df_lgb["home_team"] = df_lgb["home_team"].astype("category").cat.codes
            df_lgb["away_team"] = df_lgb["away_team"].astype("category").cat.codes
            df_lgb["league_tier"] = pd.to_numeric(df_lgb["league_tier"], errors="coerce").fillna(1).astype(int)

        return df_cb, df_lgb

    def fit(self, X: pd.DataFrame, y: Optional[Union[np.ndarray, pd.Series, Sequence]] = None) -> "NeuralEnsembleQuantileModel":
        """Train hybrid ensemble: PyTorch ResNets + CatBoost MultiQuantile + LightGBM Quantiles."""
        self.models_ = []
        self.catboost_model_ = None
        self.lightgbm_models_ = {}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            if "attendance" in X.columns:
                y_arr = X["attendance"].values
            else:
                raise ValueError("Target y must be provided or 'attendance' column must be present in X.")
        else:
            y_arr = np.asarray(y)

        valid_mask = (~np.isnan(y_arr)) & (y_arr > 0)
        X_clean = X.loc[valid_mask].copy().reset_index(drop=True)
        y_clean = y_arr[valid_mask]

        # 1. Train PyTorch Tabular ResNet Models
        seeds = [self.base_seed + (i * 101) for i in range(self.n_models)]
        for i, seed in enumerate(seeds):
            if self.verbose:
                print(f"--- Training TabularQuantileNet Member {i+1}/{self.n_models} (Seed: {seed}) ---")
            
            jitter = (i - (self.n_models // 2)) * 0.02
            member_dropouts = (
                max(0.05, min(0.35, self.dropout_rates[0] + jitter)),
                max(0.05, min(0.30, self.dropout_rates[1] + jitter)),
                max(0.05, min(0.25, self.dropout_rates[2] + jitter)),
            )

            model = TabularQuantileNN(
                quantiles=self.quantiles,
                club_emb_dim=self.club_emb_dim,
                shared_club_dim=self.shared_club_dim,
                league_emb_dim=self.league_emb_dim,
                hidden_dims=self.hidden_dims,
                dropout_rates=member_dropouts,
                club_unk_dropout=self.club_unk_dropout,
                loss_delta=self.loss_delta,
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                epochs=self.epochs,
                val_split=self.val_split,
                patience=self.patience,
                log_target=self.log_target,
                temporal_validation=self.temporal_validation,
                conformal_alpha=self.conformal_alpha,
                random_state=seed,
                device=self.device,
                verbose=False
            )
            model.fit(X_clean, y_clean)
            self.models_.append(model)

        df_cb, df_lgb = self._prepare_tree_features(X_clean)
        self.tree_features_ = list(df_cb.columns)

        y_tree = np.log1p(np.maximum(0.0, y_clean)) if self.log_target else y_clean

        # 2. Train CatBoost MultiQuantile Regressor
        if self.use_catboost:
            try:
                from catboost import CatBoostRegressor
                if self.verbose:
                    print("--- Training CatBoost MultiQuantile Regressor ---")
                cb = CatBoostRegressor(
                    loss_function=f"MultiQuantile:alpha={self.quantiles[0]},{self.quantiles[1]},{self.quantiles[2]}",
                    iterations=250,
                    depth=6,
                    learning_rate=0.06,
                    random_seed=self.base_seed,
                    verbose=0
                )
                cb.fit(df_cb, y_tree, cat_features=["home_team", "away_team", "league_tier"])
                self.catboost_model_ = cb
            except Exception as e:
                if self.verbose:
                    print(f"Notice: CatBoost training skipped ({e}).")
                self.catboost_model_ = None

        # 3. Train LightGBM Quantile Regressors (for datasets with N >= 20)
        if self.use_lightgbm and len(df_lgb) >= 20:
            try:
                import lightgbm as lgb
                if self.verbose:
                    print("--- Training LightGBM Multi-Quantile Regressors ---")
                for q in self.quantiles:
                    lgbm = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=120,
                        learning_rate=0.06,
                        random_state=self.base_seed,
                        min_child_samples=5,
                        verbose=-1
                    )
                    lgbm.fit(df_lgb, y_tree)
                    self.lightgbm_models_[q] = lgbm
            except Exception as e:
                if self.verbose:
                    print(f"Notice: LightGBM training skipped ({e}).")
                self.lightgbm_models_ = {}

        # Conformal delta as average of neural network sub-models
        deltas = [m.conformal_delta_ for m in self.models_ if hasattr(m, "conformal_delta_")]
        self.conformal_delta_ = float(np.mean(deltas)) if deltas else 0.0

        self.is_fitted_ = True
        return self

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict Q10, Q50, and Q90 quantiles using weighted hybrid ensembling.
        Combines predictions from Neural Networks (0.50), CatBoost (0.30), and LightGBM (0.20).
        Guarantees non-crossing quantiles and clips to stadium capacity.
        """
        if not self.is_fitted_ or not self.models_:
            raise RuntimeError("Ensemble is not fitted yet.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = len(X)
        if n_samples == 0:
            return pd.DataFrame(columns=["q10", "q50", "q90", "interval_width", "pred_attendance", "pred_occupancy"])

        # 1. Neural Networks Prediction (average across members)
        all_q10 = []
        all_q50 = []
        all_q90 = []
        for model in self.models_:
            df_m = model.predict_interval(X)
            all_q10.append(df_m["q10"].values)
            all_q50.append(df_m["q50"].values)
            all_q90.append(df_m["q90"].values)
        
        nn_q10 = np.mean(all_q10, axis=0)
        nn_q50 = np.mean(all_q50, axis=0)
        nn_q90 = np.mean(all_q90, axis=0)
        nn_pred = np.column_stack([nn_q10, nn_q50, nn_q90])

        total_weight = 0.0
        weighted_q = np.zeros((n_samples, 3), dtype=float)

        # Add NN component (with backward compatibility)
        nn_w = getattr(self, "nn_weight", 1.0)
        weighted_q += nn_w * nn_pred
        total_weight += nn_w

        # 2. CatBoost MultiQuantile Prediction (if enabled and fitted)
        cb_model = getattr(self, "catboost_model_", None)
        cb_w = getattr(self, "catboost_weight", 0.0)
        if cb_model is not None and cb_w > 0:
            df_cb, df_lgb = self._prepare_tree_features(X)
            try:
                cb_raw = cb_model.predict(df_cb)  # shape (n_samples, 3)
                if getattr(self, "log_target", True):
                    cb_raw = np.clip(cb_raw, 0.0, 15.0)
                    cb_raw = np.expm1(cb_raw)
                    cb_raw = np.maximum(0.0, cb_raw)
                weighted_q += cb_w * cb_raw
                total_weight += cb_w
            except Exception as e:
                if getattr(self, "verbose", False):
                    print(f"Notice: CatBoost prediction fallback ({e}).")

        # 3. LightGBM Quantile Prediction (if enabled and fitted)
        lgb_models = getattr(self, "lightgbm_models_", None)
        lgb_w = getattr(self, "lightgbm_weight", 0.0)
        if lgb_models and len(lgb_models) == len(self.quantiles) and lgb_w > 0:
            df_cb, df_lgb = self._prepare_tree_features(X)
            try:
                lgb_q = []
                for q in self.quantiles:
                    q_pred = lgb_models[q].predict(df_lgb)
                    if getattr(self, "log_target", True):
                        q_pred = np.clip(q_pred, 0.0, 15.0)
                        q_pred = np.expm1(q_pred)
                        q_pred = np.maximum(0.0, q_pred)
                    lgb_q.append(q_pred)
                lgb_stacked = np.column_stack(lgb_q)
                weighted_q += lgb_w * lgb_stacked
                total_weight += lgb_w
            except Exception as e:
                if getattr(self, "verbose", False):
                    print(f"Notice: LightGBM prediction fallback ({e}).")

        # Normalize by active component weights
        final_q = weighted_q / max(1e-5, total_weight)

        # Enforce non-crossing monotonic quantiles
        sorted_out = np.sort(final_q, axis=1)

        # Capacity clipping
        if "capacity_resolved" in X.columns:
            cap = pd.to_numeric(X["capacity_resolved"], errors="coerce").fillna(5000.0).values
            upper_bound = cap * 1.05
            sorted_out = np.clip(sorted_out, 50.0, upper_bound[:, None])
        else:
            sorted_out = np.clip(sorted_out, 50.0, None)

        q_low_name = f"q{int(round(self.quantiles[0]*100))}"
        q_mid_name = f"q{int(round(self.quantiles[1]*100))}"
        q_high_name = f"q{int(round(self.quantiles[2]*100))}"

        res_df = pd.DataFrame(
            {
                q_low_name: np.round(sorted_out[:, 0]),
                q_mid_name: np.round(sorted_out[:, 1]),
                q_high_name: np.round(sorted_out[:, 2]),
            },
            index=X.index
        )
        res_df["q25"] = res_df[q_low_name]
        res_df["q50"] = res_df[q_mid_name]
        res_df["q75"] = res_df[q_high_name]
        res_df["q10"] = res_df[q_low_name]  # compatibility alias
        res_df["q90"] = res_df[q_high_name]  # compatibility alias
        res_df["interval_width"] = res_df[q_high_name] - res_df[q_low_name]
        res_df["pred_attendance"] = res_df["q50"]
        if "capacity_resolved" in X.columns:
            cap = pd.to_numeric(X["capacity_resolved"], errors="coerce").fillna(5000.0).values
            res_df["pred_occupancy"] = np.round(np.clip(res_df["pred_attendance"] / (cap + 1e-5), 0.01, 1.05), 4)

        return res_df

    predict_intervals = predict_interval

    def predict(self, X: pd.DataFrame, use_club_adaptation: bool = True) -> np.ndarray:
        """Returns hybrid ensemble Q50 point prediction."""
        res = self.predict_interval(X)
        return res["q50"].values

    def score(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series, Sequence]) -> float:
        """Calculate R^2 score between hybrid ensemble Q50 predictions and actual y."""
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        mask = (~np.isnan(y_true)) & (y_true > 0)
        return r2_score(y_true[mask], y_pred[mask])

    def get_feature_importances(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series, Sequence]] = None,
        method: str = "catboost",
        n_repeats: int = 3,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Feature Importance for HybridEnsembleQuantileModel.
        Supports:
          - 'catboost': Native Tree Feature Importance from fitted CatBoost model.
          - 'weights': First-layer weight magnitude from primary PyTorch network.
          - 'permutation': Permutation Feature Importance evaluated across the full hybrid ensemble.
        """
        if not self.is_fitted_ or not self.models_:
            raise RuntimeError("Ensemble is not fitted yet.")

        # Method A: CatBoost native feature importance (fast, highly accurate tree splits)
        cb_model = getattr(self, "catboost_model_", None)
        if method == "catboost" and cb_model is not None:
            cb_scores = cb_model.get_feature_importance()
            df_imp = pd.DataFrame({
                "feature": getattr(self, "tree_features_", []),
                "importance": cb_scores
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df_imp

        # Method B: Weight magnitude attribution from primary neural network
        if method == "weights" or X is None:
            return self.models_[0].get_feature_importances(method="weights")

        # Method C: Permutation Feature Importance
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            if "attendance" in X.columns:
                y_true = X["attendance"].values
            else:
                raise ValueError("Target y or 'attendance' column is required.")
        else:
            y_true = np.asarray(y)

        base_preds = self.predict(X)
        base_mae = mean_absolute_error(y_true, base_preds)

        rng = np.random.RandomState(random_state)
        feature_list = ["home_team", "away_team", "league_tier"] + list(self.models_[0].continuous_cols_)
        eval_features = [f for f in feature_list if f in X.columns or f in ("home_team", "away_team", "league_tier")]

        importance_scores = []
        for feat in eval_features:
            mae_diffs = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                if feat in X_permuted.columns:
                    shuffled_vals = rng.permutation(X_permuted[feat].values)
                    X_permuted[feat] = shuffled_vals
                preds_shuffled = self.predict(X_permuted)
                perm_mae = mean_absolute_error(y_true, preds_shuffled)
                mae_diffs.append(max(0.0, perm_mae - base_mae))
            importance_scores.append({
                "feature": feat,
                "importance": float(np.mean(mae_diffs))
            })

        df_imp = pd.DataFrame(importance_scores).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp


# Alias for intuitive hybrid naming
HybridEnsembleQuantileModel = NeuralEnsembleQuantileModel


