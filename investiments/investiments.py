from __future__ import annotations

from pathlib import Path
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

SEED = 42
LOOKBACK = 10
VALIDATION_DAYS = 7
TICKER = "PETR4.SA"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def normalize_columns(df: pd.DataFrame):
    rename_map = {
        "date": "Date",
        "data": "Date",
        "close": "Close",
        "adj close": "Close",
        "fechamento": "Close",
        "preco fechamento": "Close",
        "preco_fechamento": "Close",
    }

    columns_normalized = {}
    for col in df.columns:
        key = str(col).strip().lower()
        columns_normalized[col] = rename_map.get(key, col)

    out = df.rename(columns=columns_normalized)
    if "Date" not in out.columns or "Close" not in out.columns:
        return None

    out = out[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().sort_values("Date").reset_index(drop=True)
    return out


def load_local_csv(base_dir: Path):
    candidates = [base_dir / "data", base_dir / "dataset"]
    csv_files = []
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            csv_files.extend(sorted(folder.glob("*.csv")))

    for csv_path in csv_files:
        try:
            raw = pd.read_csv(csv_path)
            parsed = normalize_columns(raw)
            if parsed is not None and len(parsed) > LOOKBACK + VALIDATION_DAYS:
                print(f"[INFO] Usando CSV local: {csv_path.name}")
                return parsed, f"csv_local:{csv_path.name}"
        except Exception as exc:
            print(f"[INFO] Ignorando arquivo {csv_path.name}: {exc}")

    return None


def load_fallback_yfinance(ticker: str, start_date: str, end_date: str):
    print(f"[INFO] Baixando dados com yfinance para {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Falha ao baixar dados com yfinance e nenhum CSV local valido foi encontrado.")

    out = df.reset_index()[["Date", "Close"]].copy()
    # yfinance can return a 2D "Close" representation depending on pandas layout.
    # Convert it to a flat numeric Series to keep feature arrays strictly 2D.
    if isinstance(out["Close"], pd.DataFrame):
        out["Close"] = out["Close"].iloc[:, 0]
    out = out.dropna().sort_values("Date").reset_index(drop=True)
    print(f"[INFO] Dados baixados: {len(out)} registros")
    return out, f"yfinance:{ticker}"


def build_supervised(df: pd.DataFrame, lookback: int):
    close_values = np.asarray(df["Close"], dtype=np.float64).reshape(-1)
    dates = df["Date"].to_numpy()

    x_data = []
    y_data = []
    y_dates = []

    for idx in range(lookback, len(close_values)):
        x_data.append(close_values[idx - lookback : idx])
        y_data.append(close_values[idx])
        y_dates.append(dates[idx])

    x_data = np.asarray(x_data, dtype=np.float64).reshape(-1, lookback)
    y_data = np.asarray(y_data, dtype=np.float64).reshape(-1, 1)
    y_dates = np.asarray(y_dates)

    if len(x_data) <= VALIDATION_DAYS:
        raise RuntimeError("Dados insuficientes para separar treino e validacao.")

    split_index = len(x_data) - VALIDATION_DAYS

    x_train_raw = x_data[:split_index]
    y_train_raw = y_data[:split_index]
    x_val_raw = x_data[split_index:]
    y_val_raw = y_data[split_index:]
    val_dates = y_dates[split_index:]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(x_train_raw)
    x_val = x_scaler.transform(x_val_raw)
    y_train = y_scaler.fit_transform(y_train_raw).ravel()

    return x_train, y_train, x_val, y_val_raw.ravel(), val_dates, y_scaler, x_scaler, x_val_raw


def train_mlp(x_train: np.ndarray, y_train: np.ndarray):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1200,
        random_state=SEED,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, x_val_raw: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    previous_close = x_val_raw[:, -1]
    true_direction = np.sign(y_true - previous_close)
    pred_direction = np.sign(y_pred - previous_close)
    direction_acc = (true_direction == pred_direction).mean() * 100

    return mae, rmse, mape, direction_acc


def plot_loss_curve(loss_curve: list[float], output_path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_curve, label="Perda de treino")
    plt.title("Curva de Perda - MLP")
    plt.xlabel("Iteracao")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_validation(val_dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(val_dates, y_true, marker="o", label="Real")
    plt.plot(val_dates, y_pred, marker="x", label="Previsto")
    plt.title("Validacao - Fechamento Real vs Previsto")
    plt.xlabel("Data")
    plt.ylabel("Preco de Fechamento")
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    set_seed(SEED)
    base_dir = Path(__file__).resolve().parent

    loaded = load_local_csv(base_dir)
    if loaded is None:
        series_df, source_name = load_fallback_yfinance(TICKER, START_DATE, END_DATE)
    else:
        series_df, source_name = loaded

    print(f"[INFO] Fonte de dados: {source_name}")
    print(f"[INFO] Janela de entrada (lookback): {LOOKBACK} dias")
    print(f"[INFO] Dias de validacao: {VALIDATION_DAYS}")

    x_train, y_train, x_val, y_val, val_dates, y_scaler, x_scaler, x_val_raw = build_supervised(
        series_df,
        LOOKBACK,
    )

    model = train_mlp(x_train, y_train)

    y_pred_scaled = model.predict(x_val).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()

    mae, rmse, mape, direction_acc = evaluate_predictions(y_val, y_pred, x_val_raw)

    print(f"\n[RESULTADO] MAE: {mae:.4f}")
    print(f"[RESULTADO] RMSE: {rmse:.4f}")
    print(f"[RESULTADO] MAPE: {mape:.2f}%")
    print(f"[RESULTADO] Acuracia de direcao: {direction_acc:.2f}%")

    model_path = base_dir / "mlp_investiments_model.joblib"
    loss_path = base_dir / "training_loss.png"
    validation_path = base_dir / "validation_prediction.png"

    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "lookback": LOOKBACK,
            "ticker": TICKER,
            "source": source_name,
        },
        model_path,
    )

    plot_loss_curve(model.loss_curve_, loss_path)
    plot_validation(val_dates, y_val, y_pred, validation_path)

    print(f"[RESULTADO] Modelo salvo: {model_path.name}")
    print(f"[RESULTADO] Grafico salvo: {loss_path.name}")
    print(f"[RESULTADO] Grafico salvo: {validation_path.name}")


if __name__ == "__main__":
    main()
