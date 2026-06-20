"""Paso 4 del flujo: medir el sesgo del detector por grupo demografico.

Une las predicciones (data/preds.csv) con las etiquetas demograficas
(data/labels.csv) y calcula, por grupo:
  - FNR (tasa de falsos negativos): entre las imagenes que SON fake, que fraccion
    el detector marco como real. FNR alto = a ese grupo se le escapan mas deepfakes.
  - accuracy y AUROC (usando p_fake).
Ademas:
  - prueba chi-cuadrado de independencia (grupo vs acierto/fallo en los fake).
  - brecha de FNR entre el grupo peor y el mejor, con IC95% por bootstrap.

Salidas: data/outputs/bias_<group>.csv y bias_<group>.png

Uso:
    python src/analyze_bias.py --group race
    python src/analyze_bias.py --group gender
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 5  # grupos con menos fakes que esto no permiten estimar FNR


def fnr(group_df: pd.DataFrame) -> float:
    """FNR sobre las imagenes fake de un grupo: fraccion marcada como real."""
    fakes = group_df[group_df["true_label"] == "fake"]
    if len(fakes) == 0:
        return np.nan
    missed = (fakes["pred_label"] == "real").sum()
    return missed / len(fakes)


def accuracy(group_df: pd.DataFrame) -> float:
    if len(group_df) == 0:
        return np.nan
    correct = (group_df["pred_label"] == group_df["true_label"]).sum()
    return correct / len(group_df)


def auroc(group_df: pd.DataFrame):
    """AUROC con p_fake como score; requiere ambas clases presentes."""
    from sklearn.metrics import roc_auc_score

    y_true = (group_df["true_label"] == "fake").astype(int)
    if y_true.nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, group_df["p_fake"])
    except ValueError:
        return np.nan


def bootstrap_gap_ci(df: pd.DataFrame, group_col: str, worst: str, best: str,
                     n_boot: int = 2000, seed: int = 0):
    """IC95% (percentil) para la brecha FNR(worst) - FNR(best) por bootstrap.

    Re-muestrea con reemplazo las imagenes fake dentro de cada grupo.
    """
    rng = np.random.default_rng(seed)
    fakes = df[df["true_label"] == "fake"]
    worst_missed = (fakes[fakes[group_col] == worst]["pred_label"] == "real").to_numpy()
    best_missed = (fakes[fakes[group_col] == best]["pred_label"] == "real").to_numpy()
    if len(worst_missed) == 0 or len(best_missed) == 0:
        return (np.nan, np.nan)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.choice(worst_missed, size=len(worst_missed), replace=True).mean()
        b = rng.choice(best_missed, size=len(best_missed), replace=True).mean()
        gaps[i] = w - b
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return (float(lo), float(hi))


def chi2_independence(df: pd.DataFrame, group_col: str):
    """Chi-cuadrado: grupo vs (acierto/fallo) sobre las imagenes fake."""
    from scipy.stats import chi2_contingency

    fakes = df[df["true_label"] == "fake"].copy()
    fakes["missed"] = (fakes["pred_label"] == "real")
    table = pd.crosstab(fakes[group_col], fakes["missed"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def per_group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, gdf in df.groupby(group_col):
        n_fake = int((gdf["true_label"] == "fake").sum())
        n_real = int((gdf["true_label"] == "real").sum())
        rows.append({
            "group": group,
            "n_total": len(gdf),
            "n_fake": n_fake,
            "n_real": n_real,
            "FNR": fnr(gdf),
            "accuracy": accuracy(gdf),
            "AUROC": auroc(gdf),
        })
    table = pd.DataFrame(rows).sort_values("FNR", ascending=False, na_position="last")
    return table.reset_index(drop=True)


def plot_fnr(table: pd.DataFrame, group_col: str, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = table.dropna(subset=["FNR"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["group"].astype(str), plot_df["FNR"])
    ax.set_ylabel("FNR (fakes marcados como reales)")
    ax.set_xlabel(group_col)
    ax.set_title(f"Tasa de falsos negativos por {group_col}")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(plot_df["FNR"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analisis de sesgo del detector por grupo")
    parser.add_argument("--group", choices=["race", "gender"], default="race",
                        help="Eje demografico a analizar.")
    parser.add_argument("--preds", default="data/preds.csv")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--out_dir", default="data/outputs")
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    preds = pd.read_csv(args.preds)
    labels = pd.read_csv(args.labels)
    df = preds.merge(labels, on="path", how="inner")
    if df.empty:
        raise SystemExit("El merge de preds + labels quedo vacio (revisa la columna 'path').")

    group_col = args.group
    print(f"[info] {len(df)} imagenes con prediccion + demografia; eje = {group_col}")

    table = per_group_table(df, group_col)

    # Grupos validos para comparar la brecha: suficientes fakes.
    valid = table[(table["n_fake"] >= MIN_GROUP_SIZE) & table["FNR"].notna()]
    gap_info = {}
    if len(valid) >= 2:
        worst = valid.iloc[0]
        best = valid.iloc[-1]
        gap = worst["FNR"] - best["FNR"]
        lo, hi = bootstrap_gap_ci(df, group_col, worst["group"], best["group"], args.n_boot)
        gap_info = {
            "worst_group": worst["group"], "worst_FNR": worst["FNR"],
            "best_group": best["group"], "best_FNR": best["FNR"],
            "FNR_gap": gap, "gap_ci95_low": lo, "gap_ci95_high": hi,
        }
    else:
        print(f"[warn] menos de 2 grupos con >= {MIN_GROUP_SIZE} fakes; no se calcula brecha")

    chi = chi2_independence(df, group_col)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bias_{group_col}.csv"
    png_path = out_dir / f"bias_{group_col}.png"
    table.to_csv(csv_path, index=False)
    plot_fnr(table, group_col, png_path)

    pd.set_option("display.width", 120)
    print("\n=== FNR / accuracy / AUROC por grupo ===")
    print(table.to_string(index=False))
    if gap_info:
        print("\n=== Brecha de FNR (peor - mejor) ===")
        print(f"  peor:  {gap_info['worst_group']} (FNR={gap_info['worst_FNR']:.3f})")
        print(f"  mejor: {gap_info['best_group']} (FNR={gap_info['best_FNR']:.3f})")
        print(f"  brecha={gap_info['FNR_gap']:.3f}  IC95%=[{gap_info['gap_ci95_low']:.3f}, "
              f"{gap_info['gap_ci95_high']:.3f}]")
    if chi:
        print("\n=== Chi-cuadrado (grupo vs fallo en fakes) ===")
        print(f"  chi2={chi['chi2']:.3f}  dof={chi['dof']}  p={chi['p_value']:.4f}")

    print(f"\n[ok] tabla -> {csv_path}")
    print(f"[ok] figura -> {png_path}")


if __name__ == "__main__":
    main()
