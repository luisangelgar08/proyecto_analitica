from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

# =========================
# INPUTS
# =========================
IN_TAB = "data/gold/tabular/loan_features.parquet"

# Ranking de features (el que generaste con el EDA)
RANK_PATHS = [
    Path("data/gold/eda/features_28/04_feature_ranking_28.csv"),
    Path("data/gold/afe/evidence/04_feature_ranking_28.csv"),
    Path("data/gold/afe/evidence/features_28/04_feature_ranking_28.csv"),
    Path("04_feature_ranking_28.csv"),
]

# =========================
# OUTPUTS
# =========================
OUT_DIR = Path("data/gold/afe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVID_DIR = OUT_DIR / "evidence"
EVID_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================
# "28" -> usa todas las mean_num_ (28)
# "22" -> filtra usando ranking: std>0 y null_ratio<=0.20
MODE = "22"

MAX_NULL_RATIO = 0.20
MIN_STD = 1e-12

DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "6GB"
DUCKDB_MAX_TEMP = "200GB"


def find_rank_file() -> Path:
    """Encuentra el ranking en alguna ruta conocida."""
    for p in RANK_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No encontré 04_feature_ranking_28.csv.\n"
        "Ponlo en: data/gold/eda/features_28/ o data/gold/afe/evidence/ o en la raíz del proyecto."
    )


def get_mean_features(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Devuelve todas las columnas mean_num_* presentes en loan_features."""
    cols = con.execute("DESCRIBE gf").df()["column_name"].tolist()
    mean_cols = [c for c in cols if c.startswith("mean_num_")]
    if not mean_cols:
        raise ValueError("No encontré columnas mean_num_* en loan_features.parquet.")
    return mean_cols


def select_features(con: duckdb.DuckDBPyConnection) -> tuple[list[str], pd.DataFrame | None]:
    """
    Retorna:
      - kept: lista de features mean_num_* que se usarán
      - decision_df: dataframe con decisión (solo cuando MODE='22'), o None si MODE='28'
    """
    mean_cols = get_mean_features(con)

    if MODE == "28":
        return mean_cols, None

    # MODE == "22"
    rank_file = find_rank_file()
    rank = pd.read_csv(rank_file)

    required = {"feature", "null_ratio", "std"}
    if not required.issubset(rank.columns):
        raise ValueError(f"El ranking no tiene columnas requeridas {required}. Tiene: {list(rank.columns)}")

    # Solo consideramos mean_num_*
    rank = rank[rank["feature"].astype(str).str.startswith("mean_num_")].copy()

    # Reglas
    rank["keep"] = (rank["null_ratio"] <= MAX_NULL_RATIO) & (rank["std"].fillna(0) > MIN_STD)

    def reason(r):
        if pd.isna(r["std"]) or r["std"] <= MIN_STD:
            return "EXCLUDE_constante_std0"
        if r["null_ratio"] > MAX_NULL_RATIO:
            return f"EXCLUDE_missingness_gt_{MAX_NULL_RATIO}"
        return "KEEP"

    rank["decision"] = rank.apply(reason, axis=1)

    kept = rank.loc[rank["keep"], "feature"].astype(str).tolist()

    # Seguridad: solo las que existan realmente en loan_features
    kept = [c for c in kept if c in mean_cols]

    if len(kept) == 0:
        raise ValueError("Después del filtro MODE=22 no quedó ninguna feature. Revisa MAX_NULL_RATIO/MIN_STD o el ranking.")

    return kept, rank


def build_afe_matrix(con: duckdb.DuckDBPyConnection, kept: list[str], out_mat: Path) -> None:
    """
    Construye matriz con z-scores:
      z = (coalesce(x, mu) - mu) / sd
    usando stats (mu, sd) calculados en 1 fila.
    """
    # stats
    stats_exprs = [f"avg({c}) AS mu_{c}, stddev_samp({c}) AS sd_{c}" for c in kept]
    con.execute("DROP TABLE IF EXISTS stats;")
    con.execute(f"CREATE TEMP TABLE stats AS SELECT {', '.join(stats_exprs)} FROM gf;")

    # z-score con CROSS JOIN stats (evita subqueries repetidas)
    z_exprs = []
    for c in kept:
        z_exprs.append(
            f"""
            CASE
              WHEN stats.sd_{c} IS NULL OR stats.sd_{c}=0 THEN 0
              ELSE (coalesce(gf.{c}, stats.mu_{c}) - stats.mu_{c}) / stats.sd_{c}
            END AS z_{c}
            """.strip()
        )

    query = f"""
    SELECT gf.loan_id,
           {', '.join(z_exprs)}
    FROM gf
    CROSS JOIN stats
    """

    con.execute(f"""
        COPY ({query})
        TO '{str(out_mat).replace("\\", "/")}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """)


def main():
    print("Running 05_prepare_afe_matrix.py")
    print("MODE:", MODE)

    # DuckDB setup
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute("SET preserve_insertion_order=false;")

    tmp = str(TMP_DIR.resolve()).replace("\\", "/")
    con.execute(f"PRAGMA temp_directory='{tmp}';")
    con.execute(f"PRAGMA max_temp_directory_size='{DUCKDB_MAX_TEMP}';")

    # Input view
    con.execute(f"CREATE VIEW gf AS SELECT * FROM read_parquet('{IN_TAB}', hive_partitioning=false);")

    # Feature selection
    kept, decision_df = select_features(con)
    print("Selected features:", len(kept))
    print("First 5:", kept[:5])

    # Outputs
    out_mat = OUT_DIR / f"afe_matrix_{MODE}.parquet"
    out_list = OUT_DIR / f"afe_feature_list_{MODE}.txt"
    out_list.write_text("\n".join(kept), encoding="utf-8")
    print("OK ->", out_list)

    if decision_df is not None:
        out_dec = EVID_DIR / f"feature_filter_decision_{MODE}.csv"
        decision_df.to_csv(out_dec, index=False, encoding="utf-8")
        print("OK ->", out_dec)

    # Build matrix
    build_afe_matrix(con, kept, out_mat)
    print("OK ->", out_mat)

    # Quick validate rows
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{str(out_mat).replace("\\", "/")}');").fetchone()[0]
    print("Rows (loans):", n)
    print("DONE")


if __name__ == "__main__":
    main()