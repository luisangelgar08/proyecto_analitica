from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

IN_ROOT = Path("data/parquet/performance")
OUT_ROOT = Path("data/parquet_silver/performance")

# 32 campos del Monthly Performance Data File (snake_case)
PERF32 = [
    "loan_sequence_number",                 # 1
    "monthly_reporting_period",             # 2
    "current_actual_upb",                   # 3
    "current_loan_delinquency_status",      # 4
    "loan_age",                             # 5
    "remaining_months_to_legal_maturity",   # 6
    "defect_settlement_date",               # 7
    "modification_flag",                    # 8
    "zero_balance_code",                    # 9
    "zero_balance_effective_date",          # 10
    "current_interest_rate",                # 11
    "current_non_interest_bearing_upb",     # 12
    "ddlpi",                                # 13
    "mi_recoveries",                        # 14
    "net_sale_proceeds",                    # 15
    "non_mi_recoveries",                    # 16
    "expenses",                             # 17
    "legal_costs",                          # 18
    "maintenance_preservation_costs",       # 19
    "taxes_and_insurance",                  # 20
    "miscellaneous_expenses",               # 21
    "actual_loss_calculation",              # 22
    "cumulative_modification_cost",         # 23
    "interest_rate_step_indicator",         # 24
    "payment_deferral_flag",                # 25
    "estimated_loan_to_value_eltv",         # 26
    "zero_balance_removal_upb",             # 27
    "delinquent_accrued_interest",          # 28
    "delinquency_due_to_disaster",          # 29
    "borrower_assistance_status_code",      # 30
    "current_month_modification_cost",      # 31
    "interest_bearing_upb",                 # 32
]

FIRST32 = [f"c{i:03d}" for i in range(32)]
SILVER_SCHEMA = pa.schema([(c, pa.string()) for c in PERF32])

def convert_file(in_file: Path, out_file: Path, compression="zstd"):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(in_file)
    writer = pq.ParquetWriter(out_file, schema=SILVER_SCHEMA, compression=compression)

    rows = 0
    try:
        for rb in pf.iter_batches(batch_size=250_000, columns=FIRST32):
            # rb tiene columnas c000..c031 (string). Renombramos al layout oficial.
            tbl = pa.Table.from_batches([rb]).rename_columns(PERF32)
            # fuerza schema (por si arrow infiere algo raro)
            tbl = tbl.cast(SILVER_SCHEMA)
            writer.write_table(tbl)
            rows += tbl.num_rows
    finally:
        writer.close()

    return rows

def main():
    # Busca todos los parquet por año/quarter
    in_files = sorted(IN_ROOT.rglob("*.parquet"))
    print("Parquets encontrados:", len(in_files))

    for f in tqdm(in_files, desc="Building SILVER"):
        # replica particiones year=YYYY/quarter=Qn
        rel = f.relative_to(IN_ROOT)
        out_f = OUT_ROOT / rel
        rows = convert_file(f, out_f)
        # imprime solo algo corto
        tqdm.write(f"OK {rel} | rows={rows:,}")

if __name__ == "__main__":
    main()