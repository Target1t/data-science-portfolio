from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


def normalized_identifier(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-end", default="2010-12-01")
    parser.add_argument("--top-customers", type=int, default=200)
    parser.add_argument("--top-products", type=int, default=500)
    parser.add_argument("--min-selection-days", type=int, default=2)
    args = parser.parse_args()

    selection_end = pd.Timestamp(args.selection_end)
    daily: defaultdict[tuple[datetime, str, str, str], float] = defaultdict(float)
    rejection = Counter()
    workbook = load_workbook(args.input, read_only=True, data_only=True)
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        header = list(next(rows))
        positions = {name: index for index, name in enumerate(header)}
        for row in rows:
            rejection["raw_rows"] += 1
            invoice = normalized_identifier(row[positions["Invoice"]])
            product = normalized_identifier(row[positions["StockCode"]])
            customer = normalized_identifier(row[positions["Customer ID"]])
            date = row[positions["InvoiceDate"]]
            quantity = row[positions["Quantity"]]
            country = normalized_identifier(row[positions["Country"]]) or "__MISSING__"
            if invoice.upper().startswith("C"):
                rejection["cancellation"] += 1
                continue
            if not product or not customer or not isinstance(date, datetime):
                rejection["missing_key"] += 1
                continue
            try:
                quantity = float(quantity)
            except (TypeError, ValueError):
                rejection["invalid_quantity"] += 1
                continue
            if quantity <= 0:
                rejection["nonpositive_quantity"] += 1
                continue
            daily[(date.replace(hour=0, minute=0, second=0, microsecond=0), customer, product, country)] += quantity

    frame = pd.DataFrame(
        ((*key, value) for key, value in daily.items()),
        columns=["date", "customer_id", "product_id", "country", "quantity"],
    )
    selection = frame.loc[frame.date.lt(selection_end)].copy()
    customer_rank = selection.groupby("customer_id").quantity.sum().nlargest(args.top_customers).index
    product_rank = selection.groupby("product_id").quantity.sum().nlargest(args.top_products).index
    eligible = selection.loc[
        selection.customer_id.isin(customer_rank) & selection.product_id.isin(product_rank)
    ]
    pair_days = eligible.groupby(["customer_id", "product_id"]).date.nunique()
    eligible_pairs = set(pair_days.loc[pair_days.ge(args.min_selection_days)].index)
    frame["pair"] = list(zip(frame.customer_id, frame.product_id))
    frame = frame.loc[frame.pair.isin(eligible_pairs)].drop(columns="pair")
    frame["series_id"] = frame.customer_id + "::" + frame.product_id
    frame["group_id"] = frame.product_id
    frame = frame.sort_values(["date", "series_id"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "transactions_daily.csv"
    frame.to_csv(output, index=False, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "source": "UCI Online Retail II",
        "source_doi": "10.24432/C5CG6D",
        "license": "CC BY 4.0",
        "selection_end_exclusive": str(selection_end.date()),
        "selection_uses_future": False,
        "top_customers": args.top_customers,
        "top_products": args.top_products,
        "min_selection_days": args.min_selection_days,
        "raw_audit": dict(rejection),
        "daily_rows": int(len(frame)),
        "series": int(frame.series_id.nunique()),
        "groups": int(frame.group_id.nunique()),
        "customers": int(frame.customer_id.nunique()),
        "date_min": str(frame.date.min().date()),
        "date_max": str(frame.date.max().date()),
        "quantity_sum": float(frame.quantity.sum()),
        "output_sha256": digest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
