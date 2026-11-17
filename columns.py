"""
The 64 financial ratios in data/polish_bankruptcy.csv.

The CSV uses descriptive column names. Each entry gives the name used in the
CSV, the original UCI attribute (Attr1 to Attr64) and its published formula.
Only the headers were renamed: values, column order, `year` and `class` are
unchanged. Columns 48 and 49 use the published subtraction formula rather than
assuming the usual EBITDA definition.
"""

SOURCE = "https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data"

# (name in the CSV, original UCI attribute, published formula)
COLUMNS = [
    ("net_profit_to_total_assets", "Attr1", "net profit / total assets"),
    ("total_liabilities_to_total_assets", "Attr2", "total liabilities / total assets"),
    ("working_capital_to_total_assets", "Attr3", "working capital / total assets"),
    (
        "current_assets_to_short_term_liabilities",
        "Attr4",
        "current assets / short-term liabilities",
    ),
    (
        "net_liquid_assets_coverage_days",
        "Attr5",
        "((cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)) * 365",
    ),
    ("retained_earnings_to_total_assets", "Attr6", "retained earnings / total assets"),
    ("ebit_to_total_assets", "Attr7", "EBIT / total assets"),
    ("book_equity_to_total_liabilities", "Attr8", "book value of equity / total liabilities"),
    ("sales_to_total_assets", "Attr9", "sales / total assets"),
    ("equity_to_total_assets", "Attr10", "equity / total assets"),
    (
        "gross_profit_plus_extraordinary_items_plus_financial_expenses_to_total_assets",
        "Attr11",
        "(gross profit + extraordinary items + financial expenses) / total assets",
    ),
    ("gross_profit_to_short_term_liabilities", "Attr12", "gross profit / short-term liabilities"),
    ("gross_profit_plus_depreciation_to_sales", "Attr13", "(gross profit + depreciation) / sales"),
    (
        "gross_profit_plus_interest_to_total_assets",
        "Attr14",
        "(gross profit + interest) / total assets",
    ),
    (
        "total_liabilities_to_gross_profit_plus_depreciation_days",
        "Attr15",
        "(total liabilities * 365) / (gross profit + depreciation)",
    ),
    (
        "gross_profit_plus_depreciation_to_total_liabilities",
        "Attr16",
        "(gross profit + depreciation) / total liabilities",
    ),
    ("total_assets_to_total_liabilities", "Attr17", "total assets / total liabilities"),
    ("gross_profit_to_total_assets", "Attr18", "gross profit / total assets"),
    ("gross_profit_to_sales", "Attr19", "gross profit / sales"),
    ("inventory_to_sales_days", "Attr20", "(inventory * 365) / sales"),
    ("sales_to_previous_year_sales", "Attr21", "sales (n) / sales (n-1)"),
    (
        "operating_profit_to_total_assets",
        "Attr22",
        "profit on operating activities / total assets",
    ),
    ("net_profit_to_sales", "Attr23", "net profit / sales"),
    (
        "three_year_gross_profit_to_total_assets",
        "Attr24",
        "gross profit (in 3 years) / total assets",
    ),
    (
        "equity_minus_share_capital_to_total_assets",
        "Attr25",
        "(equity - share capital) / total assets",
    ),
    (
        "net_profit_plus_depreciation_to_total_liabilities",
        "Attr26",
        "(net profit + depreciation) / total liabilities",
    ),
    (
        "operating_profit_to_financial_expenses",
        "Attr27",
        "profit on operating activities / financial expenses",
    ),
    ("working_capital_to_fixed_assets", "Attr28", "working capital / fixed assets"),
    ("log_total_assets", "Attr29", "logarithm of total assets"),
    ("total_liabilities_minus_cash_to_sales", "Attr30", "(total liabilities - cash) / sales"),
    ("gross_profit_plus_interest_to_sales", "Attr31", "(gross profit + interest) / sales"),
    (
        "current_liabilities_to_cost_of_goods_sold_days",
        "Attr32",
        "(current liabilities * 365) / cost of products sold",
    ),
    (
        "operating_expenses_to_short_term_liabilities",
        "Attr33",
        "operating expenses / short-term liabilities",
    ),
    (
        "operating_expenses_to_total_liabilities",
        "Attr34",
        "operating expenses / total liabilities",
    ),
    ("profit_on_sales_to_total_assets", "Attr35", "profit on sales / total assets"),
    ("total_sales_to_total_assets", "Attr36", "total sales / total assets"),
    (
        "current_assets_minus_inventory_to_long_term_liabilities",
        "Attr37",
        "(current assets - inventories) / long-term liabilities",
    ),
    ("constant_capital_to_total_assets", "Attr38", "constant capital / total assets"),
    ("profit_on_sales_to_sales", "Attr39", "profit on sales / sales"),
    (
        "current_assets_minus_inventory_minus_receivables_to_short_term_liabilities",
        "Attr40",
        "(current assets - inventory - receivables) / short-term liabilities",
    ),
    (
        "total_liabilities_to_operating_profit_plus_depreciation_12_over_365",
        "Attr41",
        "total liabilities / ((profit on operating activities + depreciation) * (12/365))",
    ),
    ("operating_profit_to_sales", "Attr42", "profit on operating activities / sales"),
    (
        "receivables_plus_inventory_turnover_days",
        "Attr43",
        "rotation receivables + inventory turnover in days",
    ),
    ("receivables_to_sales_days", "Attr44", "(receivables * 365) / sales"),
    ("net_profit_to_inventory", "Attr45", "net profit / inventory"),
    (
        "current_assets_minus_inventory_to_short_term_liabilities",
        "Attr46",
        "(current assets - inventory) / short-term liabilities",
    ),
    (
        "inventory_to_cost_of_goods_sold_days",
        "Attr47",
        "(inventory * 365) / cost of products sold",
    ),
    (
        "operating_profit_minus_depreciation_to_total_assets",
        "Attr48",
        "EBITDA (profit on operating activities - depreciation) / total assets",
    ),
    (
        "operating_profit_minus_depreciation_to_sales",
        "Attr49",
        "EBITDA (profit on operating activities - depreciation) / sales",
    ),
    ("current_assets_to_total_liabilities", "Attr50", "current assets / total liabilities"),
    ("short_term_liabilities_to_total_assets", "Attr51", "short-term liabilities / total assets"),
    (
        "short_term_liabilities_to_cost_of_goods_sold_days",
        "Attr52",
        "(short-term liabilities * 365) / cost of products sold",
    ),
    ("equity_to_fixed_assets", "Attr53", "equity / fixed assets"),
    ("constant_capital_to_fixed_assets", "Attr54", "constant capital / fixed assets"),
    ("working_capital", "Attr55", "working capital"),
    (
        "sales_minus_cost_of_goods_sold_to_sales",
        "Attr56",
        "(sales - cost of products sold) / sales",
    ),
    (
        "current_assets_minus_inventory_minus_short_term_liabilities_to_sales_minus_gross_profit_minus_depreciation",
        "Attr57",
        "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    ),
    ("total_costs_to_total_sales", "Attr58", "total costs / total sales"),
    ("long_term_liabilities_to_equity", "Attr59", "long-term liabilities / equity"),
    ("sales_to_inventory", "Attr60", "sales / inventory"),
    ("sales_to_receivables", "Attr61", "sales / receivables"),
    ("short_term_liabilities_to_sales_days", "Attr62", "(short-term liabilities * 365) / sales"),
    ("sales_to_short_term_liabilities", "Attr63", "sales / short-term liabilities"),
    ("sales_to_fixed_assets", "Attr64", "sales / fixed assets"),
]

DEFINITIONS = {name: formula for name, _, formula in COLUMNS}
UCI_NAMES = {name: attribute for name, attribute, _ in COLUMNS}
