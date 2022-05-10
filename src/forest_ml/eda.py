from pandas_profiling import ProfileReport
import pandas as pd
import os
from pathlib import Path


def create_eda(
    from_csv: Path = Path("data/train.csv"),
    to_html: str = "data/eda.html",
    open_report: bool = True,
) -> str:
    """Create pandas_profiling report from from_csv file to to_html file.
    Open report in browser if open_report=True
    """
    print("gello")
    df = pd.read_csv(from_csv, index_col="Id")
    ProfileReport(df).to_file(to_html.replace("\\", "/"))
    if open_report:
        os.startfile(to_html.replace("/", "\\"))
    return to_html
