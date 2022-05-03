from pandas_profiling import ProfileReport
import pandas as pd
import os


def create_eda(from_csv="data/train.csv", to_html="data/eda.html", open_report=True):
    """Create pandas_profiling report from from_csv file to to_html file.
    Open report in browser if open_report=True
    """
    print("gello")
    df = pd.read_csv(from_csv, index_col="Id")
    ProfileReport(df).to_file(to_html.replace("\\", "/"))
    if open_report:
        os.startfile(to_html.replace("/", "\\"))
    return to_html
