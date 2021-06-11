import os
import pathlib

import pandas as pd
from jinja2 import Environment, FileSystemLoader

HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
DATA = HERE.joinpath("data", "DUB_website_main_v2.txt")
environment = Environment(
    autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False
)

index_template = environment.get_template("index.html")


def get_df() -> pd.DataFrame:
    """Get the pre-processed dataframe."""
    df = pd.read_csv(DATA, sep="\t")
    return df


def html():
    df = get_df()
    main_keys = [
        "geneID",
        "PubMed_papers",
        "fraction_cell_lines_dependent_on_DUB",
        "pvalue",
        "p.adjust",
        "qvalue",
    ]
    main_data = {}
    for _, row in df.iterrows():
        main_data[row["DUB"]] = {k: row[k] for k in main_keys}
    rows = [(dub, *data.values()) for dub, data in main_data.items()]
    print(rows)

    terms_df = pd.DataFrame(rows, columns=["DUB", *main_keys])
    index_html = index_template.render(terms_df=terms_df)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)


if __name__ == "__main__":
    html()
