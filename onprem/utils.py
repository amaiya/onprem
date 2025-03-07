"""some utility functions"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_utils.ipynb.

# %% auto 0
__all__ = ['CAPTION_STR', 'download', 'get_datadir', 'split_list', 'filtered_generator', 'segment', 'remove_sentence',
           'contains_sentence', 'md_to_df', 'html_to_df', 'df_to_md', 'SafeFormatter', 'format_string',
           'get_template_vars']

# %% ../nbs/02_utils.ipynb 3
import os.path
import requests
import sys
import re

#--------------------------------------
# App Utilities
#--------------------------------------

def download(url, filename, verify=False):
    with open(filename, "wb") as f:
        response = requests.get(url, stream=True, verify=verify)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            # print(total)
            for data in response.iter_content(
                chunk_size=max(int(total / 1000), 1024 * 1024)
            ):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write("\r[{}{}]".format("█" * done, "." * (50 - done)))
                sys.stdout.flush()


def get_datadir():
    home = os.path.expanduser("~")
    datadir = os.path.join(home, "onprem_data")
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    return datadir




# %% ../nbs/02_utils.ipynb 4
#--------------------------------------
# Data Utilities
#--------------------------------------
def split_list(input_list, chunk_size):
    """
    Split list into chunks
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


def filtered_generator(generator, criteria=[]):
    """
    Filters a generator based on a given predicate function.

    Args:
        generator: The generator to filter.
        criteria: List of functions that take an element from the generator 
                   and return True if the element should be included, 
                   False otherwise.

    Yields:
        Elements from the original generator that satisfy the predicate.
    """
    for item in generator:
        if all(criterion(item) for criterion in criteria):
            yield item


from syntok import segmenter
import textwrap
def segment(text:str, unit:str='paragraph', maxchars:int=2048):
    """
    Segments text into a list of paragraphs or sentences depending on value of `unit` 
    (one of `{'paragraph', 'sentence'}`. The `maxchars` parameter is the maximum size
    of any unit of text.
    """
    units = []
    for paragraph in segmenter.analyze(text):
        sentences = []
        for sentence in paragraph:
            text = ""
            for token in sentence:
                text += f'{token.spacing}{token.value}'
            sentences.append(text)
        if unit == 'sentence':
            units.extend(sentences)
        else:
            units.append(" ".join(sentences))
    chunks = []
    for s in units:
        parts = textwrap.wrap(s, maxchars, break_long_words=False)
        chunks.extend(parts)
    return chunks


def remove_sentence(sentence, text, remove_follow=False, flags=re.IGNORECASE):
    """
    Removes a sentence or phrase from text ignoring whether
    tokens are delimited by spaces or newlines or tabs.

    If  `remove_follow=True`, then subsequent text until the first newline
    is also removed.
    """
    if remove_follow:
    	pattern = r'\s*'.join(map(re.escape, sentence.split())) + r'[\s\S]*?(?:\n\s*){2,}'
    	return re.sub(pattern, '\n\n', text, flags=flags).strip()
    else:
        pattern = r'\s*'.join(map(re.escape, sentence.split())) + r'\s*'
        return re.sub(pattern, '', text, flags=flags)


def contains_sentence(sentence, text):
    """
    Returns True if sentence is contained in text ignoring whether
    tokens are delmited by spaces or newlines or tabs.
    """
    pattern = r'\s*'.join(map(re.escape, sentence.split())) + r'\s*'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


from typing import Any

from io import StringIO


def md_to_df(md_str: str) -> Any:
    """Convert Markdown to dataframe."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "You must install the `pandas` package to use this node parser."
        )

    # Replace " by "" in md_str
    md_str = md_str.replace('"', '""')

    # Replace markdown pipe tables with commas
    md_str = md_str.replace("|", '","')

    # Remove the second line (table header separator)
    lines = md_str.split("\n")
    md_str = "\n".join(lines[:1] + lines[2:])

    # Remove the first and last second char of the line (the pipes, transformed to ",")
    lines = md_str.split("\n")
    md_str = "\n".join([line[2:-2] for line in lines])

    # Check if the table is empty
    if len(md_str) == 0:
        return None

    # Use pandas to read the CSV string into a DataFrame
    return pd.read_csv(StringIO(md_str))


def html_to_df(html_str: str) -> Any:
    """Convert HTML to dataframe."""
    try:
        from lxml import html
    except ImportError:
        raise ImportError(
            "You must install the `lxml` package to use this node parser."
        )

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "You must install the `pandas` package to use this node parser."
        )

    tree = html.fromstring(html_str)
    table_element = tree.xpath("//table")[0]
    rows = table_element.xpath(".//tr")
    try:
        colnames = table_element.xpath(".//th")
        colnames = [col.text for col in colnames]
    except Exception as e:
        print(str(e))
        colnames = []

    data = []
    for row in rows:
        cols = row.xpath(".//td")
        cols = [c.text.strip() if c.text is not None else "" for c in cols]
        if len(cols) == 0: continue
        data.append(cols)

    # Check if the table is empty
    if len(data) == 0:
        return None

    # Check if the all rows have the same number of columns
    if not all(len(row) == len(data[0]) for row in data):
        return None
    if colnames:
        return pd.DataFrame(data[0:], columns=colnames)
    else:
        return pd.DataFrame(data[1:], columns=data[0])


CAPTION_STR = "The following table in markdown format has the caption"
def df_to_md(df, caption=None):
    """
    Converts pd.Dataframe to markdown
    """
    table_md = "|"
    for col_name, col in df.items():
        table_md += f"{col_name}|"
    table_md += "\n|"
    for col_name, col in df.items():
        table_md += f"---|"
    table_md += "\n"
    for row in df.itertuples():
        table_md += "|"
        for col in row[1:]:
            table_md += f"{col}|"
        table_md += "\n"
    if caption:
        table_summary = f"{CAPTION_STR}: {caption} "
    table_summary += f"The following table in markdown format includes this list of columns:\n"
    for col in df.columns:
        table_summary += f"- {col}\n"

    return f'{caption}\n\n{table_summary}\n{table_md}' if caption else f'{table_summary}\n{table_md}'


# %% ../nbs/02_utils.ipynb 5
from typing import Dict, List, Optional
import re

#--------------------------------------
# Prompt Utilities
#--------------------------------------

class SafeFormatter:
    """
    Safe string formatter that does not raise KeyError if key is missing.
    Adapted from llama_index.
    """

    def __init__(self, format_dict: Optional[Dict[str, str]] = None):
        self.format_dict = format_dict or {}

    def format(self, format_string: str) -> str:
        return re.sub(r"\{([^{}]+)\}", self._replace_match, format_string)

    def parse(self, format_string: str) -> List[str]:
        return re.findall(r"\{([^{}]+)\}", format_string)

    def _replace_match(self, match: re.Match) -> str:
        key = match.group(1)
        return str(self.format_dict.get(key, match.group(0)))


def format_string(string_to_format: str, **kwargs: str) -> str:
    """Format a string with kwargs"""
    formatter = SafeFormatter(format_dict=kwargs)
    return formatter.format(string_to_format)


def get_template_vars(template_str: str) -> List[str]:
    """Get template variables from a template string."""
    variables = []
    formatter = SafeFormatter()

    for variable_name in formatter.parse(template_str):
        if variable_name:
            variables.append(variable_name)

    return [v for v in variables if " " not in v and "\n" not in v]
