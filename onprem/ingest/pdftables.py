"""extracts tables from PDFs"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_ingest.pdftables.ipynb.

# %% auto 0
__all__ = ['ingest_pdf', 'PDFTables']

# %% ../../nbs/01_ingest.pdftables.ipynb 3
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
from .. import utils as U

try:
    from gmft.auto import AutoTableFormatter, CroppedTable, TableDetector
    from gmft.pdf_bindings import PyPDFium2Document

    DETECTOR = TableDetector()
    FORMATTER = AutoTableFormatter()
    GMFT_INSTALLED = True
except ImportError:
    GMFT_INSTALLED = False


def ingest_pdf(pdf_path):  # produces list[CroppedTable]
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        tables += DETECTOR.extract(page)
    return tables, doc


class PDFTables:
    def __init__(self, dfs: Optional[List] = None, titles: Optional[List] = None):
        """
        Extracts tables from PDFs
        """
        self.dfs = dfs
        self.titles = titles

    @classmethod
    def from_file(cls, pdf_filename: str, verbose=False):
        """
        Extract tables and their captions.

        **Args**
        
        - pdf_filename: path to PDF file
        - verbose: If True, show progress
        """
        obj = cls()

        if not GMFT_INSTALLED:
            raise ImportError("Please install the gmft package: pip install gmft")

        tables, doc = ingest_pdf(pdf_filename)

        dfs = []
        captions = []
        for table in tables:
            try:
                ft = FORMATTER.extract(table)
                dfs.append(ft.df())
            except Exception as e:
                warnings.warn(f"Failed to extract table: {e}")
                continue

            try:
                tup = table.captions()
                if tup[0].lower().startswith("table"):
                    caption = tup[0]
                elif tup[1].lower().startswith("table"):
                    caption = tup[1]
                else:
                    caption = " ".join(table.captions())
                captions.append(caption)
            except:
                captions.append(None)

        obj.dfs = dfs
        obj.captions = captions
        doc.close()
        return obj


    def get_tables(self, as_markdown=False):
        return self.dfs

    def get_captions(self):
        return self.captions

    def to_markdown(self):
        """
        Convert results to LLM-friendly markdown.
        Returns a list of strings representing the tables.
        """
        results = []
        for i, df in enumerate(self.dfs):
            caption = self.captions[i]
            md = U.df_tomd(df, caption=caption)
            results.append(md)
        return results

    def get_markdown_tables(self):
        """
        Returns  list of all tables as Markdown, including captions
        """
        md_tables = []
        for i, df in enumerate(self.dfs):
            caption = self.captions[i]
            md_table = U.df_to_md(df, caption)        
            md_tables.append(md_table)
        return md_tables



