import json

import pandas as pd

from result_tables import TABLES, read_result


def prepare_tables(results):
    tables = {label: read_result(results,key) for key,label in TABLES.items()}
    verification=json.loads((results/"verification.json").read_text(encoding="utf-8"))
    verification["worksheets"]=len(TABLES)+1
    tables["Run checks"]=pd.DataFrame(list(verification.items()),columns=["Check","Value"])
    return tables


def write_workbook(tables, path):
    with pd.ExcelWriter(path,engine="xlsxwriter",engine_kwargs={
            "options":{"strings_to_formulas":False,"strings_to_urls":False}}) as writer:
        header=writer.book.add_format({"bg_color":"#20364C","font_color":"#FFFFFF",
                                       "bold":True,"valign":"vcenter"})
        decimal=writer.book.add_format({"num_format":"0.0000"})
        for name, table in tables.items():
            table.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name]
            ws.freeze_panes(1,0)
            ws.hide_gridlines(2)
            ws.autofilter(0,0,len(table),len(table.columns)-1)
            ws.set_row(0,24)
            for i,col in enumerate(table.columns):
                ws.write(0,i,col,header)
                text_width=int(table[col].astype(str).str.len().quantile(.90)) if len(table) else 0
                fmt=decimal if pd.api.types.is_float_dtype(table[col]) else None
                if fmt:
                    text_width=12
                ws.set_column(i,i,min(73,max(12,len(col)+3,text_width+2)),fmt)
