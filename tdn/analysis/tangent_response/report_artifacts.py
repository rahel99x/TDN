"""Lossless research pages below Raven's shared default inspection budget."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from tdn.analysis.response import diagnostic_metrics as shared
from tdn.runtime.metadata import write_json

PAGE_BYTES = 7 << 20
CHUNK_BYTES = 1 << 20


def _json(value):
    return (json.dumps(value,separators=(",",":"),allow_nan=False)+"\n").encode()


def _csv(rows, columns):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in rows:
        values={}
        for key in columns:
            value=row.get(key)
            if isinstance(value,(dict,list)):value=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
            if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")):value="'"+value
            values[key]=value
        writer.writerow(values)
    return stream.getvalue().encode()


class Pages:
    def __init__(self, output, campaign, config, identity):
        self.output=Path(output);self.campaign=campaign;self.config=config;self.identity=identity
        self.groups=[];self.current=None;self.projects=set()

    def add(self, name, raw, format):
        shared.relative_path(name)
        if len(raw)>PAGE_BYTES:raise ValueError("Research artifact must be split below its default reader budget")
        if self.current is None or self.current["bytes"]+len(raw)>PAGE_BYTES or len(self.current["files"])>=240:
            index=len(self.groups);project="artifact-pages/page-"+f"{index//255:03d}"
            run_id=shared.stable_id(self.campaign+"-data-"+f"{index:04d}")
            attempt=project+"/runs/"+run_id
            shared._project_files(self.output/project,self.config);self.projects.add(project)
            self.current=dict(path=attempt,run_id=run_id,bytes=0,files=[]);self.groups.append(self.current)
        group=self.current;pointer=group["path"]+"/outputs/"+name
        file=self.output/pointer;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw)
        record=dict(path=pointer,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        group["files"].append(dict(path="outputs/"+name,required=True,min_bytes=min(1,len(raw)),
            sha256=record["sha256"],format=format));group["bytes"]+=len(raw)
        return record

    def finish(self):
        for group in self.groups:
            attempt=self.output/group["path"]
            (attempt/"logs").mkdir();(attempt/"reports").mkdir()
            metadata=dict(self.identity,artifact_scope="Lossless pages of already-produced research; no scheduler allocation or new scientific result",
                native_gpu_verification="NOT_INFERRED_FROM_REPORTING")
            name="TDN/tangent-response/research-data"
            write_json(attempt/"run.json",dict(schema="raven.run/v1",project_id=self.config["project_id"],run_id=group["run_id"],
                experiment_id="tangent-response/artifact-pages",attempt=1,name=name,state="COMPLETED",
                paths=dict(metrics="metrics.jsonl",summary="summary.json",log_index="logs.json",outputs="outputs",logs="logs",contract="outputs/contract.json"),
                metadata=metadata))
            write_json(attempt/"summary.json",dict(schema="raven.summary/v1",id=group["run_id"],name=name,state="COMPLETED",
                project_id=self.config["project_id"],experiment_id="tangent-response/artifact-pages",attempt=1,exit_code=0,
                results=dict(artifact_files=len(group["files"]),artifact_bytes=group["bytes"]),metadata=metadata))
            write_json(attempt/"logs.json",dict(schema="raven.logs/v1",run_id=group["run_id"],
                logs=[dict(id="research-data",path="logs/research-data.log",label="Research data page",group="Research")]))
            (attempt/"logs/research-data.log").write_text("Complete research artifacts; source scientific decisions remain unchanged.\n")
            (attempt/"metrics.jsonl").write_bytes(_json(shared.metric_record(dict(artifact_files=len(group["files"]),artifact_bytes=group["bytes"]),phase="research_data")))
            write_json(attempt/"outputs/contract.json",dict(version=1,outputs=group["files"]))
        return [dict(path=project,attempts=sum(group["path"].startswith(project+"/") for group in self.groups),
            path_base="Research overview outputs; register each page as an independent Raven project") for project in sorted(self.projects)]


def _mapped(value, mapping):
    if isinstance(value,str):return mapping.get(value,value)
    if isinstance(value,list):return [_mapped(child,mapping) for child in value]
    if isinstance(value,dict):return {key:_mapped(child,mapping) for key,child in value.items()}
    return value


def _row_chunks(rows):
    chunk=[];size=0
    for row in rows:
        length=len(_json(row))
        if length>PAGE_BYTES//2:raise ValueError("A compiled research row exceeds the bounded artifact budget; its raw source remains retained")
        if chunk and size+length>CHUNK_BYTES:yield chunk;chunk=[];size=0
        chunk.append(row);size+=length
    if chunk or not rows:yield chunk


def _byte_chunks(raw):
    start=0
    while start<len(raw):
        end=min(len(raw),start+CHUNK_BYTES)
        # Preserve bytes while keeping each UTF-8 text fragment independently
        # readable. Native scientific JSON normally uses ASCII escapes.
        while end<len(raw) and raw[end]&0xC0==0x80:end-=1
        yield start,raw[start:end];start=end


def publish(output, data, copies, config, campaign):
    """Preserve every row/byte; only the standard overview becomes a catalog."""
    output=Path(output)
    pages=Pages(output,campaign,config,dict(source_tree_sha256=data["source_tree_sha256"],protocol_sha256=data["protocol_sha256"]))
    mapping={};raw_segments={}
    for pointer,raw in copies.items():
        if pointer.startswith("lineage/"):
            file=output/pointer;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw);continue
        suffix=Path(pointer).suffix;format="json" if suffix==".json" else "text"
        if len(raw)<=PAGE_BYTES:
            record=pages.add(pointer,raw,format);mapping[pointer]=record["path"]
        else:
            # The exact original JSON is also retained for ordinary tools. The
            # default Raven browser sees its ordered, byte-exact text segments.
            canonical="raw-sources/"+Path(pointer).name
            file=output/canonical;file.parent.mkdir(exist_ok=True);file.write_bytes(raw);mapping[pointer]=canonical
            raw_segments[canonical]=[dict(pages.add("source-segments/"+Path(pointer).name+".part"+f"{i:04d}"+".txt",
                part,"text"),byte_offset=start) for i,(start,part) in enumerate(_byte_chunks(raw))]
    data=_mapped(data,mapping)
    tables={}
    for name,rows in data["tables"].items():
        columns=sorted({key for row in rows for key in row}) or ["status"]
        chunks=[];start=0
        for index,part in enumerate(_row_chunks(rows)):
            base="tables/"+shared.stable_id(name)+"-"+f"{index:04d}"
            payload=dict(schema="tdn.tangent-response-table-chunk/v1",table=name,start_row=start,total_rows=len(rows),rows=part)
            json_record=pages.add(base+".json",_json(payload),"json")
            csv_record=pages.add(base+".csv",_csv(part,columns),"csv")
            chunks.append(dict(start_row=start,rows=len(part),json=json_record,csv=csv_record));start+=len(part)
        tables[name]=dict(rows=len(rows),columns=columns,chunks=chunks)
    sections={}
    for name in ("sources","omissions","decisions","lineage_projects"):
        rows=data[name];chunks=[];start=0
        for index,part in enumerate(_row_chunks(rows)):
            payload=dict(schema="tdn.tangent-response-section-chunk/v1",section=name,start_record=start,total_records=len(rows),records=part)
            record=pages.add("catalogs/"+name+"-"+f"{index:04d}"+".json",_json(payload),"json")
            chunks.append(dict(start_record=start,records=len(part),**record));start+=len(part)
        sections[name]=dict(records=len(rows),chunks=chunks)
    artifact_projects=pages.finish()
    index={key:value for key,value in data.items() if key not in ("tables","sources","omissions","decisions","lineage_projects")}
    index.update(schema="tdn.tangent-response-research-index/v1",research_schema=data["schema"],tables=tables,sections=sections,
        decisions=[{key:decision[key] for key in ("gate","decision","eligible","stage","source_path","source_sha256") if key in decision} for decision in data["decisions"]],
        artifact_projects=artifact_projects,raw_source_segments=raw_segments,
        path_base="This research overview outputs directory; all paths are confined relative paths",
        completeness="Every compiled row and canonical source byte is retained; this JSON is a catalog, not the full dataset")
    raw=_json(index)
    if len(raw)>CHUNK_BYTES:raise ValueError("Research overview catalog exceeds its 1 MiB default-reader bound")
    (output/"tangent_research.json").write_bytes(raw)
    return index,data


def read_rows(output,index,table):
    rows=[]
    for chunk in index["tables"][table]["chunks"]:
        payload,_,_=shared.read_metadata(Path(output)/chunk["json"]["path"],root=output,expected=chunk["json"]["sha256"])
        if payload["start_row"]!=len(rows):raise ValueError("Research table pages have a row gap")
        rows.extend(payload["rows"])
    if len(rows)!=index["tables"][table]["rows"]:raise ValueError("Research table pages changed their complete row denominator")
    return rows


def read_section(output,index,section):
    rows=[]
    for chunk in index["sections"][section]["chunks"]:
        payload,_,_=shared.read_metadata(Path(output)/chunk["path"],root=output,expected=chunk["sha256"])
        if payload["start_record"]!=len(rows):raise ValueError("Research section pages have a record gap")
        rows.extend(payload["records"])
    if len(rows)!=index["sections"][section]["records"]:raise ValueError("Research section pages changed their complete denominator")
    return rows
