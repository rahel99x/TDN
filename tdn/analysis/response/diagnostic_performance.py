"""Measured complete-call and same-state decoder costs; no utility imputation."""
from __future__ import annotations

import statistics
import time
import torch

from .references import equation_geometry
from .training import predict, endpoint_row
from .numerics import anchor

SCHEMA='tdn.response-diagnostic-performance/v1'

def benchmark_model(model, samples, normalization, *, repeats=5, warmup=1, budget=None):
    if type(repeats) is not int or not 1 <= repeats <= 1000 or type(warmup) is not int or not 0 <= warmup <= 20:
        raise ValueError('Declare bounded positive repetitions and nonnegative warmup')
    accepted=[s for s in samples if s.get('reference_accepted',s['record'].get('reference',{}).get('accepted',False))]
    if not accepted:
        return dict(schema=SCHEMA,status='NA',reason='No accepted declared validation query',repeats=repeats)
    sample=min(accepted,key=lambda s:(s['record']['parent_id'],s['record'].get('query_id','')))
    record=sample['record'];device=sample['u0'].device
    eq,geom=equation_geometry(record);h=float(record['horizon'])
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def measure(fn):
        if budget is not None:budget.check()
        sync();start=time.perf_counter();value=fn();sync()
        elapsed=time.perf_counter()-start
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite timed prediction')
        return elapsed,value
    was_training=model.training;model.eval()
    rows=[];encoding_seconds=None;decoder_rows=[]
    try:
        with torch.inference_mode():
            base=lambda:anchor(sample['u0'],h,eq,geom,record['target_kind'])
            call=lambda:predict(model,sample)
            for _ in range(warmup):measure(call);measure(base)
            prediction=baseline=None
            for round_index in range(repeats):
                order=('model','anchor') if round_index%2==0 else ('anchor','model')
                for method in order:
                    seconds,value=measure(call if method=='model' else base)
                    rows.append(dict(round=round_index,method=method,seconds=seconds))
                    if method=='model':prediction=value
                    else:baseline=value
            metadata=model.architecture_metadata()
            if metadata.get('reusable_encoding'):
                sync();start=time.perf_counter();encoded=model.encode(sample['u0'],eq,geom,record['target_kind']);sync()
                encoding_seconds=time.perf_counter()-start
                decode=lambda:model.decode(encoded,h)
                decoded=decode()
                if not torch.equal(decoded,prediction):
                    raise ValueError('Cached inference differs from the complete selected model call')
                for round_index in range(repeats):
                    seconds,_=measure(decode);decoder_rows.append(dict(round=round_index,seconds=seconds))
            model_error=endpoint_row(prediction,sample,normalization)
            anchor_error=endpoint_row(baseline,sample,normalization)
    finally:model.train(was_training)
    model_times=[row['seconds'] for row in rows if row['method']=='model']
    anchor_times=[row['seconds'] for row in rows if row['method']=='anchor']
    return dict(schema=SCHEMA,status='MEASURED',device=str(device),query_id=record.get('query_id'),parent_id=record['parent_id'],
        target_kind=record['target_kind'],horizon=h,repeats=repeats,warmup=warmup,timing_rows=rows,
        complete_call_median_seconds=statistics.median(model_times),anchor_median_seconds=statistics.median(anchor_times),
        model_to_anchor_time_ratio=statistics.median(model_times)/statistics.median(anchor_times),
        encoding_seconds=encoding_seconds,cached_decode_rows=decoder_rows,
        cached_decode_median_seconds=statistics.median([r['seconds'] for r in decoder_rows]) if decoder_rows else None,
        model_error=model_error,anchor_error=anchor_error,
        p99_seconds=None,p99_status='UNAVAILABLE_SAMPLE_COUNT_BELOW1000' if repeats<1000 else 'NOT_ESTIMATED_BY_THIS_PANEL',
        input_transfer_seconds=None,fresh_process_startup_seconds=None,
        scope='Resident-input synchronized complete call, paired same-query anchor, and optional same-state cached decode; one prospectively chosen accepted validation query',
        utility_status='NOT_ESTABLISHED_MATCHED_TOLERANCE_FRONTIER_NOT_MEASURED',
        monetary_cost=None,energy_cost=None)
