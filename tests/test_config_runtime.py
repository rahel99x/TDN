import copy
from pathlib import Path
import pytest
from tdn.config import load_config, validate_config, config_hash
from tdn.runtime.storage import contained_path
ROOT=Path(__file__).resolve().parents[1]

def test_strict_complete_config():
    c=load_config(ROOT/"configs/smoke.yaml")
    assert config_hash(c)==config_hash(copy.deepcopy(c))
    c["model"]["queried_h"]=1
    with pytest.raises(ValueError):validate_config(c)

@pytest.mark.parametrize("section,key,value",[("teacher","error_fraction",0.1),("training","max_steps",5001),("precision","teacher","float32"),("model","anchored",True),("runtime","soft_vram_gib",40),("data","workers",2)])
def test_reject_protocol_violations(section,key,value):
    c=load_config(ROOT/"configs/smoke.yaml");c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)

def test_confirmation_is_not_a_pilot_claim():
    c=load_config(ROOT/"configs/smoke.yaml");c["purpose"]="confirmatory"
    with pytest.raises(ValueError,match="Confirmation is blocked"):validate_config(c)

def test_symlink_escape(tmp_path):
    root=tmp_path/"root";root.mkdir();outside=tmp_path/"outside";outside.mkdir()
    try:
        (root/"cache").symlink_to(outside,target_is_directory=True)
    except OSError:
        pytest.skip("Windows account lacks symlink permission")
    with pytest.raises(ValueError,match="escapes"):contained_path(root/"cache"/"state",root)

def test_training_respects_stricter_initial_free_memory_fraction(monkeypatch):
    import torch
    from tdn.train.loop import _MemoryBudget
    c=load_config(ROOT/"configs/smoke.yaml")
    c["runtime"]["soft_vram_fraction"]=.5
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda _device:(2*2**30,40*2**30))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda _device:None)
    memory=_MemoryBudget(c,"cuda")
    assert memory.soft==2**30
    assert memory.initial_free==2*2**30
