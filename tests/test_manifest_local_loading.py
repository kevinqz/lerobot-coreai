# test_manifest_local_loading.py — local vs HF manifest resolution (v1.3.3).

import json

import pytest

from lerobot_coreai.errors import ManifestError
from lerobot_coreai.manifest import load_manifest, resolve_manifest


def _write_manifest(dirpath, valid_manifest_dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "lerobot-coreai.json").write_text(json.dumps(valid_manifest_dict))
    return dirpath


def test_local_directory_no_network(tmp_path, valid_manifest_dict):
    d = _write_manifest(tmp_path / "artifact", valid_manifest_dict)
    manifest, kind, sha, net = resolve_manifest(str(d))
    assert kind == "local_directory"
    assert net is False
    assert sha.startswith("sha256:")
    assert manifest.policy_repo_id == valid_manifest_dict["policy"]["repo_id"]


def test_local_file_no_network(tmp_path, valid_manifest_dict):
    _write_manifest(tmp_path / "art", valid_manifest_dict)
    fpath = tmp_path / "art" / "lerobot-coreai.json"
    manifest, kind, sha, net = resolve_manifest(str(fpath))
    assert kind == "local_file" and net is False


def test_nonexistent_local_path_fails_no_hf_fallback(tmp_path):
    with pytest.raises(ManifestError):
        resolve_manifest(str(tmp_path / "does-not-exist"))
    with pytest.raises(ManifestError):
        resolve_manifest("./definitely/not/here")


def test_directory_without_manifest_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifestError):
        resolve_manifest(str(tmp_path / "empty"))


def test_hf_repo_id_is_not_treated_as_local(tmp_path, valid_manifest_dict, monkeypatch):
    # An org/name repo id must route to HF (we stub the HF path to avoid network).
    called = {"hf": False}

    def _fake_get(url, **kw):
        called["hf"] = True
        class _R:
            status_code = 200
            content = json.dumps(valid_manifest_dict).encode()
            def json(self):
                return valid_manifest_dict
        return _R()

    monkeypatch.setattr("lerobot_coreai.manifest.httpx.get", _fake_get)
    manifest, kind, sha, net = resolve_manifest("kevinqz/EVO1-SO100-CoreAI")
    assert kind == "hf_repo" and net is True and called["hf"] is True


def test_load_manifest_wrapper_local(tmp_path, valid_manifest_dict):
    d = _write_manifest(tmp_path / "art", valid_manifest_dict)
    m = load_manifest(str(d))
    assert m.policy_repo_id == valid_manifest_dict["policy"]["repo_id"]
