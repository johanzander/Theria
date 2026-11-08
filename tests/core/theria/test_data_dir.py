import json

from core.theria.data_dir import data_dir, read_data_files, write_data_files


def test_the_add_on_keeps_its_files_in_data():
    assert data_dir({}) == "/data"


def test_a_local_run_names_its_own_directory():
    assert data_dir({"THERIA_DATA_DIR": "/tmp/x"}) == "/tmp/x"


def test_the_files_a_plan_depends_on_are_read(tmp_path):
    (tmp_path / "plan_overrides.json").write_text(json.dumps({"target_c": 18.5}))
    (tmp_path / "plan_state.json").write_text(json.dumps({"mode": "active"}))
    files = read_data_files(str(tmp_path))
    assert files == {
        "plan_overrides.json": {"target_c": 18.5},
        "plan_state.json": {"mode": "active"},
    }


def test_the_influxdb_password_is_not_handed_out(tmp_path):
    options = {"plan": {"mode": "shadow"}, "influxdb": {"password": "secret"}}
    (tmp_path / "options.json").write_text(json.dumps(options))
    assert read_data_files(str(tmp_path)) == {
        "options.json": {"plan": {"mode": "shadow"}}
    }


def test_files_that_are_not_settings_or_state_are_not_read(tmp_path):
    (tmp_path / "plan_record.jsonl").write_text("{}\n")
    assert read_data_files(str(tmp_path)) == {}


def test_written_files_match_the_add_on(tmp_path):
    write_data_files(str(tmp_path), {"plan_state.json": {"mode": "active"}})
    assert json.loads((tmp_path / "plan_state.json").read_text()) == {"mode": "active"}


def test_a_local_file_the_add_on_does_not_have_is_removed(tmp_path):
    (tmp_path / "plan_overrides.json").write_text(json.dumps({"target_c": 21.0}))
    write_data_files(str(tmp_path), {"plan_state.json": {"mode": "shadow"}})
    assert not (tmp_path / "plan_overrides.json").exists()


def test_local_history_is_left_alone_when_files_are_written(tmp_path):
    (tmp_path / "plan_record.jsonl").write_text("{}\n")
    write_data_files(str(tmp_path), {})
    assert (tmp_path / "plan_record.jsonl").exists()
