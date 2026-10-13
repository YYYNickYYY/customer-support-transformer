"""Absent optional bundles are distinct from damaged local evidence."""
from conftest import optional_bundle_reason


def test_source_checkout_skips_only_declared_local_cases(tmp_path):
    assert optional_bundle_reason(tmp_path, 'test_training_eighth.py',
                                  'test_actual_preflight_no_model_load_training_or_forward')
    assert optional_bundle_reason(tmp_path, 'test_training_eighth.py', 'test_epochs_only') is None
    assert optional_bundle_reason(tmp_path, 'test_model.py', 'test_forward') is None
    assert optional_bundle_reason(tmp_path, 'test_workflow_prefix_control.py', 'example', ('setup',))
    assert optional_bundle_reason(tmp_path, 'test_workflow_prefix_control.py', 'example') is None


def test_partial_or_corrupt_bundle_is_not_skipped(tmp_path):
    (tmp_path / 'outputs').mkdir()
    # Even an empty local bundle is not silently accepted as source-only.
    assert optional_bundle_reason(tmp_path, 'test_training_eighth.py',
                                  'test_actual_preflight_no_model_load_training_or_forward') is None


def test_workbench_is_optional_but_existing_workbench_is_checked(tmp_path):
    assert optional_bundle_reason(tmp_path, 'test_training_seventh.py', 'test_workspace_entry')
    (tmp_path / '第七次容量实验.code-workspace').write_text('invalid JSON', encoding='utf-8')
    assert optional_bundle_reason(tmp_path, 'test_training_seventh.py', 'test_workspace_entry') is None
