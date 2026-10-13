"""Publication boundary for legacy tests bound to private/local experiment files.

Do not alter hash-frozen tests to add skip decorators. Only an entirely absent
outputs bundle means a source-only checkout; an existing but incomplete/corrupt
bundle must still fail. Ordinary unit tests are never caught or suppressed.
"""
from pathlib import Path

import pytest


LOCAL_MODULES = {
    'test_decoding_preregistration.py',
    'test_input_sensitivity.py',
    'test_input_sensitivity_results.py',
    'test_sixth_evaluation.py',
}
LOCAL_CASES = {
    'test_condition_use_measurement_v2.py': {'test_provenance_and_fixed_original_questions'},
    'test_decoding_closure.py': {
        'test_full_ratings_and_balanced_arms', 'test_same_text_cannot_receive_different_scores',
        'test_initial_judgments_not_silently_overwritten', 'test_final_veto_and_paired_repetition_counts'},
    'test_fifth_diagnostic_analysis.py': {'test_old_reply_prompt_compatibility_is_in_memory_only'},
    'test_opening_scope_control.py': {'test_prepare_freezes_all_jobs_without_weights_and_is_read_only_on_repeat'},
    'test_workflow_prefix_control.py': {'test_prepare_freeze_repeat_without_inference'},
    'test_target_revision_v2.py': {'test_reviewed_exclusions_and_required_seams', 'test_frozen_v2_replay_and_tamper_detection'},
    'test_target_revision_v3.py': {'test_approved_replacements_preserve_headings_and_conditions', 'test_frozen_v3_replay_tamper_refuse_overwrite'},
    'test_target_revision_v31.py': {'test_final_replay_preserves_fragments_and_resolves_repetition'},
    'test_training_sixth.py': {'test_same_135_and_corrected_baseline', 'test_reject_comparison_drift'},
    'test_training_seventh.py': {'test_real_preflight_is_read_only'},
    'test_training_eighth.py': {'test_actual_preflight_no_model_load_training_or_forward'},
}
LOCAL_SETUP = {
    'test_condition_measurement_v2_runner.py',
    'test_opening_scope_control.py',
    'test_workflow_prefix_control.py',
}
WORKSPACES = {
    ('test_training_seventh.py', 'test_workspace_entry'): '第七次容量实验.code-workspace',
    ('test_training_eighth.py', 'test_workspace_routes_and_no_default'): '第八次轮数实验.code-workspace',
}


def optional_bundle_reason(root, filename, testname, fixtures=()):
    """Use explicit test identities, never exception-catching or hash bypasses."""
    workspace = WORKSPACES.get((filename, testname))
    if workspace and not (root / workspace).exists():
        return f'Personal workbench is intentionally not published: {workspace}'
    needs_outputs = (filename in LOCAL_MODULES
                     or testname in LOCAL_CASES.get(filename, set())
                     or (filename in LOCAL_SETUP and 'setup' in fixtures))
    if needs_outputs and not (root / 'outputs').exists():
        return 'Historical experiment replay requires the unpublished outputs/runs bundle; source unit tests still run'
    return None


def pytest_collection_modifyitems(config, items):
    root = Path(config.rootpath)
    for item in items:
        reason = optional_bundle_reason(root, item.path.name,
                                        getattr(item, 'originalname', None) or item.name,
                                        item.fixturenames)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
