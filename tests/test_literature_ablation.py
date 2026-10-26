"""CPU checks for paired experiment design, metrics, and crash-safe recovery."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("literature_ablation", ROOT / "tools/train_literature_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from fwt_v2.engine import model_state_digest
from fwt_v2.training_state import capture_rng, optimizer_state_integrity, restore_rng


def step(model, optimizer):
    x, y = torch.randn(4, 3), torch.randn(4, 1)
    optimizer.zero_grad(set_to_none=True)
    (model(x) - y).square().mean().backward()
    optimizer.step()


def state_fixture(seed=4, arm="old_bce", epochs=1, selected_epoch=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    history, best = [], None
    for epoch in range(epochs):
        for _ in range(51):
            step(model, optimizer)
        ap = .6 if epoch == selected_epoch else .5
        history.append(dict(epoch=epoch, validation=dict(pooled_ap=ap)))
        state = dict(schema=runner.SCHEMA, binding=dict(seed=seed, arm=arm), epoch=epoch,
            completed_epochs=epoch + 1, updates=(epoch + 1) * 51, validation_ap=ap,
            model_state_dict=runner.tensor_cpu_tree(model.state_dict()),
            model_state_sha256=model_state_digest(model),
            optimizer_state_dict=runner.tensor_cpu_tree(optimizer.state_dict()),
            rng_state=capture_rng("cpu"), history=deepcopy(history), selected_epoch=selected_epoch)
        state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
        if epoch == selected_epoch:
            best = runner.serialize_state(state)
    return model, optimizer, state, best


def record(event, probability, target, valid=None):
    p, y = np.asarray(probability, np.float64), np.asarray(target, np.float64)
    valid = np.ones_like(y, bool) if valid is None else np.asarray(valid, bool)
    return dict(sample_id=event, event_id=event, probability=p, target=y, valid_mask=valid,
                domain_mask=np.ones_like(valid), calibration_logit=np.log(p / (1 - p)))


class DesignTests(unittest.TestCase):
    def plan_fixture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        tr, val = [f"tr-{i}" for i in range(811)], [f"v-{i}" for i in range(568)]
        manifest = root / "data" / "manifest.json"
        write(manifest, {"samples": [dict(sample_id="tb", role="backbone_train")]
            + [dict(sample_id=s, role="residual_train") for s in tr]
            + [dict(sample_id=s, role="validation") for s in val]})
        normalizer, backbone, bundle = manifest.parent / "normalizer.json", root / "h1.pth", manifest.parent / "bundle.json"
        write(normalizer, {})
        backbone.write_bytes(b"fixed h1")
        write(bundle, dict(data_scope="development_only", heldout_arrays_included=False))
        reference_path = root / "reference" / "plan.json"
        write(reference_path.parent / "input-binding.json", {})
        sources = {"fwt_v2/models.py": "old-models", "fwt_v2/rcda.py": "old-rcda",
                   "fwt_v2/data.py": "data", "tools/train_refiner_multiseed.py": "old-runner"}
        reference = dict(schema="fwt_logit_refiner_multiseed_plan_v1", expected_h1_validation_ap=runner.EXPECTED_AP,
            source_files=sources, input_paths=dict(manifest=str(manifest), normalizer=str(normalizer), backbone=str(backbone)),
            input_files_sha256={str(p): runner.sha(p) for p in (manifest, normalizer, backbone, bundle)},
            bound_data_metadata_paths={}, training_sample_ids=tr, validation_sample_ids=val,
            diffusion_config={}, data_config={}, model_config={}, manifest_sha256="manifest", normalizer_sha256="normalizer")
        write(reference_path, reference)
        write(reference_path.parent / "receipt.json", dict(schema="fwt_logit_refiner_multiseed_completed_v1",
            status="completed", seeds={str(s): dict(status="completed", completed_epochs=10, total_updates=510)
                for s in (1, 2, 3)}, input_binding_sha256=runner.sha(reference_path.parent / "input-binding.json")))
        current = sources | {"fwt_v2/models.py": "new-models", "fwt_v2/rcda.py": "new-rcda",
            "fwt_v2/literature_refiner.py": "compact", "tools/train_literature_ablation.py": "new-runner"}
        args = SimpleNamespace(reference_plan=reference_path, backbone=backbone, manifest=manifest,
            normalizer=normalizer, output=root / "pilot", stage="pilot", candidate=None,
            execute=False, resume_run=None, selection_decision=None)
        return args, reference, current

    def test_plan_binds_original_sources_inputs_and_forbids_heldout_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            args, reference, sources = self.plan_fixture(Path(folder))
            with patch.object(runner, "source_files", return_value=sources):
                plan = runner.make_plan(args)
                self.assertEqual(plan["seeds"], [4])
                self.assertEqual(plan["arms"], list(runner.ARMS))
                self.assertEqual(plan["source_files"], sources)
                self.assertIn(str(args.reference_plan), plan["input_files_sha256"])
            with patch.object(runner, "source_files", return_value=sources | {"fwt_v2/data.py": "changed"}):
                with self.assertRaisesRegex(ValueError, "unchanged reference source differs"):
                    runner.make_plan(args)
            # Even if an altered reference honestly binds the altered manifest,
            # data roles still independently forbid final/calibration arrays.
            manifest = json.loads(args.manifest.read_text())
            manifest["samples"].append(dict(sample_id="heldout", role="final"))
            args.manifest.write_text(json.dumps(manifest))
            reference["input_files_sha256"][str(args.manifest)] = runner.sha(args.manifest)
            args.reference_plan.write_text(json.dumps(reference))
            with patch.object(runner, "source_files", return_value=sources):
                with self.assertRaisesRegex(ValueError, "held-out roles forbidden"):
                    runner.make_plan(args)

    def test_confirm_requires_frozen_candidate_and_same_pilot_source_input_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, _, sources = self.plan_fixture(root)
            with patch.object(runner, "source_files", return_value=sources):
                pilot = runner.make_plan(args)
                args.output.mkdir()
                pilot_path = args.output / "plan.json"
                pilot_path.write_text(json.dumps(pilot))
                pilot_receipt = args.output / "receipt.json"
                pilot_receipt.write_text(json.dumps(dict(schema="fwt_literature_ablation_completed_v1", status="completed",
                    stage="pilot", plan_sha256=runner.sha(pilot_path), seeds={"4": {a: dict(status="completed",
                        total_updates=510, completed_epochs=10) for a in runner.ARMS}})))
                decision = dict(schema="fwt_literature_selection_decision_v1", candidate="compact_bce", selection_frozen=True,
                    pilot_plan=str(pilot_path), pilot_plan_sha256=runner.sha(pilot_path),
                    pilot_receipt=str(pilot_receipt), pilot_receipt_sha256=runner.sha(pilot_receipt))
                decision_path = root / "decision.json"
                decision_path.write_text(json.dumps(decision))
                args.stage, args.candidate, args.output = "confirm", "compact_bce", root / "confirm"
                with self.assertRaisesRegex(ValueError, "selection-decision"):
                    runner.make_plan(args)
                args.selection_decision = decision_path
                confirm = runner.make_plan(args)
                self.assertEqual(confirm["seeds"], [5, 6, 7])
                self.assertEqual(confirm["arms"], ["old_bce", "compact_bce"])
                self.assertIn(str(decision_path), confirm["input_files_sha256"])
                self.assertIn(str(pilot_receipt), confirm["input_files_sha256"])
                args.candidate = "old_tversky"
                with self.assertRaisesRegex(ValueError, "candidate/freeze differs"):
                    runner.make_plan(args)
                args.candidate = "compact_bce"
                changed = sources | {"tools/train_literature_ablation.py": "changed-new-runner"}
                with patch.object(runner, "source_files", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "identical source/reference binding"):
                        runner.make_plan(args)

    def test_frozen_stage_and_independent_complete_orders(self):
        self.assertEqual(runner.stage_design("pilot"), ((4,), runner.ARMS))
        self.assertEqual(runner.stage_design("confirm", "compact_bce"),
                         ((5, 6, 7), ("old_bce", "compact_bce")))
        for stage, candidate in (("pilot", "old_tversky"), ("confirm", None), ("confirm", "old_bce")):
            with self.assertRaises(ValueError):
                runner.stage_design(stage, candidate)
        ids = [str(i) for i in range(811)]
        before = torch.get_rng_state().clone()
        orders = [runner.seeded_order(seed, ids) for seed in (4, 5, 6, 7)]
        self.assertEqual(len({runner.canonical_sha(x) for x in orders}), 4)
        self.assertEqual(orders[0], runner.seeded_order(4, ids))
        for order in orders:
            self.assertEqual(len(order), 10)
            for epoch in order:
                self.assertEqual([len(b) for b in epoch], [16] * 50 + [11])
                self.assertEqual(sorted(x for b in epoch for x in b), sorted(ids))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for invalid in (1, True, "4"):
            with self.assertRaises(ValueError):
                runner.seeded_order(invalid, ids)

    def test_same_architecture_loss_pair_has_identical_initial_weights_and_rng(self):
        from fwt_v2.diffusion import DiffusionConfig
        config = DiffusionConfig(backbone_feature_channels=(4, 4, 4, 4), relational_channels=4,
            hidden_channels=(4, 4, 4, 4), condition_channels=2,
            prediction_type="clean_probability", output_initialization="zero")
        fixed = torch.nn.Linear(3, 2)
        fixed_state, digest = deepcopy(fixed.state_dict()), model_state_digest(fixed)
        def load(*_):
            model = torch.nn.Linear(3, 2)
            model.load_state_dict(fixed_state)
            return model, {}
        hashes = {}
        with patch("fwt_v2.engine.load_det", side_effect=load):
            for seed in (4, 5):
                for architecture in ("old", "compact"):
                    a, a_sha, a_rng = runner.initialize_head(seed, architecture + "_bce", "unused", "cpu", config, digest)
                    b, b_sha, b_rng = runner.initialize_head(seed, architecture + "_tversky", "unused", "cpu", config, digest)
                    self.assertEqual(a_sha, b_sha)
                    self.assertTrue(torch.equal(a_rng["torch_cpu"], b_rng["torch_cpu"]))
                    self.assertEqual(a.configuration(), b.configuration())
                    hashes[seed, architecture] = a_sha
            with self.assertRaisesRegex(ValueError, "different fixed H1"):
                runner.initialize_head(4, "old_bce", "unused", "cpu", config, "wrong")
        for architecture in ("old", "compact"):
            self.assertNotEqual(hashes[4, architecture], hashes[5, architecture])


class MetricTests(unittest.TestCase):
    def test_pooled_ap_sorts_once_and_proper_scores_are_event_equal(self):
        rows = [record("a", [.8, .2], [1, 0]), record("b", [.6, .4, .3, .9], [1, 0, 0, 0], [1, 1, 1, 0])]
        with patch("sklearn.metrics.average_precision_score", wraps=average_precision_score) as ap:
            result = runner.epoch_metrics(rows)
        self.assertEqual(ap.call_count, 1)
        self.assertEqual(result["pooled_ap"], average_precision_score([1, 0, 1, 0, 0], [.8, .2, .6, .4, .3]))
        event_nll = [-np.log(.8), (-np.log(.6) - np.log(.6) - np.log(.7)) / 3]
        self.assertAlmostEqual(result["nll"], np.mean(event_nll))
        self.assertAlmostEqual(result["brier"], np.mean([.04, (.16 + .16 + .09) / 3]))
        altered = deepcopy(rows)
        altered[1]["probability"][-1] = .001
        altered[1]["calibration_logit"][-1] = -1234
        self.assertEqual(result, runner.epoch_metrics(altered))
        self.assertEqual(result["n_valid_pixel_days"], 5)

    def test_selected_threshold_uses_greater_equal_and_highest_f1_tie(self):
        # At .9: TP=1, FP=0, FN=1 -> F1=2/3; at .2: TP=2, FP=2 -> F1=2/3.
        rows = [record("a", [.9, .8, .7, .2], [1, 0, 0, 1])]
        result = runner.selected_operating_points(rows)
        self.assertEqual(result["validation_max_f1"]["threshold"], .9)
        self.assertEqual(result["validation_recall_50"]["threshold"], .9)
        self.assertEqual(result["validation_recall_70"]["threshold"], .2)
        self.assertEqual(result["validation_recall_80"]["recall"], 1.)
        rows.append(record("negative", [.99, .8], [0, 0]))
        result = runner.selected_positive_event_ap(rows)
        self.assertEqual(result["n_positive_events"], 1)
        self.assertEqual(result["positive_event_ap"], average_precision_score([1, 0, 0, 1], [.9, .8, .7, .2]))

    def test_cuda_events_are_only_read_after_explicit_epoch_collection(self):
        instances = []
        class Event:
            def __init__(self, **_):
                self.read = False
                instances.append(self)
            def record(self):
                pass
            def elapsed_time(self, end):
                self.read = True
                return 20.
        events = defaultdict(list)
        with patch("torch.cuda.Event", Event), patch("torch.cuda.synchronize") as synchronize:
            for _ in range(3):
                with runner.cuda_interval(events, "head/forward"):
                    pass
            self.assertFalse(any(x.read for x in instances))
            synchronize.assert_not_called()
            self.assertEqual(runner.cuda_elapsed_seconds(events), {"head/forward": .06})


class RecoveryTests(unittest.TestCase):
    def test_full_state_recovery_matches_next_optimizer_and_all_rngs(self):
        model, optimizer, state, _ = state_fixture()
        step(model, optimizer)
        expected = (random.random(), np.random.random(), torch.rand(2))
        restored = torch.nn.Linear(3, 1)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=.01)
        runner.restore_state(state, restored, restored_optimizer, state["binding"], "cpu")
        step(restored, restored_optimizer)
        actual = (random.random(), np.random.random(), torch.rand(2))
        self.assertEqual(model_state_digest(restored), model_state_digest(model))
        self.assertEqual(optimizer_state_integrity(optimizer.state_dict()), optimizer_state_integrity(restored_optimizer.state_dict()))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_binding_optimizer_and_progress_tampering_are_rejected(self):
        model, optimizer, state, _ = state_fixture()
        with self.assertRaisesRegex(ValueError, "schema/binding"):
            runner.restore_state(state, model, optimizer, dict(state["binding"], arm="compact_bce"), "cpu")
        bad = deepcopy(state)
        bad["updates"] += 1
        with self.assertRaisesRegex(ValueError, "progress/history"):
            runner.restore_state(bad, model, optimizer, state["binding"], "cpu")
        bad = deepcopy(state)
        next(iter(bad["optimizer_state_dict"]["state"].values()))["exp_avg"].add_(1)
        with self.assertRaisesRegex(ValueError, "AdamW integrity"):
            runner.restore_state(bad, model, optimizer, state["binding"], "cpu")

    def test_atomic_transaction_survives_incomplete_best_last_mirrors(self):
        _, _, state, best = state_fixture()
        transaction = runner.build_transaction(4, {"old_bce": state}, {"old_bce": best}, state["rng_state"])
        with tempfile.TemporaryDirectory() as folder, patch("fwt_v2.artifacts.WORKSPACE", Path(folder)):
            seed_dir = Path(folder) / "seed"
            arm = seed_dir / "old_bce"
            arm.mkdir(parents=True)
            (arm / "best.pth").write_bytes(b"old best")
            (arm / "last.pth").write_bytes(b"old last")
            real_atomic_bytes = runner.atomic_bytes
            calls = 0
            def fail_second(path, value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated disk write interruption")
                return real_atomic_bytes(path, value)
            with patch.object(runner, "atomic_bytes", side_effect=fail_second):
                with self.assertRaises(OSError):
                    runner.publish_transaction(seed_dir, transaction)
            authoritative = torch.load(seed_dir / "transaction.pth", weights_only=True)
            runner.decode_transaction(authoritative, 4, ["old_bce"])
            self.assertEqual((arm / "last.pth").read_bytes(), b"old last")
            recovered = Path(folder) / "recovered"
            (recovered / "old_bce").mkdir(parents=True)
            runner.publish_transaction(recovered, authoritative)
            self.assertEqual(runner.sha(recovered / "old_bce" / "best.pth"), authoritative["arms"]["old_bce"]["best_sha256"])
            self.assertEqual(runner.sha(recovered / "old_bce" / "last.pth"), authoritative["arms"]["old_bce"]["last_sha256"])

    def test_complete_seed_then_partial_seed_resume_preserves_remaining_budget(self):
        with tempfile.TemporaryDirectory() as folder, patch("fwt_v2.artifacts.WORKSPACE", Path(folder)):
            root = Path(folder)
            remaining = []
            for seed, epochs in ((5, 10), (6, 3)):
                _, _, state, best = state_fixture(seed=seed, epochs=epochs)
                old_dir, new_dir = root / f"old-{seed}", root / f"new-{seed}"
                (old_dir / "old_bce").mkdir(parents=True)
                (new_dir / "old_bce").mkdir(parents=True)
                transaction = runner.build_transaction(seed, {"old_bce": state}, {"old_bce": best}, state["rng_state"])
                runner.publish_transaction(old_dir, transaction)
                model = torch.nn.Linear(3, 1)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
                start, history, best_bytes, aps, selected = runner.restore_seed_transaction(
                    old_dir / "transaction.pth", new_dir, seed, {"old_bce": model}, {"old_bce": optimizer},
                    {"old_bce": state["binding"]}, "cpu")
                remaining.append(list(range(start, runner.EPOCHS)))
                self.assertEqual(len(history["old_bce"]), epochs)
                self.assertEqual(model_state_digest(model), state["model_state_sha256"])
                self.assertEqual(best_bytes["old_bce"], best)
                self.assertEqual(aps["old_bce"], .6)
                self.assertEqual(selected["old_bce"], 0)
                self.assertEqual(runner.sha(new_dir / "old_bce" / "last.pth"), runner.sha(old_dir / "old_bce" / "last.pth"))
            self.assertEqual(remaining, [[], list(range(3, 10))])

    def test_transaction_rejects_wrong_best_tampered_bytes_and_mixed_epochs(self):
        _, _, state, best = state_fixture(epochs=2)
        transaction = runner.build_transaction(4, {"old_bce": state}, {"old_bce": best}, state["rng_state"])
        bad = deepcopy(transaction)
        bad["arms"]["old_bce"]["best_data"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "bytes failed integrity"):
            runner.decode_transaction(bad, 4, ["old_bce"])
        bad = deepcopy(transaction)
        last = runner.deserialize_state(runner.transaction_checkpoint_bytes(bad["arms"]["old_bce"], "last"))
        last["selected_epoch"] = 1
        value = runner.serialize_state(last)
        bad["arms"]["old_bce"].update(last_data=torch.frombuffer(bytearray(value), dtype=torch.uint8),
                                           last_sha256=hashlib.sha256(value).hexdigest())
        with self.assertRaisesRegex(ValueError, "selection or progress"):
            runner.decode_transaction(bad, 4, ["old_bce"])
        with self.assertRaisesRegex(ValueError, "same completed epoch"):
            runner.build_transaction(4, {"a": state, "b": dict(state, completed_epochs=1)}, {"a": best, "b": best}, state["rng_state"])

    def test_existing_output_is_never_modified_on_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.json"
            path.write_text("original")
            args = SimpleNamespace(output=Path(folder), execute=True)
            with patch.object(runner, "parse_args", return_value=args), patch.object(runner, "make_plan", return_value={}), \
                    patch.object(runner, "execute", side_effect=FileExistsError("existing")):
                with self.assertRaises(FileExistsError):
                    runner.main([])
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
