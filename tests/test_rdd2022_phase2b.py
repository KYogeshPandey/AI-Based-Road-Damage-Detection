"""Tests for RDD2022 Phase 2B canonical construction and grouped splitting."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_damage.dataset.rdd2022_audit import _tree_fingerprint  # noqa: E402
from road_damage.dataset.rdd2022_canonical import (  # noqa: E402
    NON_TARGET_ACTION,
    REJECT_INVALID_STATUS,
    REJECT_TINY_STATUS,
    TARGET_ACTION,
    UNKNOWN_ACTION,
    apply_boundary_guards,
    apply_special_case_policy,
    assign_capture_groups,
    assign_exact_duplicate_groups,
    canonical_action,
    classify_labelled_image,
    generate_near_duplicate_candidates,
    load_phase2b_config,
    parse_voc_annotation,
    plan_grouped_split,
    split_conflict_audit,
    validate_phase2b_config,
    validate_voc_box,
)
from road_damage.dataset.rdd2022_common import RDD2022Error  # noqa: E402
from road_damage.dataset.rdd2022_construct import run_construction  # noqa: E402
from road_damage.video.inspect_video import _load_opencv  # noqa: E402


class Phase2BPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase2b_config(
            PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b.yaml"
        )

    def test_fixed_class_policy_does_not_merge_non_targets(self) -> None:
        self.assertEqual(canonical_action("D00", self.config), TARGET_ACTION)
        self.assertEqual(canonical_action("D01", self.config), NON_TARGET_ACTION)
        self.assertEqual(canonical_action("D11", self.config), NON_TARGET_ACTION)
        self.assertEqual(canonical_action("D0w0", self.config), UNKNOWN_ACTION)

    def test_configuration_rejects_route_independence_claim(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["split"]["description"] = "route-independent split"
        with self.assertRaises(RDD2022Error):
            validate_phase2b_config(changed)

    def test_valid_voc_coordinates_preserve_raw_and_derive_half_open(self) -> None:
        result = validate_voc_box(
            {"xmin": "1", "ymin": "2", "xmax": "10", "ymax": "20"},
            100,
            100,
            self.config["annotation_validation"],
        )
        self.assertEqual(
            result["canonical_xyxy_pixel_zero_based_half_open"], [0, 1, 10, 20]
        )
        self.assertEqual(result["source_width_pixels_inclusive"], 10)
        self.assertNotIn("out_of_bounds", result["validation_flags"])

    def test_invalid_target_box_flags_inverted_and_out_of_bounds(self) -> None:
        inverted = validate_voc_box(
            {"xmin": "20", "ymin": "1", "xmax": "10", "ymax": "5"},
            100,
            100,
            self.config["annotation_validation"],
        )
        outside = validate_voc_box(
            {"xmin": "1", "ymin": "1", "xmax": "101", "ymax": "5"},
            100,
            100,
            self.config["annotation_validation"],
        )
        self.assertIn("inverted_box", inverted["validation_flags"])
        self.assertIn("zero_area_box", inverted["validation_flags"])
        self.assertIn("out_of_bounds", outside["validation_flags"])

    def _parse(self, root: Path, objects: list[tuple[str, tuple[str, str, str, str]]]) -> dict:
        body = "".join(
            f"<object><name>{name}</name><difficult>0</difficult><truncated>0</truncated>"
            f"<bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>"
            f"<xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object>"
            for name, box in objects
        )
        path = root / "sample.xml"
        path.write_text(
            f"<annotation><size><width>100</width><height>100</height><depth>3</depth>"
            f"</size>{body}</annotation>",
            encoding="utf-8",
        )
        return parse_voc_annotation(path, "image", 100, 100, self.config)

    def test_mixed_target_non_target_image_is_retained_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parsed = self._parse(
                Path(temporary),
                [("D00", ("1", "1", "20", "20")), ("D01", ("2", "2", "4", "4"))],
            )
            result = classify_labelled_image(parsed, self.config)
        self.assertEqual(result["image_category"], "mixed_target_and_known_non_target")
        self.assertTrue(result["is_positive"])

    def test_known_non_target_only_and_empty_xml_are_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            known = classify_labelled_image(
                self._parse(root, [("D11", ("1", "1", "4", "4"))]), self.config
            )
            empty = classify_labelled_image(self._parse(root, []), self.config)
        self.assertEqual(known["negative_subtype"], "known_non_target_hard_negative")
        self.assertEqual(empty["negative_subtype"], "empty_xml_background")

    def test_d0w0_and_invalid_target_quarantine_whole_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = classify_labelled_image(
                self._parse(root, [("D0w0", ("1", "1", "4", "4"))]), self.config
            )
            invalid = classify_labelled_image(
                self._parse(root, [("D40", ("1", "1", "101", "4"))]), self.config
            )
        self.assertEqual(unknown["inclusion_status"], "quarantined")
        self.assertIn("unknown_raw_class:D0w0", unknown["inclusion_exclusion_reason"])
        self.assertEqual(invalid["inclusion_status"], "quarantined")
        self.assertTrue(invalid["inclusion_exclusion_reason"][0].startswith("invalid_target_annotation"))


class Phase2B1PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase2b_config(
            PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b_v1_1.yaml"
        )

    def _parse(
        self,
        root: Path,
        filename: str,
        objects: list[tuple[str, tuple[str, str, str, str]]],
    ) -> dict:
        body = "".join(
            f"<object><name>{name}</name><bndbox><xmin>{box[0]}</xmin>"
            f"<ymin>{box[1]}</ymin><xmax>{box[2]}</xmax>"
            f"<ymax>{box[3]}</ymax></bndbox></object>"
            for name, box in objects
        )
        path = root / f"{Path(filename).stem}.xml"
        path.write_text(
            "<annotation><size><width>100</width><height>100</height><depth>3</depth>"
            f"</size>{body}</annotation>",
            encoding="utf-8",
        )
        parsed = parse_voc_annotation(path, "image", 100, 100, self.config)
        apply_special_case_policy(parsed, filename, self.config)
        return parsed

    def test_zero_lower_edge_is_normalized_and_audited(self) -> None:
        result = validate_voc_box(
            {"xmin": "0", "ymin": "1", "xmax": "10", "ymax": "20"},
            100,
            100,
            self.config["annotation_validation"],
        )
        self.assertEqual(result["canonical_xyxy_pixel_zero_based_half_open"], [0, 0, 10, 20])
        self.assertTrue(result["source_zero_edge_coordinate"])
        self.assertIn("source_zero_edge_coordinate", result["validation_flags"])
        self.assertNotIn("out_of_bounds", result["validation_flags"])

    def test_negative_and_upper_bound_coordinates_still_reject(self) -> None:
        negative = validate_voc_box(
            {"xmin": "-1", "ymin": "1", "xmax": "10", "ymax": "20"},
            100,
            100,
            self.config["annotation_validation"],
        )
        upper = validate_voc_box(
            {"xmin": "1", "ymin": "1", "xmax": "101", "ymax": "20"},
            100,
            100,
            self.config["annotation_validation"],
        )
        self.assertIn("negative_coordinate", negative["validation_flags"])
        self.assertIsNone(negative["canonical_xyxy_pixel_zero_based_half_open"])
        self.assertIn("out_of_bounds", upper["validation_flags"])
        self.assertIsNone(upper["canonical_xyxy_pixel_zero_based_half_open"])

    def test_frozen_invalid_object_rejection_retains_image_as_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parsed = self._parse(
                Path(temporary),
                "Japan_011217.jpg",
                [("D50", ("10", "10", "20", "20")), ("D20", ("2", "0", "20", "30"))],
            )
            classification = classify_labelled_image(parsed, self.config)
        self.assertEqual(parsed["objects"][1]["object_status"], REJECT_INVALID_STATUS)
        self.assertEqual(classification["inclusion_status"], "retained")
        self.assertTrue(classification["is_negative"])
        self.assertEqual(classification["non_target_object_count"], 1)

    def test_frozen_tiny_rejection_preserves_other_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parsed = self._parse(
                Path(temporary),
                "Japan_001265.jpg",
                [
                    ("D00", ("2", "2", "20", "20")),
                    ("D20", ("2", "2", "20", "20")),
                    ("D20", ("2", "2", "20", "20")),
                    ("D20", ("50", "50", "50", "51")),
                ],
            )
            classification = classify_labelled_image(parsed, self.config)
        self.assertEqual(parsed["objects"][3]["object_status"], REJECT_TINY_STATUS)
        self.assertEqual(classification["valid_target_object_count"], 3)
        self.assertEqual(classification["rejected_target_object_count"], 1)
        self.assertEqual(classification["inclusion_status"], "retained")

    def test_frozen_ambiguous_case_quarantines_whole_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parsed = self._parse(
                Path(temporary), "India_005123.jpg", [("D00", ("0", "20", "40", "80"))]
            )
            classification = classify_labelled_image(parsed, self.config)
        self.assertEqual(parsed["objects"][0]["object_status"], "whole_image_quarantine")
        self.assertEqual(classification["inclusion_status"], "quarantined")

    def test_d0w0_remains_unknown_and_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parsed = self._parse(
                Path(temporary), "India_006389.jpg", [("D0w0", ("2", "2", "20", "20"))]
            )
            classification = classify_labelled_image(parsed, self.config)
        self.assertEqual(parsed["objects"][0]["object_status"], UNKNOWN_ACTION)
        self.assertEqual(classification["inclusion_status"], "quarantined")


class GroupingAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase2b_config(
            PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b.yaml"
        )

    def _record(self, image_id: str, country: str = "India", number: int = 0) -> dict:
        return {
            "canonical_image_id": image_id,
            "country": country,
            "original_partition": "train",
            "original_filename": f"{country}_{number:06d}.jpg",
            "exif_timestamp_trustworthy": False,
            "exif_timestamp_sort_ms": None,
            "inclusion_status": "retained",
            "image_sha256": f"sha-{image_id}",
            "perceptual_hash_dhash64": f"{number:016x}",
            "_similarity_gray_bytes": bytes([number % 256]) * 1024,
            "decoded_width": 600,
            "decoded_height": 600,
            "is_positive": True,
            "target_class_counts": {name: 1 for name in ("D00", "D10", "D20", "D40")},
            "exact_duplicate_group": None,
            "derived_split": None,
        }

    def test_timestamp_grouping_and_filename_fallback(self) -> None:
        records = [self._record("a", number=1), self._record("b", number=2)]
        records[0].update(exif_timestamp_trustworthy=True, exif_timestamp_sort_ms=1_000)
        groups = assign_capture_groups(records, self.config["grouping"])
        self.assertEqual(records[0]["grouping_method"], "timestamp")
        self.assertEqual(records[1]["grouping_method"], "filename_proxy")
        self.assertEqual({item["grouping_confidence"] for item in groups}, {"medium", "low"})

    def test_exact_duplicates_get_one_hard_group(self) -> None:
        records = [self._record("a"), self._record("b")]
        records[0]["image_sha256"] = records[1]["image_sha256"] = "same"
        groups = assign_exact_duplicate_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(records[0]["exact_duplicate_group"], records[1]["exact_duplicate_group"])
        self.assertTrue(groups[0]["hard_split_constraint"])

    def test_near_duplicate_similarity_does_not_remove_images(self) -> None:
        records = [self._record("a", number=0), self._record("b", number=1)]
        records[0]["_similarity_gray_bytes"] = bytes([100]) * 1024
        records[1]["_similarity_gray_bytes"] = bytes([101]) * 1024
        candidates, _, audit = generate_near_duplicate_candidates(records, self.config)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["hard_split_constraint"])
        self.assertFalse(audit["automatic_removal_or_merge_performed"])

    def test_filename_boundary_guard_records_both_sides(self) -> None:
        lower = self._record("lower", number=95)
        upper = self._record("upper", number=105)
        groups = assign_capture_groups([lower, upper], self.config["grouping"])
        exclusions = apply_boundary_guards(
            [lower, upper],
            groups,
            {"lower": "train", "upper": "val"},
            self.config["grouping"],
        )
        self.assertEqual(set(exclusions), {"lower", "upper"})

    def test_group_split_is_deterministic_and_preserves_constraints(self) -> None:
        records = []
        for country in ("India", "Japan"):
            for number in range(900):
                records.append(self._record(f"{country}-{number}", country, number))
        assign_capture_groups(records, self.config["grouping"])
        first = plan_grouped_split(records, [], self.config["split"])
        second = plan_grouped_split(records, [], self.config["split"])
        self.assertEqual(first["image_assignments"], second["image_assignments"])
        for item in records:
            item["derived_split"] = first["image_assignments"][item["canonical_image_id"]]
        audit = split_conflict_audit(records, [])
        self.assertEqual(audit["capture_group_conflicts"], [])
        self.assertTrue(audit["all_required_countries_in_every_split"])
        self.assertTrue(audit["all_target_classes_in_every_split"])


class Phase2B1GroupingAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase2b_config(
            PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b_v1_1.yaml"
        )

    def _record(self, image_id: str, country: str, number: int) -> dict:
        return {
            "canonical_image_id": image_id,
            "country": country,
            "original_partition": "train",
            "original_filename": f"{country}_{number:06d}.jpg",
            "exif_timestamp_trustworthy": False,
            "exif_timestamp_sort_ms": None,
            "inclusion_status": "retained",
            "image_sha256": f"sha-{image_id}",
            "perceptual_hash_dhash64": f"{number:016x}",
            "_similarity_gray_bytes": bytes([number % 256]) * 1024,
            "decoded_width": 600,
            "decoded_height": 600,
            "is_positive": number % 3 != 0,
            "target_class_counts": {
                name: int((number + index) % 4 == 0)
                for index, name in enumerate(("D00", "D10", "D20", "D40"))
            },
            "exact_duplicate_group": None,
            "derived_split": None,
        }

    def test_numeric_block_terminology_and_zero_guard(self) -> None:
        lower = self._record("lower", "India", 95)
        upper = self._record("upper", "India", 105)
        groups = assign_capture_groups([lower, upper], self.config["grouping"])
        self.assertEqual(lower["grouping_method"], "numeric_block_proxy")
        self.assertEqual(upper["grouping_method"], "numeric_block_proxy")
        exclusions = apply_boundary_guards(
            [lower, upper], groups, {"lower": "train", "upper": "val"}, self.config["grouping"]
        )
        self.assertEqual(exclusions, {})

    def test_timestamp_guard_is_preserved(self) -> None:
        records = [self._record(f"t{index}", "Japan", index) for index in range(8)]
        for index, item in enumerate(records):
            item.update(
                exif_timestamp_trustworthy=True,
                exif_timestamp_sort_ms=index * 5_000,
            )
        groups = assign_capture_groups(records, self.config["grouping"])
        first_group = records[0]["capture_group"]
        assignments = {
            item["canonical_image_id"]: (
                "train" if item["capture_group"] == first_group else "val"
            )
            for item in records
        }
        exclusions = apply_boundary_guards(
            records, groups, assignments, self.config["grouping"]
        )
        self.assertTrue(exclusions)
        self.assertTrue(all(item["grouping_method"] == "timestamp" for item in records))

    def test_all_near_duplicate_candidates_are_evaluated_before_cap(self) -> None:
        records = [self._record(f"n{index}", "Japan", index) for index in range(10)]
        for index, item in enumerate(records):
            item["image_sha256"] = f"unique-{index}"
            item["perceptual_hash_dhash64"] = "0000000000000000"
            item["_similarity_gray_bytes"] = bytes([100]) * 1024
        changed = json.loads(json.dumps(self.config))
        changed["near_duplicates"]["maximum_candidates_per_image"] = 1
        candidates, _, audit = generate_near_duplicate_candidates(records, changed)
        self.assertEqual(len(candidates), 45)
        self.assertEqual(audit["automatically_evaluated_candidates"], 45)
        self.assertEqual(audit["confirmed_candidate_count"], 45)
        self.assertTrue(audit["candidate_evaluation_preceded_display_cap"])

    def test_optimized_split_is_deterministic_and_country_balanced(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["grouping"]["filename_numeric_group_size"] = 10
        records = [
            self._record(f"{country}-{number}", country, number)
            for country in ("India", "Japan")
            for number in range(300)
        ]
        assign_capture_groups(records, changed["grouping"])
        first = plan_grouped_split(records, [], changed["split"], changed["grouping"])
        second = plan_grouped_split(records, [], changed["split"], changed["grouping"])
        self.assertEqual(first["image_assignments"], second["image_assignments"])
        self.assertTrue(first["optimization_audit"]["enabled"])
        for split_name in ("train", "val", "test"):
            members = [
                item for item in records
                if first["image_assignments"][item["canonical_image_id"]] == split_name
            ]
            india_share = sum(item["country"] == "India" for item in members) / len(members)
            self.assertLessEqual(abs(india_share - 0.5), 0.05)

    def test_optimizer_does_not_increase_projected_timestamp_guard_loss(self) -> None:
        records = [
            self._record(f"{country}-time-{number}", country, number)
            for country in ("India", "Japan")
            for number in range(120)
        ]
        for item in records:
            number = int(item["original_filename"].split("_")[-1].split(".")[0])
            item.update(
                exif_timestamp_trustworthy=True,
                exif_timestamp_sort_ms=number * 5_000,
            )
        assign_capture_groups(records, self.config["grouping"])
        plan = plan_grouped_split(records, [], self.config["split"], self.config["grouping"])
        audit = plan["optimization_audit"]
        self.assertLessEqual(
            audit["after"]["projected_timestamp_guard_images"],
            audit["before"]["projected_timestamp_guard_images"],
        )
        self.assertLessEqual(
            audit["after"]["timestamp_split_transitions"],
            audit["before"]["timestamp_split_transitions"],
        )


class SyntheticConstructionIntegrationTests(unittest.TestCase):
    def test_end_to_end_reference_dataset_excludes_official_test_and_preserves_raw(self) -> None:
        cv2 = _load_opencv()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            manifests = root / "phase2a_manifests"
            output = root / "processed" / "synthetic"
            target_counts = {name: 0 for name in ("D00", "D10", "D20", "D40")}
            labelled_counts: dict[str, int] = {}
            official_counts: dict[str, int] = {}
            for country_index, country in enumerate(("India", "Japan")):
                images = raw / country / "train" / "images"
                xmls = raw / country / "train" / "annotations" / "xmls"
                tests = raw / country / "test" / "images"
                images.mkdir(parents=True)
                xmls.mkdir(parents=True)
                tests.mkdir(parents=True)
                for number in range(12):
                    filename = f"{country}_{number:06d}.jpg"
                    rng = np.random.default_rng(country_index * 100 + number)
                    image = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(images / filename), image))
                    if number < 9:
                        object_xml = "".join(
                            f"<object><name>{name}</name><bndbox><xmin>2</xmin><ymin>2</ymin>"
                            f"<xmax>20</xmax><ymax>20</ymax></bndbox></object>"
                            for name in target_counts
                        )
                        for name in target_counts:
                            target_counts[name] += 1
                    elif number == 9:
                        object_xml = (
                            "<object><name>D01</name><bndbox><xmin>2</xmin><ymin>2</ymin>"
                            "<xmax>20</xmax><ymax>20</ymax></bndbox></object>"
                        )
                    elif number == 10:
                        object_xml = ""
                    else:
                        object_xml = (
                            "<object><name>D0w0</name><bndbox><xmin>2</xmin><ymin>2</ymin>"
                            "<xmax>20</xmax><ymax>20</ymax></bndbox></object>"
                        )
                    (xmls / f"{Path(filename).stem}.xml").write_text(
                        "<annotation><size><width>32</width><height>32</height><depth>3</depth>"
                        f"</size>{object_xml}</annotation>",
                        encoding="utf-8",
                    )
                test_image = np.full((32, 32, 3), 80 + country_index, np.uint8)
                self.assertTrue(cv2.imwrite(str(tests / f"{country}_test.jpg"), test_image))
                labelled_counts[country] = 12
                official_counts[country] = 1
            before = _tree_fingerprint(raw)
            manifests.mkdir()
            (manifests / "acquisition_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "dataset": {
                            "licenses": {
                                "figshare": {"stated_license": "CC BY 4.0"},
                                "authors_repository": {"stated_license": "CC BY-SA 4.0"},
                                "project_conservative_interpretation": "CC BY-SA 4.0",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (manifests / "integrity_report.json").write_text(
                json.dumps({"raw_tree_after": before}), encoding="utf-8"
            )
            config = load_phase2b_config(
                PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b.yaml"
            )
            config["source_root"] = str(raw)
            config["phase2a_manifests_root"] = str(manifests)
            config["output_root"] = str(output)
            config["expected_source"] = {
                "labelled_images_by_country": labelled_counts,
                "official_unlabelled_test_images_by_country": official_counts,
                "target_objects": target_counts,
            }
            config["grouping"]["filename_numeric_group_size"] = 1
            config["grouping"]["filename_guard_numeric_ids"] = 0
            config["split"]["ratio_tolerance_percentage_points"] = 10.0
            config["review_artifacts"]["representative_per_class_country"] = 1
            config["review_artifacts"]["representative_negative_count"] = 2
            config["review_artifacts"]["representative_per_eval_split"] = 2
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_construction(config_path)

            self.assertEqual(result, output)
            self.assertTrue((output / "reports" / "validation_report.json").is_file())
            self.assertFalse((output / "exports" / "yolo_detection").exists())
            official = (output / "manifests" / "official_unlabelled_test.jsonl").read_text().splitlines()
            self.assertEqual(len(official), 2)
            self.assertTrue(all(json.loads(line)["derived_split"] is None for line in official))
            quarantine = (output / "manifests" / "quarantine.jsonl").read_text().splitlines()
            self.assertEqual(len(quarantine), 2)
            self.assertEqual(_tree_fingerprint(raw), before)
            validation = json.loads(
                (output / "reports" / "validation_report.json").read_text()
            )
            self.assertTrue(validation["deterministic_rerun"]["identical"])

    def test_v1_1_special_policy_is_end_to_end_and_parent_is_immutable(self) -> None:
        cv2 = _load_opencv()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            manifests = root / "phase2a_manifests"
            parent = root / "processed" / "parent_v1"
            output = root / "processed" / "v1_1"
            teacher = root / "teacher.avi"
            parent_candidates = parent / "groups" / "near_duplicate_candidates.jsonl"
            parent_candidates.parent.mkdir(parents=True)
            parent_candidates.write_text("", encoding="utf-8")
            parent_marker = parent / "dataset_card.json"
            parent_marker.write_text('{"version":"1.0"}\n', encoding="utf-8")
            parent_before = hashlib.sha256(parent_marker.read_bytes()).hexdigest()
            teacher.write_bytes(b"synthetic-teacher-video")

            names = {
                "India": [
                    *(f"India_{number:06d}.jpg" for number in range(30)),
                    "India_005123.jpg", "India_006389.jpg", "India_009013.jpg",
                ],
                "Japan": [
                    *(f"Japan_{number:06d}.jpg" for number in range(30)),
                    "Japan_000275.jpg", "Japan_001265.jpg", "Japan_003421.jpg",
                    "Japan_004342.jpg", "Japan_010631.jpg", "Japan_011217.jpg",
                    "Japan_012185.jpg",
                ],
            }
            target_counts = {name: 0 for name in ("D00", "D10", "D20", "D40")}
            official_counts: dict[str, int] = {}
            for country_index, country in enumerate(("India", "Japan")):
                image_root = raw / country / "train" / "images"
                xml_root = raw / country / "train" / "annotations" / "xmls"
                test_root = raw / country / "test" / "images"
                image_root.mkdir(parents=True)
                xml_root.mkdir(parents=True)
                test_root.mkdir(parents=True)
                for index, filename in enumerate(names[country]):
                    image = np.random.default_rng(country_index * 100 + index).integers(
                        0, 255, (32, 32, 3), dtype=np.uint8
                    )
                    self.assertTrue(cv2.imwrite(str(image_root / filename), image))
                    classes_and_boxes = [
                        (class_name, ("2", "2", "20", "20"))
                        for class_name in ("D00", "D10", "D20", "D40")
                    ]
                    if filename == "India_005123.jpg":
                        classes_and_boxes = [("D00", ("0", "2", "20", "20"))]
                    elif filename == "India_006389.jpg":
                        classes_and_boxes = [("D0w0", ("2", "2", "20", "20"))]
                    elif filename == "India_009013.jpg":
                        classes_and_boxes = [
                            ("D40", ("0", "2", "20", "20")),
                            ("D20", ("2", "2", "20", "20")),
                        ]
                    elif filename == "Japan_004342.jpg":
                        classes_and_boxes = [
                            ("D40", ("0", "2", "8", "10")),
                            ("D40", ("12", "12", "25", "25")),
                        ]
                    elif filename == "Japan_012185.jpg":
                        classes_and_boxes = [
                            ("D00", ("0", "2", "20", "20")),
                            ("D20", ("0", "2", "20", "20")),
                        ]
                    elif filename == "Japan_011217.jpg":
                        classes_and_boxes = [
                            ("D50", ("2", "2", "8", "8")),
                            ("D20", ("2", "0", "20", "20")),
                        ]
                    elif filename == "Japan_001265.jpg":
                        classes_and_boxes = [
                            ("D00", ("2", "2", "20", "20")),
                            ("D20", ("2", "2", "20", "20")),
                            ("D20", ("2", "2", "20", "20")),
                            ("D20", ("10", "10", "10", "11")),
                        ]
                    elif filename in {"Japan_003421.jpg", "Japan_010631.jpg", "Japan_000275.jpg"}:
                        classes_and_boxes = [("D20", ("0", "2", "20", "20"))]
                    object_xml = ""
                    for raw_class, box in classes_and_boxes:
                        if raw_class in target_counts:
                            target_counts[raw_class] += 1
                        object_xml += (
                            f"<object><name>{raw_class}</name><bndbox><xmin>{box[0]}</xmin>"
                            f"<ymin>{box[1]}</ymin><xmax>{box[2]}</xmax>"
                            f"<ymax>{box[3]}</ymax></bndbox></object>"
                        )
                    (xml_root / f"{Path(filename).stem}.xml").write_text(
                        "<annotation><size><width>32</width><height>32</height><depth>3</depth>"
                        f"</size>{object_xml}</annotation>", encoding="utf-8"
                    )
                test_image = np.full((32, 32, 3), 90 + country_index, np.uint8)
                self.assertTrue(cv2.imwrite(str(test_root / f"{country}_test.jpg"), test_image))
                official_counts[country] = 1

            raw_before = _tree_fingerprint(raw)
            manifests.mkdir()
            (manifests / "acquisition_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "dataset": {
                            "licenses": {
                                "figshare": {"stated_license": "CC BY 4.0"},
                                "authors_repository": {"stated_license": "CC BY-SA 4.0"},
                                "project_conservative_interpretation": "CC BY-SA 4.0",
                            }
                        },
                    }
                ), encoding="utf-8"
            )
            (manifests / "integrity_report.json").write_text(
                json.dumps({"raw_tree_after": raw_before}), encoding="utf-8"
            )
            config = load_phase2b_config(
                PROJECT_ROOT / "configs" / "dataset" / "rdd2022_phase2b_v1_1.yaml"
            )
            config["source_root"] = str(raw)
            config["phase2a_manifests_root"] = str(manifests)
            config["parent_dataset_root"] = str(parent)
            config["output_root"] = str(output)
            config["external_teacher_video"] = {
                "path": str(teacher),
                "expected_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
            }
            config["expected_source"] = {
                "labelled_images_by_country": {name: len(values) for name, values in names.items()},
                "official_unlabelled_test_images_by_country": official_counts,
                "target_objects": target_counts,
            }
            config["grouping"]["filename_numeric_group_size"] = 1
            config["review_artifacts"]["representative_per_class_country"] = 1
            config["review_artifacts"]["representative_negative_count"] = 2
            config["review_artifacts"]["representative_per_eval_split"] = 2
            config["review_artifacts"]["representative_aspect_ratio_count"] = 2
            config_path = root / "config_v1_1.yaml"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_construction(config_path)

            self.assertEqual(result, output)
            self.assertEqual(hashlib.sha256(parent_marker.read_bytes()).hexdigest(), parent_before)
            self.assertEqual(_tree_fingerprint(raw), raw_before)
            self.assertEqual(
                len((output / "manifests" / "quarantine.jsonl").read_text().splitlines()), 5
            )
            self.assertEqual(
                len((output / "manifests" / "rejected_objects.jsonl").read_text().splitlines()), 2
            )
            card = json.loads((output / "dataset_card.json").read_text())
            self.assertEqual(card["parent_version"], "1.0")
            self.assertEqual(card["dataset_version"], "1.1.0")
            self.assertFalse((output / "exports" / "yolo_detection").exists())


if __name__ == "__main__":
    unittest.main()
