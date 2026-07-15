"""Unit tests for nsa.dataset_align (no Pi cache or rawpy required)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from nsa.dataset_align import (
    DEFAULT_HCG_SCENE_RULES,
    assign_ag_tags,
    build_hcg_sort_manifest,
    cache_readiness,
    manifest_file_list,
    resolve_cache_file,
    sort_cache_into_bursts,
    split_hcg_from_mixed_burst,
    write_manifest,
)


SAMPLE_PROJECT = {
    "captures": [
        {
            "filename": "imx662_5000k_5l_00001.dng",
            "captured_at": "2026-07-10T09:00:00Z",
            "controls": {"gain": 3.24, "exposure": 1000},
        },
        {
            "filename": "imx662_5000k_5l_00002.dng",
            "captured_at": "2026-07-10T09:00:01Z",
            "controls": {"gain": 7.94, "exposure": 1000},
        },
        {
            "filename": "imx662_5000k_398l_00001.dng",
            "captured_at": "2026-07-10T08:00:00Z",
            "controls": {"gain": 3.24, "exposure": 1000},
        },
        {
            "filename": "imx662_5000k_1l_00001.dng",
            "captured_at": "2026-07-10T09:00:00Z",
            "controls": {"gain": 3.24, "exposure": 1000},
        },
        {
            "filename": "imx662_5000k_1l_00002.dng",
            "captured_at": "2026-07-10T10:15:00Z",
            "controls": {"gain": 7.94, "exposure": 1000},
        },
    ],
}


class TestManifestBuilding(unittest.TestCase):
    def test_assign_ag_tags(self) -> None:
        items = [
            {"filename": "a.dng", "gain": 3.24},
            {"filename": "b.dng", "gain": 7.94},
            {"filename": "c.dng", "gain": 999.0},
        ]
        tags = assign_ag_tags(items)
        self.assertEqual(tags["ag2"], ["a.dng"])
        self.assertEqual(tags["ag8"], ["b.dng"])
        self.assertNotIn("ag512", tags)

    def test_build_hcg_sort_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pj = Path(tmp) / "project.json"
            pj.write_text(json.dumps(SAMPLE_PROJECT), encoding="utf-8")
            manifest = build_hcg_sort_manifest(pj)
        self.assertIn("cabinet_H_2", manifest)
        self.assertEqual(
            manifest["cabinet_H_2"]["ag2"],
            ["imx662_5000k_5l_00001.dng"],
        )
        self.assertIn("cabinet_D_10", manifest)
        # F_5 time filter: only the 10:15 frame qualifies
        self.assertEqual(
            manifest["cabinet_F_5"]["ag8"],
            ["imx662_5000k_1l_00002.dng"],
        )
        self.assertNotIn("ag2", manifest["cabinet_F_5"])

    def test_manifest_file_list(self) -> None:
        manifest = {"s1": {"ag2": ["a.dng", "b.dng"], "ag4": ["c.dng"]}}
        self.assertEqual(manifest_file_list(manifest), ["a.dng", "b.dng", "c.dng"])

    def test_default_scene_rules_cover_hcg_scenes(self) -> None:
        scenes = {rule.scene for rule in DEFAULT_HCG_SCENE_RULES}
        self.assertEqual(scenes, {"cabinet_H_2", "cabinet_D_10", "cabinet_F_5"})


class TestCacheSort(unittest.TestCase):
    def test_sort_cache_into_bursts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = tmp_path / "cache"
            bursts = tmp_path / "bursts"
            cache.mkdir()
            (cache / "imx662_5000k_5l_00001.dng").write_bytes(b"dng")
            manifest = {"cabinet_H_2": {"ag2": ["imx662_5000k_5l_00001.dng"]}}
            counts = sort_cache_into_bursts(cache, bursts, manifest)
            self.assertEqual(counts["cabinet_H_2"]["ag2"], 1)
            dest = bursts / "cabinet_H_2" / "ag2" / "imx662_5000k_5l_00001.dng"
            self.assertTrue(dest.is_file())

    def test_resolve_cache_file_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            nested = cache / "imx662"
            nested.mkdir()
            fn = "imx662_5000k_5l_00099.dng"
            (nested / fn).write_bytes(b"x")
            self.assertEqual(resolve_cache_file(cache, fn), nested / fn)

    def test_cache_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "a.dng").write_bytes(b"x")
            manifest = {"s": {"ag2": ["a.dng", "missing.dng"]}}
            report = cache_readiness(cache, manifest)
            self.assertEqual(report["wanted_files"], 2)
            self.assertEqual(report["present_files"], 1)
            self.assertEqual(report["fraction"], 0.5)


class TestH2Split(unittest.TestCase):
    def test_split_hcg_from_mixed_burst(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bursts = Path(tmp) / "bursts"
            mixed = bursts / "cabinet_H_2" / "ag2"
            mixed.mkdir(parents=True)
            (mixed / "burst_001.dng").write_bytes(b"lcg")
            hcg = mixed / "imx662_5000k_5l_00001.dng"
            hcg.write_bytes(b"hcg")
            moved = split_hcg_from_mixed_burst(bursts, gains=[2])
            self.assertEqual(moved["ag2"], 1)
            self.assertTrue((bursts / "cabinet_H_2_hcg" / "ag2" / hcg.name).is_file())
            self.assertTrue((mixed / "burst_001.dng").is_file())
            self.assertFalse(hcg.exists())


class TestWriteManifest(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hcg_sort_manifest.json"
            data = {"cabinet_H_2": {"ag2": ["a.dng"]}}
            write_manifest(data, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, data)


class TestAlignPiCache(unittest.TestCase):
    def test_pairs_without_manifest_skips(self) -> None:
        from nsa.dataset_align import align_pi_cache

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            (project / "PI_RAW" / "Data").mkdir(parents=True)
            (project / "bursts").mkdir()
            result = align_pi_cache(
                cache_root=Path(tmp) / "missing_cache",
                project_root=project,
                sort=False,
                build_pairs=True,
            )
            self.assertTrue(any("manifest" in s for s in result.skipped))


class TestExistingHcgPairs(unittest.TestCase):
    """Smoke-test against repo-local HCG pairs (no rawpy needed)."""

    def test_local_imx662h_pairs_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        data = root / "datasets" / "PI_RAW" / "Data"
        if not data.is_dir():
            self.skipTest("datasets/PI_RAW not present")
        hcg_dirs = list(data.rglob("imx662h_ag*_test"))
        if not hcg_dirs:
            self.skipTest("no imx662h pairs on disk")
        sample = hcg_dirs[0]
        self.assertTrue((sample / "noisy.dng").is_file())
        self.assertTrue((sample / "gt.tif").is_file())
        gain = json.loads((sample / "gain.json").read_text(encoding="utf-8"))
        self.assertTrue(gain.get("hcg_enabled"))


if __name__ == "__main__":
    unittest.main()
