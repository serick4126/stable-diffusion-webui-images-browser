"""Compare the base ref against the working tree: identical outputs, then timings.

python tests/verify_perf.py --sandbox DIR [--base main] [--db wib.sqlite3] [--dirs D ...] [--runs 5]
python tests/verify_perf.py --sandbox DIR --only-load-exif   (stdlib only, for old SQLite runtimes)
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import logging
import os
import random
import re
import shutil
import sqlite3
import stat
import statistics
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent
WIB_DB = "scripts/wib/wib_db.py"
IMAGE_BROWSER = "scripts/image_browser.py"
UP, DOWN = "\U000025b2", "\U000025bc"
IB_FUNCS = ["atof", "natural_keys", "exif_search", "traverse_all_files", "cache_exif",
            "get_ranking", "get_all_images", "_process_exif_data"]
SORTS = ["path name", "date", "aesthetic_score", "cfg scale", "steps", "seed", "sampler", "size", "model", "model hash", "ranking"]


def source(ref, rel):
    if ref is None:
        return (REPO / rel).read_text(encoding="utf-8")
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{rel}"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def load_wib_db(src, sandbox, name):
    sandbox.mkdir(parents=True, exist_ok=True)
    mods = types.ModuleType("modules")
    mods.scripts = types.SimpleNamespace(basedir=lambda: str(sandbox))
    mods.shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace())
    sys.modules.update({"modules": mods, "modules.scripts": mods.scripts, "modules.shared": mods.shared})
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        pil = types.ModuleType("PIL")
        pil.Image = types.ModuleType("PIL.Image")
        sys.modules.update({"PIL": pil, "PIL.Image": pil.Image})
    path = sandbox / f"{name}_wib_db.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"wib_db_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    root = sandbox.resolve()
    default_db = inspect.unwrap(mod.transaction).__defaults__[0]
    for value in (mod.db_file, mod.source_db_file, default_db):
        assert Path(value).resolve().is_relative_to(root), f"connection target escaped sandbox: {value}"
    assert not getattr(mod.shared.cmd_opts, "image_browser_tmp_db", False), "tmp db must be disabled"
    assert mod.db_file != mod.tmp_db_file, "db_file must not be the tmp db"
    return mod


quiet_logger = logging.getLogger("verify")
quiet_logger.setLevel(logging.ERROR)


def load_ib(src, wib_db, opts):
    from PIL import Image, UnidentifiedImageError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in IB_FUNCS]
    assert {f.name for f in funcs} == set(IB_FUNCS), "image_browser.py functions not found"
    ns = {
        "os": os, "re": re, "sys": sys, "stat": stat, "time": time, "random": random,
        "Image": Image, "UnidentifiedImageError": UnidentifiedImageError,
        "wib_db": wib_db, "opts": opts, "logger": quiet_logger,
        "modules": types.SimpleNamespace(extras=types.SimpleNamespace(
            run_pnginfo=lambda image: ("", image.info.get("parameters"), ""))),
        "up_symbol": UP, "down_symbol": DOWN, "np": "negative_prompt: ",
        "image_ext_list": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg", ".avif"],
        "video_ext_list": [".mp4", ".mov", ".avi", ".wmv", ".flv", ".mkv", ".webm", ".mpeg", ".mpg", ".3gp", ".ogv", ".m4v"],
        "List": List, "Tuple": Tuple, "current_depth": 0, "yappi_do": False, "path_maps": {}, "exif_cache": {}, "aes_cache": {},
    }
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "image_browser.py", "exec"), ns)
    return ns


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as e:
        return ("err", type(e).__name__)


def opts_stub(scan_exif=True):
    return types.SimpleNamespace(image_browser_scan_exif=scan_exif, image_browser_with_subdirs=True)


def call_all(ns, d, sort_by="date", order=DOWN, keyword="", ranking="All", rmin="", rmax="", amin="", amax="",
             exif_kw="", neg="No", regex=False):
    return outcome(ns["get_all_images"], d, sort_by, order, keyword, "image_browser_tab_txt2img", 0,
                   ranking, rmin, rmax, amin, amax, exif_kw, neg, regex)


class Pair:
    """Old and new code, each bound to its own copy of the same initial DB."""

    def __init__(self, args, sandbox, db=None, scan_exif=True):
        self.sides = {}
        for name, ref in (("old", args.base), ("new", None)):
            box = sandbox / name
            if box.exists():
                shutil.rmtree(box)
            box.mkdir(parents=True)
            if db is not None:
                shutil.copyfile(db, box / "wib.sqlite3")
            wib = load_wib_db(source(ref, WIB_DB), box, name)
            if db is None:
                wib.check()
            ns = load_ib(source(ref, IMAGE_BROWSER), wib, opts_stub(scan_exif))
            self.sides[name] = (wib, ns)

    def each(self, fn):
        return {name: fn(wib, ns) for name, (wib, ns) in self.sides.items()}


failures = []


def summary(value):
    found = {"ok": 0, "err": set(), "files": 0}
    def walk(v):
        if isinstance(v, tuple) and len(v) == 2 and v[0] in ("ok", "err"):
            if v[0] == "err":
                found["err"].add(v[1])
            else:
                found["ok"] += 1
                found["files"] += len(v[1])
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            found["files"] += len(v)
    walk(value)
    return f"ok={found['ok']} err={sorted(found['err'])} items={found['files']}"


def check(label, results):
    same = results["old"] == results["new"]
    if not same:
        failures.append(label)
    print(f"  [{'ok' if same else 'DIFF'}] {label}  ({summary(results['new'])})")


def backup_copy(src, dst):
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as s, sqlite3.connect(dst) as d:
        s.execute("PRAGMA query_only=ON")
        s.backup(d)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    return h.hexdigest(), st.st_size, st.st_mtime_ns


EXIF_ROWS = [
    ("a.png", "prompt", "1girl, solo\nsecond line"), ("a.png", "negative_prompt", "bad, worse"),
    ("a.png", "Steps", "20"), ("a.png", "CFG scale", "7"), ("a.png", "Model", "m,with,comma"),
    ("b.png", "prompt", ""), ("b.png", "Seed", "42"), ("b.png", "Note", "literal \\n here"),
    ("c.png", "negative_prompt", "only negative"), ("c.png", "Size", "512x768"),
    ("d.png", "Version", "v1"), ("d.png", "Model hash", "abc"), ("d.png", "Clip skip", "2"),
]


def suite_load_exif(args, sandbox):
    print("load_exif_data / load_exif_data_by_key")
    for version in (None, (3, 43, 2)):
        for rows, seed in ((EXIF_ROWS, {}), ([], {}), (EXIF_ROWS, {"zz.png": "keep", "a.png": "overwrite me"})):
            def run(wib, ns):
                with wib.transaction() as cur:
                    cur.execute("DELETE FROM exif_data")
                    cur.executemany("INSERT INTO exif_data (file, key, value) VALUES (?, ?, ?)", rows)
                return wib.load_exif_data(dict(seed)), wib.load_exif_data_by_key({}, "aesthetic_score", "Score")
            pair = Pair(args, sandbox / "load_exif")
            if version is not None:
                for wib, _ in pair.sides.values():
                    wib.sqlite3 = types.SimpleNamespace(**{k: getattr(sqlite3, k) for k in dir(sqlite3) if not k.startswith("__")})
                    wib.sqlite3.sqlite_version_info = version
            check(f"rows={len(rows)} seed={len(seed)} sqlite={version or sqlite3.sqlite_version}", pair.each(run))


def make_images(folder, n):
    from PIL import Image, PngImagePlugin
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    models = ["alpha", "beta", "Gamma", "delta 2", "delta 10"]
    for i in range(n):
        info = PngImagePlugin.PngInfo()
        if i % 9 != 8:
            cfg = rng.choice(["7", "4.5", "11", "7"])
            info.add_text("parameters", f"prompt {i}, tag\nNegative prompt: neg {i % 3}\n"
                          f"Steps: {rng.randint(10, 40)}, Sampler: Euler, CFG scale: {cfg}, Seed: {rng.randint(0, 99)}, "
                          f"Size: 64x{64 + i % 2 * 32}, Model hash: h{i % 4}, Model: {rng.choice(models)}, "
                          f"aesthetic_score: {rng.choice(['5.5', '6.25', '4.0', '5.5'])}")
        Image.new("RGB", (8, 8), (i, 0, 0)).save(folder / f"img_{i:03d}.png", pnginfo=info)
        os.utime(folder / f"img_{i:03d}.png", (1_700_000_000 + i % 7, 1_700_000_000 + i % 7))
    (folder / "sub").mkdir(exist_ok=True)
    Image.new("RGB", (8, 8)).save(folder / "sub" / "nested.png")


def set_rankings(pair, folder, with_null=False):
    files = sorted(str(p) for p in Path(folder).rglob("*.png"))
    rows = [(f, os.path.basename(f), str(1 + i % 5)) for i, f in enumerate(files[:20])]
    rows += [(str(Path(folder) / "outside.png"), "outside.png", "5")]
    if with_null:
        rows += [(files[20], os.path.basename(files[20]), None)]
    for wib, _ in pair.sides.values():
        with wib.transaction() as cur:
            cur.execute("DELETE FROM ranking")
            cur.executemany("INSERT INTO ranking (file, name, ranking) VALUES (?, ?, ?)", rows)


def suite_synthetic(args, sandbox):
    print("get_all_images on synthetic folders")
    folder_a, folder_b, empty = sandbox / "imgs_a", sandbox / "imgs_b", sandbox / "imgs_empty"
    for f in (folder_a, folder_b, empty):
        if f.exists():
            shutil.rmtree(f)
    make_images(folder_a, 45)
    make_images(folder_b, 12)
    empty.mkdir()
    a, b, e = (os.path.realpath(f) for f in (folder_a, folder_b, empty))

    pair = Pair(args, sandbox / "synthetic")
    check("first scan fills caches", pair.each(lambda w, ns: (call_all(ns, a), call_all(ns, b), call_all(ns, e))))
    check("exif and aes caches", pair.each(lambda w, ns: (ns["exif_cache"], ns["aes_cache"])))
    set_rankings(pair, folder_a)
    for sort_by in SORTS:
        for order in (UP, DOWN):
            check(f"sort {sort_by} {'up' if order == UP else 'down'}", pair.each(lambda w, ns: (call_all(ns, a, sort_by, order), call_all(ns, e, sort_by, order))))
    check("ranking sort after a ranking change", pair.each(
        lambda w, ns: (w.update_ranking(os.path.join(a, "img_000.png"), "1"), call_all(ns, a, "ranking", UP))))
    set_rankings(pair, folder_a, with_null=True)
    check("ranking sort with a NULL ranking row keeps TypeError", pair.each(
        lambda w, ns: [call_all(ns, a, "ranking", o) for o in (UP, DOWN)]))

    filters = [dict(ranking="1"), dict(ranking="5"), dict(ranking="None"), dict(ranking="Min-max", rmin="2", rmax="4"),
               dict(ranking="Min-max", rmin="x", rmax="9"), dict(amin="5"), dict(amax="5.5"), dict(amin="4.5", amax="6"),
               dict(amin="bad", amax=""), dict(amin="5", ranking="Min-max", rmin="1", rmax="3"),
               dict(exif_kw="neg 1", amin="5"), dict(exif_kw="Gamma", ranking="None"),
               dict(exif_kw="alpha", neg="Only", ranking="3"), dict(exif_kw="neg", neg="Yes", regex=True, amin="1"),
               dict(keyword="img_01", ranking="2")]
    for flt in filters:
        check(f"filter {flt}", pair.each(lambda w, ns: call_all(ns, a, **flt)))

    def sequence(w, ns):
        return [call_all(ns, a, ranking="Min-max", rmin="2", rmax="3"), call_all(ns, b), call_all(ns, b, amin="5"),
                call_all(ns, a, amin="99"), call_all(ns, a, ranking="4"), call_all(ns, a), call_all(ns, a, amin="5.5", amax="5.5")]
    check("filter, plain, refilter sequence across folders", pair.each(sequence))

    no_scan = Pair(args, sandbox / "noscan", scan_exif=False)
    check("empty cache, exif sort", no_scan.each(lambda w, ns: [call_all(ns, a, s, o) for s in ("seed", "cfg scale") for o in (UP, DOWN)]))
    def partial(w, ns):
        ns["exif_cache"] = {os.path.join(a, "img_000.png"): "Seed: 3, CFG scale: 7"}
        return [call_all(ns, a, s, o) for s in ("seed", "cfg scale") for o in (UP, DOWN)] + [call_all(ns, e, "seed", UP)]
    check("non-empty cache missing target files keeps KeyError", no_scan.each(partial))


def suite_real(args, sandbox):
    master = sandbox / "real_master.sqlite3"
    if not master.exists():
        backup_copy(args.db, master)
    dirs = [os.path.realpath(d) for d in args.dirs]
    print(f"real DB and folders: {dirs}")
    pair = Pair(args, sandbox / "real", db=master)

    t = {}
    def load(w, ns):
        t0 = time.perf_counter()
        ns["exif_cache"] = w.load_exif_data({})
        ns["aes_cache"] = w.load_exif_data_by_key({}, "aesthetic_score", "Score")
        t[id(ns)] = time.perf_counter() - t0
        return ns["exif_cache"], ns["aes_cache"]
    check("startup caches on real DB", pair.each(load))
    check("first visit scans uncached files", pair.each(lambda w, ns: [call_all(ns, d) for d in dirs]))
    check("caches after first visit", pair.each(lambda w, ns: (ns["exif_cache"], ns["aes_cache"])))
    for d in dirs:
        for sort_by in ("date", "path name", "ranking", "cfg scale", "seed", "model", "aesthetic_score"):
            check(f"{os.path.basename(d)} sort {sort_by}", pair.each(lambda w, ns: [call_all(ns, d, sort_by, o) for o in (UP, DOWN)]))
        for flt in (dict(amin="5"), dict(ranking="None"), dict(exif_kw="1girl", amax="6")):
            check(f"{os.path.basename(d)} filter {flt}", pair.each(lambda w, ns: call_all(ns, d, **flt)))
    return pair, dirs


def timings(args, pair, dirs):
    print(f"timings: median of {args.runs} interleaved runs per side, seconds")
    cases = [("load_exif_data", None, lambda w, ns, d: w.load_exif_data({}))]
    for d in dirs:
        base = os.path.basename(d)
        cases += [
            (f"{base} page turn, date", d, lambda w, ns, d: call_all(ns, d)),
            (f"{base} sort cfg scale", d, lambda w, ns, d: call_all(ns, d, "cfg scale")),
            (f"{base} sort ranking", d, lambda w, ns, d: call_all(ns, d, "ranking")),
        ]
    rows = []
    for label, d, fn in cases:
        samples = {"old": [], "new": []}
        for i in range(args.runs):
            for name in (("old", "new") if i % 2 == 0 else ("new", "old")):
                wib, ns = pair.sides[name]
                t0 = time.perf_counter()
                result = fn(wib, ns, d)
                samples[name].append(time.perf_counter() - t0)
                assert not (isinstance(result, tuple) and result[0] == "err"), f"{label} {name} raised {result[1]}"
        o, n = statistics.median(samples["old"]), statistics.median(samples["new"])
        rows.append((label, o, n, samples))
        print(f"  {label:40s} old {o:7.3f} [{min(samples['old']):.3f}-{max(samples['old']):.3f}]"
              f"  new {n:7.3f} [{min(samples['new']):.3f}-{max(samples['new']):.3f}]  x{o / n:5.2f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sandbox", required=True, type=Path)
    p.add_argument("--base", default="main")
    p.add_argument("--db", default=str(REPO / "wib.sqlite3"))
    p.add_argument("--dirs", nargs="*", default=[])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--only-load-exif", action="store_true")
    args = p.parse_args()
    sandbox = args.sandbox.resolve()
    assert not sandbox.is_relative_to(REPO), "sandbox must be outside the repository"
    sandbox.mkdir(parents=True, exist_ok=True)
    print(f"python {sys.version.split()[0]}, sqlite {sqlite3.sqlite_version}, base {args.base}")

    guarded = [Path(args.db)] + [REPO.parent.parent / "config.json"]
    before = {str(g): fingerprint(g) for g in guarded if g.exists()}

    suite_load_exif(args, sandbox)
    if not args.only_load_exif:
        suite_synthetic(args, sandbox)
        if args.dirs:
            pair, dirs = suite_real(args, sandbox)
            if not failures:
                timings(args, pair, dirs)

    after = {str(g): fingerprint(g) for g in guarded if g.exists()}
    assert before == after, "production DB or config.json changed"
    print(f"production files unchanged: {list(before)}")
    print("FAILED: " + "; ".join(failures) if failures else "all equivalence checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
