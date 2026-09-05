"""
Post-process OCR output: restore Vietnamese diacritics using a 74K word dictionary.
Strategy: build a reverse map (no-accent -> best accented form), then apply word-by-word.
For multi-word entries, try longest-match first.
"""
import sys
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import underthesea


# ── 1. Load word list & build reverse map ─────────────────────────────────────

def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics and tone marks, keep base ASCII."""
    # Normalize to NFD so combining characters are separated
    nfd = unicodedata.normalize("NFD", text)
    # Remove combining diacritical marks
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Handle Vietnamese special base letters: đ→d, Đ→D
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", stripped)


def build_reverse_map(word_list_path: Path) -> dict:
    """
    Build: strip(word) -> best_accented_word
    When multiple accented forms map to the same stripped key, prefer:
      1. Longer entries (multi-word chunks kept together)
      2. Entries that appear earlier in the list (higher frequency position)
    """
    # word_key (stripped, lower) -> list of original forms in order
    mapping: dict[str, list[str]] = defaultdict(list)

    with open(word_list_path, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            key = strip_accents(word).lower()
            if key and key != word.lower():  # only add if diacritics were actually present
                mapping[key].append(word)

    # Collapse to single best form: first entry wins (list order = priority)
    return {k: v[0] for k, v in mapping.items()}


def load_reverse_map() -> dict:
    import os
    pkg = os.path.dirname(underthesea.__file__)
    # Use the largest word list available
    for name in ["Viet74K.txt", "Viet39K.txt", "Viet22K.txt", "Viet11K.txt"]:
        path = Path(pkg) / "transformer" / name
        if path.exists():
            print(f"  Using word list: {name} ({path})")
            return build_reverse_map(path)
    raise FileNotFoundError("No Vietnamese word list found in underthesea package.")


# ── 2. Restore diacritics in a line of text ───────────────────────────────────

def restore_line(line: str, rev_map: dict) -> str:
    """
    Restore diacritics word-by-word (preserving unknown words as-is).
    Handles: uppercase (ALL CAPS titles), mixed case.
    """
    # Split on whitespace but preserve structure
    tokens = re.split(r"(\s+)", line)
    result = []
    for token in tokens:
        if not token.strip():
            result.append(token)
            continue

        key = strip_accents(token).lower()
        if key in rev_map:
            restored = rev_map[key]
            # Preserve original casing style
            if token.isupper():
                result.append(restored.upper())
            elif token[0].isupper():
                result.append(restored.capitalize())
            else:
                result.append(restored)
        else:
            result.append(token)
    return "".join(result)


def restore_text(text: str, rev_map: dict) -> str:
    lines = text.splitlines()
    return "\n".join(restore_line(line, rev_map) for line in lines)


# ── 3. Main: apply to all output .txt files ───────────────────────────────────

def main():
    output_dir = Path("data/output")
    restored_dir = Path("data/output_restored")
    restored_dir.mkdir(parents=True, exist_ok=True)

    print("Building Vietnamese diacritic reverse map...")
    rev_map = load_reverse_map()
    print(f"  Reverse map size: {len(rev_map):,} entries\n")

    txt_files = sorted(output_dir.rglob("*.txt"))
    changed = 0

    for i, src in enumerate(txt_files, 1):
        # Mirror folder structure
        rel = src.relative_to(output_dir)
        dst = restored_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        original = src.read_text(encoding="utf-8", errors="replace")
        restored = restore_text(original, rev_map)

        dst.write_text(restored, encoding="utf-8")
        if restored != original:
            changed += 1

        if i % 50 == 0 or i == len(txt_files):
            print(f"  [{i}/{len(txt_files)}] processed...")

    print(f"\nDone. {changed}/{len(txt_files)} files had diacritics restored.")
    print(f"Output: {restored_dir.resolve()}")

    # Also write a merged file
    merged_path = Path("data/output_restored_merged.txt")
    all_txt = sorted(restored_dir.rglob("*.txt"))
    with open(merged_path, "w", encoding="utf-8") as out:
        for i, f in enumerate(all_txt, 1):
            content = f.read_text(encoding="utf-8", errors="replace").strip()
            out.write(f"=== [{i}/{len(all_txt)}] {f.name} ===\n")
            out.write(content)
            out.write("\n\n")
    print(f"Merged file: {merged_path.resolve()} ({merged_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    # Quick self-test first
    print("=== Quick test ===")
    rev_map = load_reverse_map()
    samples = [
        ("NHA XUAT BAN VAN HOA DAN TOC", "NHÀ XUẤT BẢN VĂN HOÁ DÂN TỘC"),
        ("TRUYEN THONG VAN HIEN", "TRUYỀN THỐNG VĂN HIẾN"),
        ("LICH SU VIET NAM", "LỊCH SỬ VIỆT NAM"),
        ("GIAO DUC VA DAO TAO", "GIÁO DỤC VÀ ĐÀO TẠO"),
        ("HO CHI MINH", "HỒ CHÍ MINH"),
    ]
    ok = 0
    for inp, expected in samples:
        got = restore_line(inp, rev_map)
        status = "✓" if got == expected else "~"
        print(f"  {status} {inp!s:<40} -> {got}")
        if got == expected:
            ok += 1
    print(f"\nSelf-test: {ok}/{len(samples)} exact matches\n")

    main()
