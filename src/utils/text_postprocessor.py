import re
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Common regex fixes for typical OCR optical errors in Vietnamese texts
COMMON_OCR_REPLACEMENTS = [
    # Capital 'I' mistaken for lowercase 'l' at word start (e.g. Iớp -> lớp, Iịch -> lịch)
    (r"\bI([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])", r"l\1"),
    
    # Common words with corrupted diacritics / letters
    (r"\bNCUYỄN\b", "NGUYỄN"),
    (r"\bNCUYEN\b", "NGUYEN"),
    (r"\bNCUVỄN\b", "NGUYỄN"),
    (r"\bNCUNỂN\b", "NGUYỄN"),
    (r"\bĐẩng chí\b", "Đồng chí"),
    (r"\bĐổng chí\b", "Đồng chí"),
    (r"\bĐổng chỉ\b", "Đồng chí"),
    (r"\bđổng chí\b", "đồng chí"),
    (r"\bchí([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])", r"chí \1"),
    (r"\bcí\b", "chí"),
    
    # Common historical / administrative terms
    (r"\bnuoc\b", "nước"),
    (r"\bnuóc\b", "nước"),
    (r"\bngưdi\b", "người"),
    (r"\bđổng thờl\b", "đồng thời"),
    (r"\bvăn hlến\b", "văn hiến"),
    (r"\bgiao luu\b", "giao lưu"),
    (r"\bthực hành đặc trung\b", "thực hành đặc trưng"),
    (r"\bhuyện ủỵ\b", "Huyện ủy"),
    (r"\bHuyện ủỵ\b", "Huyện ủy"),
    (r"\bHuyệnửỵ\b", "Huyện ủy"),
    (r"\bHuyệnửy\b", "Huyện ủy"),
    (r"\bHuyệnủy\b", "Huyện ủy"),
    (r"\bHuyện úỵ\b", "Huyện ủy"),
    (r"\bBan Thường vu\b", "Ban Thường vụ"),
    (r"\bChlhuy\b", "Chỉ huy"),
    (r"\bGuânshuyện\b", "Quân sự huyện"),
    (r"\bChủ tjch\b", "Chủ tịch"),
    (r"\bthành vỈên\b", "thành viên"),
    (r"\bIruông ban\b", "Trưởng ban"),
    (r"\bTruởng\b", "Trưởng"),
    (r"\bIão thành\b", "lão thành"),
    (r"\btiển khởi nghĩa\b", "tiền khởi nghĩa"),
    
    # Numbers with letter 'O' instead of zero '0'
    (r"(\d+)\.00O\b", r"\g<1>.000"),
    (r"(\d+)\.0O0\b", r"\g<1>.000"),
    (r"(\d+)O\b", r"\g<1>0"),
]


class TextPostProcessor:
    def __init__(self, use_dictionary: bool = True):
        self.use_dictionary = use_dictionary
        self.rev_map = None
        if use_dictionary:
            self._load_dictionary()

    def _load_dictionary(self):
        try:
            import underthesea
            pkg = Path(underthesea.__file__).parent
            dict_path = None
            for name in ["Viet74K.txt", "Viet39K.txt", "Viet22K.txt"]:
                p = pkg / "transformer" / name
                if p.exists():
                    dict_path = p
                    break

            if dict_path:
                mapping = defaultdict(list)
                with open(dict_path, encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            k = self._strip_accents(w).lower()
                            if k and k != w.lower():
                                mapping[k].append(w)
                self.rev_map = {k: v[0] for k, v in mapping.items()}
        except Exception:
            self.rev_map = None

    @staticmethod
    def _strip_accents(text: str) -> str:
        nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        stripped = stripped.replace("đ", "d").replace("Đ", "D")
        return unicodedata.normalize("NFC", stripped)

    def clean_text(self, text: str) -> str:
        """Apply regex-based rule corrections and optional dictionary tone restoration."""
        if not text:
            return ""

        result = text
        # 1. Regex rule fixes
        for pattern, repl in COMMON_OCR_REPLACEMENTS:
            result = re.sub(pattern, repl, result)

        # 2. Dictionary diacritic restoration on unaccented words (optional)
        if self.rev_map:
            lines = result.splitlines()
            restored_lines = []
            for line in lines:
                tokens = re.split(r"(\s+)", line)
                out_tokens = []
                for token in tokens:
                    if not token.strip():
                        out_tokens.append(token)
                        continue
                    key = self._strip_accents(token).lower()
                    # Only restore if token has no accents at all and exists in map
                    if token == self._strip_accents(token) and key in self.rev_map:
                        restored = self.rev_map[key]
                        if token.isupper():
                            out_tokens.append(restored.upper())
                        elif token[0].isupper():
                            out_tokens.append(restored.capitalize())
                        else:
                            out_tokens.append(restored)
                    else:
                        out_tokens.append(token)
                restored_lines.append("".join(out_tokens))
            result = "\n".join(restored_lines)

        return result
