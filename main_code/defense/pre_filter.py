import re
import unicodedata
from typing import Any

from tree_sitter import Query, QueryCursor

from guardrail_common import (
    adjusted_special_ratio,
    get_comment_query,
    get_string_query,
    is_common_code_identifier,
    is_normal_c_format_string,
    node_kind,
    s1_special_threshold,
)


class PreFilter:
    def __init__(
        self,
        parser,
        language,
        lang_name="c",
        s1_word=100,
        s1_str=None,
        s1_other=None,
        s1_ascii=0.001,
        s1_identifier=None,
        s1_comment=None,
        s1_error=None,
        mixed_error_ratio=0.15,
    ):
        self.parser = parser
        self.language = language
        self.lang_name = lang_name.lower()

        self.s1_word = s1_word
        self.s1_str = s1_str
        self.s1_other = s1_other
        self.s1_ascii = s1_ascii
        self.s1_identifier = s1_identifier
        self.s1_comment = s1_comment
        self.s1_error = s1_error
        self.mixed_error_ratio = mixed_error_ratio

        self.string_patterns = {
            "SQL_Injection": re.compile(
                r"(?i)\b(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+.+?\s+SET)\b|--\s*$|\bOR\s+1\s*=\s*1\b"
            ),
            "Shell_Injection": re.compile(
                r"(?i)(;\s*(rm\s+-rf|wget|curl|nc|bash|sh|perl|python)\b|\|\s*(bash|sh)|>\s*/dev/null|/bin/sh)"
            ),
            "Path_Traversal": re.compile(
                r"(?i)(\.\./\.\./|\betc/passwd\b|\betc/shadow\b|%2e%2e%2f)"
            ),
            "Prompt_Template_Injection": re.compile(
                r"(?i)(\{\{.*?\}\}|\(\)\s*=>|<script>|javascript:|\[\'\$[A-Za-z]+)"
            ),
            "Chr_Obfuscation": re.compile(
                r"(chr\(\d+\)\s*\+\s*){2,}chr\(\d+\)"
            ),
        }

        self.string_query = Query(self.language, get_string_query(self.lang_name))
        self.identifier_query = Query(self.language, "(identifier) @identifier")
        self.error_query = Query(self.language, "(ERROR) @error")
        self.comment_query = Query(self.language, get_comment_query(self.lang_name))

    def get_captures_from_query(self, query, root_node):
        cursor = QueryCursor(query)
        captures = []
        for name, nodes in cursor.captures(root_node).items():
            for node in nodes:
                captures.append((node, name))
        captures.sort(key=lambda x: x[0].start_byte)
        return captures

    def _threshold_for_kind(self, kind: str) -> float:
        return s1_special_threshold(
            self.lang_name,
            kind,
            s1_str=self.s1_str,
            s1_identifier=self.s1_identifier,
            s1_comment=self.s1_comment,
            s1_error=self.s1_error,
            s1_other=self.s1_other,
        )

    def _identifier_anomaly(self, text: str) -> tuple[bool, str | None]:
        if is_common_code_identifier(text):
            return False, None
        if text.startswith("_dead_"):
            return True, f"Suspicious_Identifier ({text[:20]}...)"
        sub_words = re.split(r"_|(?=[A-Z])", text)
        max_sub_len = max((len(w) for w in sub_words if w), default=0)
        if max_sub_len > 25:
            return True, f"Suspicious_Identifier ({text[:20]}...)"
        return False, None

    def _check_structural_anomaly(self, text: str, node_type: str, capture_name: str | None = None):
        kind = node_kind(node_type, capture_name)

        if kind == "identifier":
            hit, reason = self._identifier_anomaly(text)
            if hit:
                return True, reason

        control_chars = sum(
            1 for c in text if unicodedata.category(c).startswith("C") and c not in "\n\r\t"
        )
        if control_chars > 0:
            if kind == "comment":
                return True, "Comment_Abnormal_Control_Chars"
            return True, "Abnormal_Control_Chars"

        # Comments are allowed to contain natural language and punctuation.  After
        # checking control characters, only long/suspicious comments should reach
        # the ratio checks below.
        if len(text) < 15:
            return False, None

        if kind == "string" and is_normal_c_format_string(text, self.lang_name):
            return False, None

        if kind not in {"string", "comment"}:
            max_word_len = max((len(w) for w in text.split()), default=0)
            if max_word_len > self.s1_word:
                return True, f"Long_Continuous_String ({max_word_len})"

        if self.lang_name == "python" and kind == "error":
            # Python doctest / REPL prompts are common in docstrings.  Do not skip
            # `->`; it is exactly one of the useful signals for C-in-Python wrappers.
            if ">>>" in text and not re.search(r"\b(static|struct|unsigned|void|int|long|sizeof)\b", text):
                return False, None

        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > 5 and kind not in {"comment", "string"}:
            non_ascii_ratio = non_ascii_count / max(1, len(text))
            if non_ascii_ratio > self.s1_ascii:
                return True, f"High_Non_ASCII_Ratio ({non_ascii_ratio:.2f})"

        special_ratio = adjusted_special_ratio(text, kind)
        threshold = self._threshold_for_kind(kind)
        if special_ratio > threshold:
            return True, f"High_Special_Char_Ratio ({special_ratio:.2f}>{threshold:.2f})"

        return False, None

    def _detect_mixed_language_error(self, tree, code_bytes: bytes) -> list[dict[str, Any]]:
        try:
            error_nodes = [node for node, _ in self.get_captures_from_query(self.error_query, tree.root_node)]
        except Exception:
            return []

        if not error_nodes:
            return []

        total_error_bytes = sum(max(0, n.end_byte - n.start_byte) for n in error_nodes)
        error_ratio = total_error_bytes / max(1, len(code_bytes))
        hits: list[dict[str, Any]] = []

        c_like_keywords = re.compile(
            r"\b(static|struct|unsigned|signed|void|int|long|short|sizeof|return\s+-E[A-Z]+|goto|#include|typedef|enum)\b"
        )
        c_like_syntax = re.compile(r"(;|\{|\}|->|\[[^\]]*\]|\([^)]+\)\s*\{)")

        c_like_error_bytes = 0
        for node in error_nodes:
            text = node.text.decode("utf8", errors="ignore")
            looks_c_like = c_like_keywords.search(text) is not None and c_like_syntax.search(text) is not None
            if looks_c_like:
                c_like_error_bytes += max(0, node.end_byte - node.start_byte)
                hits.append(
                    {
                        "type": "Mixed_Language_C_In_Python" if self.lang_name == "python" else "Mixed_Language_Syntax_Block",
                        "span": (node.start_byte, node.end_byte),
                        "matched_text": text[:80].replace("\n", " "),
                    }
                )

        c_like_ratio = c_like_error_bytes / max(1, len(code_bytes))
        if self.lang_name == "python" and (error_ratio > self.mixed_error_ratio or c_like_ratio > 0.06):
            hits.append(
                {
                    "type": f"High_ERROR_Node_Ratio ({error_ratio:.2f})",
                    "span": (error_nodes[0].start_byte, error_nodes[-1].end_byte),
                    "matched_text": "parser-error-density",
                }
            )
        elif error_ratio > 0.30 and c_like_error_bytes > 0:
            hits.append(
                {
                    "type": f"High_ERROR_Node_Ratio ({error_ratio:.2f})",
                    "span": (error_nodes[0].start_byte, error_nodes[-1].end_byte),
                    "matched_text": "parser-error-density",
                }
            )

        # Deduplicate identical spans/types.
        seen = set()
        deduped = []
        for hit in hits:
            key = (hit["type"], hit["span"])
            if key not in seen:
                seen.add(key)
                deduped.append(hit)
        return deduped

    def _detect_dead_decoys(self, tree, code_bytes):
        if self.lang_name == "java":
            query_funcs_str = "(method_declaration) @func"
        else:
            query_funcs_str = "(function_definition) @func"

        try:
            query_funcs = Query(self.language, query_funcs_str)
            func_captures = self.get_captures_from_query(query_funcs, tree.root_node)
        except Exception:
            func_captures = []

        all_funcs = {}
        solidity_keywords = {
            "msg", "sender", "value", "require", "assert", "revert", "block", "timestamp", "now",
            "tx", "origin", "address", "uint256", "uint", "bool", "string", "memory", "storage",
            "calldata", "true", "false", "this", "balance", "transfer", "send", "call", "length",
            "push", "return", "returns",
        }

        func_identifiers = {}
        query_idents = Query(self.language, "(identifier) @ident")

        for node, _ in func_captures:
            name = ""
            for child in node.children:
                if child.type in ["identifier", "name"]:
                    name = child.text.decode("utf-8", errors="ignore")
                    break
            if name:
                all_funcs[name] = node
                idents = set()
                for ident_node, _ in self.get_captures_from_query(query_idents, node):
                    ident_name = ident_node.text.decode("utf-8", errors="ignore")
                    if ident_name not in solidity_keywords and ident_name != name:
                        idents.add(ident_name)
                func_identifiers[name] = idents

        if not all_funcs:
            return []

        graph = {name: set() for name in all_funcs.keys()}
        query_calls = Query(self.language, "(identifier) @call")
        for name, node in all_funcs.items():
            for c_node, _ in self.get_captures_from_query(query_calls, node):
                c_name = c_node.text.decode("utf-8", errors="ignore")
                if c_name in all_funcs and c_name != name:
                    graph[name].add(c_name)
                    graph[c_name].add(name)

        func_names = list(all_funcs.keys())
        for i in range(len(func_names)):
            for j in range(i + 1, len(func_names)):
                f1, f2 = func_names[i], func_names[j]
                shared_vars = func_identifiers[f1].intersection(func_identifiers[f2])
                if shared_vars:
                    graph[f1].add(f2)
                    graph[f2].add(f1)

        visited = set()
        components = []
        for name in all_funcs:
            if name not in visited:
                comp = set()
                queue = [name]
                visited.add(name)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        if not components:
            return []
        main_component = max(components, key=len)
        core_funcs = {"constructor", "fallback", "receive", "main"}
        for comp in components:
            if comp.intersection(core_funcs):
                main_component = comp
                break

        decoys_found = []
        for comp in components:
            if comp != main_component:
                for name in comp:
                    if name.startswith("_dead_") or len(comp) <= 2:
                        node = all_funcs[name]
                        decoys_found.append({"name": name, "span": (node.start_byte, node.end_byte)})
        return decoys_found

    def extract_threshold_features(self, code: str) -> dict[str, Any]:
        features = {
            "s1_hard": False,
            "s1_max_word": 0,
            "s1_spec_string": 0.0,
            "s1_spec_identifier": 0.0,
            "s1_spec_comment": 0.0,
            "s1_spec_error": 0.0,
            "s1_non_ascii": 0.0,
        }
        if not code:
            return features

        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception:
            return features

        for hit in self._detect_mixed_language_error(tree, code_bytes):
            if "Mixed_Language" in hit["type"] or "High_ERROR_Node_Ratio" in hit["type"]:
                features["s1_hard"] = True

        if self._detect_dead_decoys(tree, code_bytes):
            features["s1_hard"] = True

        queries = [self.string_query, self.comment_query, self.identifier_query, self.error_query]
        for query in queries:
            for node, cap in self.get_captures_from_query(query, tree.root_node):
                text = node.text.decode("utf8", errors="ignore")
                kind = node_kind(node.type, cap)

                for pattern in self.string_patterns.values():
                    if pattern.search(text):
                        features["s1_hard"] = True
                        break

                if kind == "identifier":
                    hit, _ = self._identifier_anomaly(text)
                    if hit:
                        features["s1_hard"] = True

                if len(text) >= 15:
                    if kind not in {"string", "comment"}:
                        features["s1_max_word"] = max(
                            features["s1_max_word"],
                            max((len(w) for w in text.split()), default=0),
                        )

                    ratio = adjusted_special_ratio(text, kind)
                    key = f"s1_spec_{kind}"
                    if key in features and not (kind == "string" and is_normal_c_format_string(text, self.lang_name)):
                        features[key] = max(features[key], ratio)

                    non_ascii_count = sum(1 for c in text if ord(c) > 127)
                    if non_ascii_count > 5 and kind not in {"comment", "string"}:
                        features["s1_non_ascii"] = max(features["s1_non_ascii"], non_ascii_count / max(1, len(text)))

        return features

    def detect(self, code):
        if not code:
            return False, code, []
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except Exception:
            return False, code, []

        triggered = False
        debug_info = []
        nodes_to_scan = []

        for query in [self.string_query, self.comment_query, self.identifier_query, self.error_query]:
            for node, cap in self.get_captures_from_query(query, tree.root_node):
                nodes_to_scan.append((node, cap))

        for hit in self._detect_mixed_language_error(tree, code_bytes):
            triggered = True
            debug_info.append(
                {
                    "layer": "Stage_I_AST",
                    "type": hit["type"],
                    "matched_text": hit.get("matched_text", "mixed-language/parser-error"),
                    "span": hit["span"],
                    "kind": "error",
                }
            )

        for node, cap in nodes_to_scan:
            text = node.text.decode("utf8", errors="ignore")
            node_type = node.type
            kind = node_kind(node_type, cap)
            matched_regex = False
            for attack_type, pattern in self.string_patterns.items():
                if pattern.search(text):
                    triggered = True
                    matched_regex = True
                    debug_info.append(
                        {
                            "layer": "Stage_I_AST",
                            "type": f"Regex_Match_{attack_type}",
                            "matched_text": text[:80].replace("\n", " "),
                            "span": (node.start_byte, node.end_byte),
                            "kind": kind,
                        }
                    )
                    break
            if matched_regex:
                continue

            is_anomalous, anomaly_type = self._check_structural_anomaly(text, node_type, cap)
            if is_anomalous:
                triggered = True
                debug_info.append(
                    {
                        "layer": "Stage_I_AST",
                        "type": f"Anomaly_{anomaly_type}_{kind}",
                        "matched_text": text[:80].replace("\n", " "),
                        "span": (node.start_byte, node.end_byte),
                        "kind": kind,
                    }
                )

        decoys = self._detect_dead_decoys(tree, code_bytes)
        if decoys:
            triggered = True
            for d in decoys:
                debug_info.append(
                    {
                        "layer": "Stage_I_AST",
                        "type": "Flashboom_Decoy_Detected",
                        "matched_text": f"Function: {d['name']}",
                        "span": d["span"],
                        "kind": "identifier",
                    }
                )

        repaired_code = code
        if triggered:
            # Map spans to their corresponding node types to apply smart, neutral replacements
            span_to_kind = {}
            for d in debug_info:
                if "span" in d and "kind" in d:
                    span_to_kind[d["span"]] = d["kind"]

            replacements = sorted({(d["span"][0], d["span"][1]) for d in debug_info if "span" in d}, key=lambda x: x[0], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for start, end in replacements:
                if 0 <= start < end <= len(new_code_bytes):
                    kind = span_to_kind.get((start, end), "other")
                    if kind == "comment":
                        rep_bytes = b""
                    elif kind == "string":
                        rep_bytes = b'""'
                    elif kind == "identifier":
                        rep_bytes = b"tmp_var"
                    else:
                        rep_bytes = b"None" if self.lang_name == "python" else b"0"
                    new_code_bytes[start:end] = rep_bytes
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")
        return triggered, repaired_code, debug_info