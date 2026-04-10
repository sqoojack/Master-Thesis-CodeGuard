import re

class PreFilter:
    def __init__(self, parser, language, lang_name="c", s1_word=200, s1_str=0.5, s1_other=0.4, s1_ascii=0.15):
        self.parser = parser
        self.language = language
        self.lang_name = lang_name.lower()
        
        self.s1_word = s1_word
        self.s1_str = s1_str
        self.s1_other = s1_other
        self.s1_ascii = s1_ascii
        
        self.string_patterns = {
            "SQL_Injection": re.compile(
                r"(?i)\b(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+.+?\s+SET)\b|--\s*$|\bOR\s+1\s*=\s*1\b"
            ),
            "Shell_Injection": re.compile(
                r"(?i)(;\s*(rm\s+-rf|wget|curl|nc|bash|sh|perl|python)\b|\|\s*(bash|sh)|>\s*/dev/null)"
            ),
            "Path_Traversal": re.compile(
                r"(?i)(\.\./\.\./|\betc/passwd\b|\betc/shadow\b|%2e%2e%2f)"
            ),
            "Prompt_Template_Injection": re.compile(
                r"(?i)(\{\{.*?\}\}|\(\)\s*=>|<script>|javascript:|\[\'\$[A-Za-z]+)"
            )
        }
        
        self.string_query = self.language.query("(string_literal) @string")
        self.identifier_query = self.language.query("(identifier) @identifier")
        self.error_query = self.language.query("(ERROR) @error")

        if self.lang_name == "java":
            self.comment_query = self.language.query("(line_comment) @comment (block_comment) @comment")
        else:
            self.comment_query = self.language.query("(comment) @comment")

    def _check_structural_anomaly(self, text, node_type):
        if len(text) < 15:
            return False, None
            
        if node_type == 'comment':
            return False, None
            
        if node_type != 'string_literal':
            max_word_len = max((len(w) for w in text.split()), default=0)
            if max_word_len > self.s1_word:
                return True, f"Long_Continuous_String ({max_word_len})"
            
        special_chars = set("{}[]()=><$|\\\"'`~^")
        special_count = sum(1 for c in text if c in special_chars)
        special_ratio = special_count / len(text)
        
        threshold = self.s1_str if node_type == 'string_literal' else self.s1_other
        if special_ratio > threshold:
            return True, f"High_Special_Char_Ratio ({special_ratio:.2f})"
            
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if non_ascii_count > 5 and (non_ascii_count / len(text)) > self.s1_ascii:
            return True, "Abnormal_Non_ASCII_Ratio"
            
        return False, None
    
    def _detect_dead_decoys(self, tree, code_bytes):
        # 1. Define query for functions based on language
        if self.lang_name == "java":
            query_funcs_str = "(method_declaration) @func"
        elif self.lang_name == "solidity":
            query_funcs_str = "(function_definition) @func"
        else:
            query_funcs_str = "(function_definition) @func (method_declaration) @func"

        try:
            query_funcs = self.language.query(query_funcs_str)
            func_captures = query_funcs.captures(tree.root_node)
        except Exception:
            func_captures = []

        all_funcs = {}
        
        # Stopwords for shared identifier analysis to prevent false edges
        solidity_keywords = {
            "msg", "sender", "value", "require", "assert", "revert", 
            "block", "timestamp", "now", "tx", "origin", "address", 
            "uint256", "uint", "bool", "string", "memory", "storage", 
            "calldata", "true", "false", "this", "balance", "transfer", 
            "send", "call", "length", "push", "return", "returns"
        }

        # 2. Extract all functions and their local identifiers
        func_identifiers = {}
        query_idents = self.language.query("(identifier) @ident")

        for node, _ in func_captures:
            name = ""
            for child in node.children:
                if child.type in ["identifier", "name"]:
                    name = child.text.decode("utf-8", errors="ignore")
                    break
            
            if name:
                all_funcs[name] = node
                
                # Extract all identifiers inside the function body
                idents = set()
                for ident_node, _ in query_idents.captures(node):
                    ident_name = ident_node.text.decode("utf-8", errors="ignore")
                    if ident_name not in solidity_keywords and ident_name != name:
                        idents.add(ident_name)
                func_identifiers[name] = idents

        if not all_funcs:
            return []

        # 3. Build Adjacency List for the Interaction Graph
        # Edges are formed by either explicit function calls or shared state variables
        graph = {name: set() for name in all_funcs.keys()}
        
        # 3a. Add edges based on Call Graph
        query_calls = self.language.query("(identifier) @call")
        for name, node in all_funcs.items():
            for c_node, _ in query_calls.captures(node):
                c_name = c_node.text.decode("utf-8", errors="ignore")
                if c_name in all_funcs and c_name != name:
                    graph[name].add(c_name)
                    graph[c_name].add(name) # Undirected graph for connectivity

        # 3b. Add edges based on Shared State/Identifiers (State Coupling)
        func_names = list(all_funcs.keys())
        for i in range(len(func_names)):
            for j in range(i + 1, len(func_names)):
                f1, f2 = func_names[i], func_names[j]
                shared_vars = func_identifiers[f1].intersection(func_identifiers[f2])
                if shared_vars:
                    graph[f1].add(f2)
                    graph[f2].add(f1)

        # 4. Find Connected Components using BFS
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

        # 5. Identify the Main Component (usually the largest one or containing constructor)
        # Components disconnected from the main one are considered Dead Decoys
        main_component = max(components, key=len)
        
        # Heuristic check: Ensure known entry functions force a component to be "Main"
        core_funcs = {"constructor", "fallback", "receive"}
        for comp in components:
            if comp.intersection(core_funcs):
                main_component = comp
                break

        decoys_found = []
        for comp in components:
            if comp != main_component:
                for name in comp:
                    node = all_funcs[name]
                    decoys_found.append({
                        "name": name,
                        "span": (node.start_byte, node.end_byte)
                    })

        return decoys_found

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
            for node, _ in query.captures(tree.root_node):
                nodes_to_scan.append(node)

        for node in nodes_to_scan:
            text = node.text.decode("utf8", errors="ignore")
            node_type = node.type
            
            matched_regex = False
            for attack_type, pattern in self.string_patterns.items():
                if pattern.search(text):
                    triggered = True
                    matched_regex = True
                    debug_info.append({
                        "layer": "Stage_I_AST",
                        "type": f"Regex_Match_{attack_type}",
                        "matched_text": text[:50].replace('\n', ' '),
                        "span": (node.start_byte, node.end_byte)
                    })
                    break
            
            if matched_regex:
                continue

            is_anomalous, anomaly_type = self._check_structural_anomaly(text, node_type)
            if is_anomalous:
                triggered = True
                debug_info.append({
                    "layer": "Stage_I_AST",
                    "type": f"Anomaly_{anomaly_type}_{node_type}",
                    "matched_text": text[:50].replace('\n', ' '),
                    "span": (node.start_byte, node.end_byte)
                })

        decoys = self._detect_dead_decoys(tree, code_bytes)
        if decoys:
            triggered = True
            for d in decoys:
                debug_info.append({
                    "layer": "Stage_I_AST",
                    "type": "Flashboom_Decoy_Detected",
                    "matched_text": f"Function: {d['name']}",
                    "span": d['span']
                })

        repaired_code = code
        if triggered:
            replacements = sorted({(d["span"][0], d["span"][1]) for d in debug_info}, key=lambda x: x[0], reverse=True)
            new_code_bytes = bytearray(code_bytes)
            for start, end in replacements:
                new_code_bytes[start:end] = b"[FILTERED_BY_STAGE_1]"
            repaired_code = new_code_bytes.decode("utf8", errors="ignore")

        return triggered, repaired_code, debug_info