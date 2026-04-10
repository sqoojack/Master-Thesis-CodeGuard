# tree.py
# 策略樹 + Thompson Sampling (對應論文 Section 4.3)

import json
import random
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StrategyNode:
    """策略樹節點"""
    path: List[str]           # 策略序列，如 ["rename_vars", "opaque"]
    parent: Optional['StrategyNode'] = None
    children: List['StrategyNode'] = field(default_factory=list)
    
    # Thompson Sampling 參數 (Beta 分佈)
    alpha: float = 1.0        # 成功次數 (潛力高分)
    beta: float = 1.0         # 失敗次數 (潛力低分)
    
    # 統計
    total_phi: float = 0.0    # 累積潛力分數
    visit_count: int = 0      # 訪問次數
    phi_history: List[float] = field(default_factory=list)
    
    def get_path_str(self) -> str:
        """路徑字串表示"""
        return " → ".join(self.path)
    
    def sample(self, length_penalty: float = 0.9) -> float:
        """
        Thompson Sampling 採樣值
        論文公式: Beta(α, β) * length_penalty^len(path)
        """
        if self.visit_count == 0:
            return random.random() * (length_penalty ** len(self.path))
        
        # Beta 分佈採樣
        sample_value = random.betavariate(self.alpha, self.beta)
        
        # 長度懲罰 (路徑越長，基礎值越低)
        penalty = length_penalty ** len(self.path)
        
        return sample_value * penalty
    
    def update(self, phi_score: float, max_phi: float = 100) -> None:
        """
        根據潛力分數更新節點統計
        論文公式: alpha += norm_score, beta += (1 - norm_score)
        """
        self.phi_history.append(phi_score)
        self.total_phi += phi_score
        self.visit_count += 1
        
        # 正規化潛力分數到 [0, 1]
        norm_score = min(1.0, max(0.0, phi_score / max_phi))
        
        # 更新 Beta 分佈參數
        self.alpha += norm_score
        self.beta += (1.0 - norm_score)
    
    def get_avg_phi(self) -> float:
        """平均潛力分數"""
        if self.visit_count == 0:
            return 0.0
        return self.total_phi / self.visit_count
    
    def get_clade_potential(self) -> float:
        """
        論文公式 (5): Clade-Level CoT Potential
        CoTMP(π) = E[max(φ(c))] over all descendants
        """
        if not self.children:
            return self.get_avg_phi()
        
        # 計算所有後代的最大潛力
        child_max = max([c.get_clade_potential() for c in self.children])
        node_max = self.get_avg_phi()
        
        return max(node_max, child_max)
    
    def to_dict(self) -> dict:
        """轉換為字典 (用於儲存)"""
        return {
            "path": self.path,
            "alpha": self.alpha,
            "beta": self.beta,
            "visit_count": self.visit_count,
            "avg_phi": self.get_avg_phi(),
            "children": [c.to_dict() for c in self.children]
        }


class StrategyTree:
    """
    策略樹管理器
    對應論文 Section 4.3: Lineage-Based Potential Guided Strategy Tree
    """
    
    def __init__(self, max_depth: int = 5, length_penalty: float = 0.85):
        self.root = StrategyNode(path=[])
        self.max_depth = max_depth
        self.length_penalty = length_penalty
        self.node_map: Dict[str, StrategyNode] = {}  # path_str -> node
        
        # 註冊根節點
        self.node_map[""] = self.root
        
        # 可用策略列表 (延遲初始化)
        self._available_strategies = None
    
    def _get_available_strategies(self) -> List[str]:
        """獲取所有可用策略 (延遲載入，避免循環依賴)"""
        if self._available_strategies is None:
            try:
                from strategy import StrategyLibrary
                lib = StrategyLibrary()
                self._available_strategies = lib.list_all()
            except ImportError:
                # 如果 strategy 模組不存在，使用預設列表
                self._available_strategies = [
                    "rename_vars", "dead_code", "reorder",
                    "opaque", "closure", "redundant_loop", "expand_ternary",
                    "encode", "indirect", "shred", "split"
                ]
        return self._available_strategies
    
    def _get_path_key(self, path: List[str]) -> str:
        return "|".join(path)
    
    def get_node(self, path: List[str]) -> Optional[StrategyNode]:
        """根據路徑獲取節點"""
        key = self._get_path_key(path)
        return self.node_map.get(key)
    
    def add_node(self, path: List[str], phi_score: float = 0) -> StrategyNode:
        """添加新節點"""
        if not path:
            return self.root
        
        key = self._get_path_key(path)
        if key in self.node_map:
            node = self.node_map[key]
            if phi_score > 0:
                node.update(phi_score)
            return node
        
        # 檢查父節點是否存在
        parent_path = path[:-1]
        parent = self.get_node(parent_path)
        if parent is None:
            parent = self.add_node(parent_path)
        
        # 創建新節點
        node = StrategyNode(path=path.copy(), parent=parent)
        parent.children.append(node)
        self.node_map[key] = node
        
        if phi_score > 0:
            node.update(phi_score)
        
        return node
    
    def update_node(self, path: List[str], phi_score: float) -> None:
        """更新節點的潛力分數"""
        node = self.get_node(path)
        if node:
            node.update(phi_score)
        else:
            self.add_node(path, phi_score)
    
    def select_best_path(self, max_length: int = 4, exploration_rate: float = 0.4) -> List[str]:
        """
        使用 Thompson Sampling 選擇最佳路徑
        加入 exploration_rate: 以一定機率隨機探索未嘗試過的路徑
        """
        # 探索模式：隨機選擇一個未嘗試過的路徑
        if random.random() < exploration_rate:
            return self._random_path(max_length)
        
        # 開發模式：選擇最佳節點
        best_node = self._select_best_node(self.root, 0, max_length)
        if best_node and best_node.path:
            return best_node.path
        return self._random_path(max_length)  # 如果沒有找到，fallback 到隨機
    
    def _random_path(self, max_length: int = 4) -> List[str]:
        """隨機生成一條路徑 (用於探索)"""
        all_strategies = self._get_available_strategies()
        if not all_strategies:
            return []
        
        # 隨機長度 1 到 max_length
        length = random.randint(1, min(max_length, len(all_strategies)))
        
        # 隨機選擇策略 (不重複)
        return random.sample(all_strategies, length)
    
    def _select_best_node(self, node: StrategyNode, depth: int, max_depth: int) -> Optional[StrategyNode]:
        """遞迴選擇最佳節點"""
        if depth >= max_depth or not node.children:
            return node if node.path else None
        
        # 對子節點進行 Thompson Sampling 採樣
        best_child = None
        best_score = -1
        
        for child in node.children:
            score = child.sample(self.length_penalty)
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child:
            return self._select_best_node(best_child, depth + 1, max_depth)
        return node if node.path else None
    
    def select_top_k_paths(self, k: int = 3, max_length: int = 4) -> List[List[str]]:
        """選擇前 K 個最佳路徑 (用於多樣化探索)"""
        candidates = []
        
        def collect(node: StrategyNode, depth: int):
            if depth >= max_length:
                return
            if node.path:
                # 使用平均潛力分數作為評估
                score = node.get_avg_phi()
                candidates.append((score, node.path))
            for child in node.children:
                collect(child, depth + 1)
        
        collect(self.root, 0)
        
        # 按分數排序，取前 K 個
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [path for score, path in candidates[:k]]
    
    def add_branch(self, parent_path: List[str], new_strategies: List[str]) -> List[List[str]]:
        """
        在父節點下添加新分支
        回傳: 新添加的路徑列表
        """
        parent_node = self.get_node(parent_path)
        if parent_node is None:
            parent_node = self.add_node(parent_path)
        
        new_paths = []
        for strategy in new_strategies:
            new_path = parent_path + [strategy]
            new_node = self.add_node(new_path)
            new_paths.append(new_path)
        
        return new_paths
    
    def get_clade_paths(self, path: List[str]) -> List[List[str]]:
        """獲取某個節點的所有後代路徑"""
        node = self.get_node(path)
        if not node:
            return []
        
        paths = []
        def collect(n: StrategyNode):
            if n.path:
                paths.append(n.path)
            for child in n.children:
                collect(child)
        
        collect(node)
        return paths
    
    def get_best_clade(self) -> Tuple[List[str], float]:
        """
        獲取最佳譜系 (Clade)
        論文公式 (6): max E[max φ(c)]
        """
        best_clade = None
        best_potential = -1
        
        for node in self.node_map.values():
            if node.path:  # 非根節點
                potential = node.get_clade_potential()
                if potential > best_potential:
                    best_potential = potential
                    best_clade = node.path
        
        return best_clade or [], best_potential
    
    def prune(self, path: List[str]) -> None:
        """剪枝: 移除某個節點及其所有後代"""
        node = self.get_node(path)
        if not node or not node.parent:
            return
        
        # 從父節點移除
        if node in node.parent.children:
            node.parent.children.remove(node)
        
        # 從 node_map 移除
        to_remove = []
        for key, n in self.node_map.items():
            if key.startswith(self._get_path_key(path)):
                to_remove.append(key)
        for key in to_remove:
            del self.node_map[key]
    
    def initialize_with_all_single_strategies(self):
        """初始化：加入所有單一策略作為根節點的子節點"""
        all_strategies = self._get_available_strategies()
        for strategy in all_strategies:
            self.add_node([strategy])
    
    def to_dict(self) -> dict:
        """序列化整個樹"""
        return self.root.to_dict()
    
    def print_tree(self, node: StrategyNode = None, indent: int = 0):
        """列印策略樹 (除錯用)"""
        if node is None:
            node = self.root
            print("Strategy Tree:")
        
        if node.path:
            avg_phi = node.get_avg_phi()
            print(f"{'  ' * indent}[{node.get_path_str()}] avg_phi={avg_phi:.2f}, visits={node.visit_count}")
        
        for child in node.children:
            self.print_tree(child, indent + 1)


# ========== 測試 ==========
if __name__ == "__main__":
    # 建立策略樹
    tree = StrategyTree(max_depth=4)
    
    # 初始化所有單一策略
    tree.initialize_with_all_single_strategies()
    
    # 添加一些測試節點
    tree.add_node(["rename_vars", "opaque"], phi_score=30)
    tree.add_node(["rename_vars", "opaque", "encode"], phi_score=60)
    
    # 列印樹
    tree.print_tree()
    
    # 測試選擇 (包含探索)
    print(f"\n隨機探索路徑: {tree.select_best_path(exploration_rate=1.0)}")
    print(f"最佳路徑 (開發模式): {tree.select_best_path(exploration_rate=0.0)}")
    print(f"混合模式: {tree.select_best_path(exploration_rate=0.3)}")