import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime


class CoTReflector:
    def __init__(self, playbook_path="playbook.json"):
        self.playbook_path = playbook_path
        self.playbook = self._load_playbook()

    def _load_playbook(self):
        try:
            with open(self.playbook_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                "version": "2.0",
                "strategies": [], 
                "strategy_sequences": [], 
                "failed_patterns": []
            }

    def analyze(self, strategy_path: List[str], verify_result: Dict) -> Dict:
        """
        核心反思邏輯：對應論文 4.2 節與 4.3 節
        回傳 Dict，包含 feedback、suggestions、phi_score、evaded
        """
        # 1. 提取驗證訊號
        phi_score = verify_result.get("phi_score", 0)
        evaded = verify_result.get("evaded", False)
        potential = verify_result.get("potential", {})
        
        # 2. 計算並記錄策略潛力
        self._update_node_potential(strategy_path, phi_score, evaded)
        
        # 3. 診斷失敗原因並生成建議
        suggestions = []
        if not evaded:
            phase2 = verify_result.get("phase2", {})
            reasoning_chain = phase2.get("reasoning_chain", [])
            vulnerability_sequences = phase2.get("vulnerability_sequences", [])
            
            # 分析失敗原因
            failure_analysis = self._analyze_failure(
                strategy_path, phi_score, reasoning_chain, vulnerability_sequences
            )
            
            # 記錄失敗
            self._record_failure(strategy_path, failure_analysis)
            
            # 生成建議
            suggestions = self._generate_suggestions(strategy_path, phi_score, failure_analysis)
        
        # 4. 生成反饋文字
        feedback = self._generate_feedback(phi_score, evaded)
        
        # 5. 清理過舊記錄（保留最近 50 條）
        if len(self.playbook.get("failed_patterns", [])) > 50:
            self.playbook["failed_patterns"] = self.playbook["failed_patterns"][-50:]
        
        self._save_playbook()
        
        # 回傳 Dict（嚴絲合縫）
        return {
            "feedback": feedback,
            "suggestions": suggestions,
            "phi_score": phi_score,
            "evaded": evaded
        }

    def _update_node_potential(self, path: List[str], phi: float, success: bool):
        """更新策略路徑的 Thompson Sampling 權重"""
        sequences = self.playbook.setdefault("strategy_sequences", [])
        
        # 尋找現有路徑
        entry = None
        for s in sequences:
            if s.get("sequence") == path:
                entry = s
                break
        
        if not entry:
            entry = {
                "sequence": path.copy(),
                "attempts": 0,
                "successes": 0,
                "avg_phi": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            sequences.append(entry)
        
        entry["attempts"] += 1
        if success:
            entry["successes"] += 1
        
        # 計算移動平均潛力
        old_avg = entry["avg_phi"]
        old_attempts = entry["attempts"] - 1
        if old_attempts > 0:
            entry["avg_phi"] = (old_avg * old_attempts + phi) / entry["attempts"]
        else:
            entry["avg_phi"] = phi
        
        entry["last_updated"] = datetime.now().isoformat()

    def _analyze_failure(self, path: List[str], phi: float, 
                         reasoning_chain: List[str], 
                         vulnerability_sequences: List[List[str]]) -> Dict:
        """分析失敗原因，回傳結構化的分析結果"""
        analysis = {
            "path": path,
            "phi_score": phi,
            "timestamp": datetime.now().isoformat(),
            "detected_patterns": [],
            "severity": "low"
        }
        
        # 分析推理鏈中的關鍵詞
        if reasoning_chain:
            reasoning_str = " ".join(reasoning_chain).lower()
            
            if "render_template_string" in reasoning_str:
                analysis["detected_patterns"].append("template_function_detected")
            if "user_input" in reasoning_str or "parameter" in reasoning_str:
                analysis["detected_patterns"].append("input_source_detected")
            if "ssti" in reasoning_str or "template injection" in reasoning_str:
                analysis["detected_patterns"].append("ssti_detected")
            if "xss" in reasoning_str:
                analysis["detected_patterns"].append("xss_detected")
        
        # 分析漏洞類型多樣性（幻覺指標）
        if vulnerability_sequences:
            all_types = []
            for types in vulnerability_sequences:
                all_types.extend(types)
            if len(set(all_types)) > 2:
                analysis["detected_patterns"].append("hallucination_detected")
        
        # 設定嚴重程度
        if phi < 10:
            analysis["severity"] = "high"
        elif phi < 30:
            analysis["severity"] = "medium"
        else:
            analysis["severity"] = "low"
        
        return analysis

    def _record_failure(self, path: List[str], analysis: Dict):
        """記錄失敗模式"""
        failed_patterns = self.playbook.setdefault("failed_patterns", [])
        
        # 檢查是否已存在
        existing = None
        for fp in failed_patterns:
            if fp.get("path") == path:
                existing = fp
                break
        
        if existing:
            existing["count"] = existing.get("count", 0) + 1
            existing["last_seen"] = datetime.now().isoformat()
            existing["analysis"] = analysis
        else:
            failed_patterns.append({
                "path": path,
                "analysis": analysis,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            })

    def _generate_suggestions(self, path: List[str], phi: float, 
                              analysis: Dict) -> List[str]:
        """根據失敗原因生成具體建議"""
        suggestions = []
        detected = analysis.get("detected_patterns", [])
        
        if "template_function_detected" in detected:
            suggestions.append("嘗試使用 rename_func 策略隱藏 render_template_string")
        
        if "input_source_detected" in detected:
            suggestions.append("嘗試使用 shred 或 split 策略混淆數據流")
        
        if "ssti_detected" in detected or "xss_detected" in detected:
            suggestions.append("當前混淆不足以隱藏漏洞模式，嘗試 opaque 或 rename_func")
        
        if not suggestions and phi < 30:
            suggestions.append("策略完全無效，嘗試不同的策略組合")
        
        if 30 <= phi < 60:
            suggestions.append(f"策略部分有效 (phi={phi:.1f})，繼續深化或組合其他策略")
        
        return suggestions

    def _generate_feedback(self, phi: float, success: bool) -> str:
        """根據論文 4.2 節生成反思回饋文字"""
        if success:
            return f"SUCCESS: 策略成功繞過檢測！phi={phi:.1f}"
        if phi > 40:
            return f"GOOD: phi={phi:.1f} - 偵測器出現明顯推理不穩定，接近繞過"
        if phi > 20:
            return f"PARTIAL: phi={phi:.1f} - 偵測器出現幻覺或不穩定，繼續深化"
        return f"FAILED: phi={phi:.1f} - 策略無效，需要全新組合"

    def get_best_sequences(self, top_k: int = 3, min_attempts: int = 1) -> List[List[str]]:
        """獲取最佳策略序列"""
        sequences = self.playbook.get("strategy_sequences", [])
        filtered = [s for s in sequences if s.get("attempts", 0) >= min_attempts]
        sorted_seq = sorted(filtered, key=lambda x: x.get("avg_phi", 0), reverse=True)
        return [s["sequence"] for s in sorted_seq[:top_k]]

    def get_sequence_stats(self, path: List[str]) -> Optional[Dict]:
        """獲取特定策略序列的統計"""
        sequences = self.playbook.get("strategy_sequences", [])
        for s in sequences:
            if s.get("sequence") == path:
                return {
                    "avg_phi": s.get("avg_phi", 0),
                    "attempts": s.get("attempts", 0),
                    "successes": s.get("successes", 0)
                }
        return None

    def print_summary(self):
        """列印反思摘要"""
        print("\n" + "=" * 60)
        print("Reflector Summary")
        print("=" * 60)
        
        print("\n📊 Best Strategy Sequences:")
        for seq in self.get_best_sequences(5):
            stats = self.get_sequence_stats(seq)
            if stats:
                print(f"  {seq}: avg_phi={stats['avg_phi']:.1f}, attempts={stats['attempts']}")
        
        print("\n⚠️ Failed Patterns:")
        for fp in self.playbook.get("failed_patterns", [])[:5]:
            path = fp.get("path", [])
            analysis = fp.get("analysis", {})
            detected = analysis.get("detected_patterns", [])
            print(f"  {path}: {detected if detected else 'unknown'}")

    def _save_playbook(self):
        """儲存 playbook"""
        with open(self.playbook_path, 'w', encoding='utf-8') as f:
            json.dump(self.playbook, f, indent=2, ensure_ascii=False)