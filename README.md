# CodeGuard: A Dual-Guardrail Defense Against Indirect Prompt Injection in External Code

本專案為碩士論文「CodeGuard: A Dual-Guardrail Defense Against Indirect Prompt Injection in External Code」的實作程式碼。本研究提出一個雙重護欄（Dual-Guardrail）防禦框架，旨在保護大型語言模型（LLMs）免受隱藏於外部程式碼中的間接提示注入（Indirect Prompt Injection, IPI）攻擊。

## 專案架構

專案主要分為資料集、攻擊實作、防禦機制與評估模組四大部分：

- **`Dataset/`**: 包含用於測試與評估的各類攻擊資料集。
  - `Adaptive_attack/`: 適應性攻擊資料（Contextual, Copy Trigger, Decoys）。
  - `ShadowCode/`: 隱蔽程式碼注入攻擊資料。
  - `Syntactic/`: 語法結構攻擊資料。
  - `XOXO_attack/`: 語意注入攻擊資料。
  - `merged_all/`: 合併後的完整資料集。

- **`main_code/attack/`**: 實作並重現多種針對程式碼模型的注入與擾動攻擊手法。
  - `Adaptive_attack/`: 適應性提示注入攻擊實作。
  - `Adversarial_attack/` (INSEC): 基於符號與字元的對抗性擾動攻擊。
  - `Semantic_attack/`: 語意層級攻擊（包含 ITGen 與 XOXO Attack）。
  - `Syntactic_Structure_attack/`: 程式碼語法結構改變攻擊。

- **`main_code/defense/` & `main_code/defense_v2/`**: CodeGuard 防禦框架的核心實作。
  - `pre_filter.py`: 進入護欄前的初步特徵過濾機制。
  - `Adversarial_Guardrail.py`: 對抗性護欄，負責偵測異常的語法結構、對抗性擾動或不自然的變數命名。
  - `Semantic_Guardrail.py`: 語意護欄，負責檢測程式碼片段中是否潛藏惡意指令或異常語意（如試圖越權的載荷）。
  - `dynamic_threshold.py`: 動態閾值模組，根據輸入特徵自適應調整觸發防禦的敏感度與風險評分。
  - `main.py`: 防禦框架的主程式入口。

- **`main_code/evaluate/`**: 實驗評估腳本。
  - 包含 `Inference_XOXO.py` 與 `Inference_shadowcode.py` 等腳本，用於推論並量測攻擊成功率（ASR, Attack Success Rate）與防禦有效性。

## 核心防禦機制：CodeGuard

CodeGuard 採用雙層過濾架構來確保外部程式碼的安全性：
1. **Adversarial Guardrail (對抗性護欄)**: 專注於程式碼的表面特徵與抽象語法樹（AST）結構，識別不自然的替換（如 INSEC 攻擊）或刻意安插的語法混淆。
2. **Semantic Guardrail (語意護欄)**: 深入分析程式碼上下文，攔截試圖覆寫系統提示（System Prompt）、竊取資料或觸發特定惡意條件的語意載荷（如 XOXO 與 ShadowCode）。
3. **Dynamic Thresholding (動態閾值)**: 結合上述兩者的量測指標，動態判定該程式碼區塊的風險分數，以決定是否阻擋、清洗或放行該輸入。

## 環境設定與安裝

請確保環境中已安裝 Python 3.8 或以上版本。安裝所需的相依套件：

```
pip install -r requirements.txt
```

執行方式
1. 啟動 CodeGuard 防禦框架
進入 defense 目錄並執行主程式來進行防禦測試:

```
cd main_code/defense/
python main.py
```
2. 執行攻擊與評估
若要生成攻擊樣本或評估特定攻擊的成功率，可執行 evaluate 或 attack 中的對應腳本。例如，評估 ShadowCode 攻擊效果：

```
python main_code/evaluate/Inference_shadowcode.py
```
評估 XOXO 攻擊效果：

```
python main_code/evaluate/Inference_XOXO.py
```