# config.py
import torch

class Config:
    # 根據論文的預設參數
    MODEL_NAME = "google/codegemma-2b"  # 論文使用的模型之一 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 優化參數
    TOP_K = 600             # Algorithm 1 中的候選池大小 預設400
    EVAL_BATCH_SIZE = 8    # 評估時的 Mini-batch 大小，避免 Logits 撐爆 VRAM
    PERTURBATION_LEN = 8   # 擾動長度 (不含 Keyword) 
    KEYWORD_LEN = 2         # Keyword 長度 
    MAX_LENGTH = 1024
    
    # 新增：交替優化參數
    ALTERNATING_EPOCHS = 3  # 交替優化的最大輪數 (Keyword <-> Perturbation)
    PERTURB_ITERATIONS = 8 # 每一輪中，擾動優化的迭代次數
    
    
    # Forward Reasoning Enhancement (FRE) 參數 
    FRE_R = 0.4             # Loss 加權係數
    FRE_H = 2               # 增強前 h 個 Token 的權重
    
    # 種子
    SEED = 100