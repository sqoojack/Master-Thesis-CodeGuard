import networkx as nx
import matplotlib.pyplot as plt

def draw_tree():
    # 建立一個有向圖
    G = nx.DiGraph()
    
    # 定義節點與連接關係 (父, 子)
    edges = [
        ("translation_unit", "declaration"),
        ("translation_unit", "comment: '// 這是註解'"),
        ("declaration", "primitive_type: 'int'"),
        ("declaration", "variable_declarator"),
        ("declaration", "punctuation: ';'"),
        ("variable_declarator", "identifier: 'score'"),
        ("variable_declarator", "operator: '='"),
        ("variable_declarator", "number_literal: '100'")
    ]
    
    G.add_edges_from(edges)
    
    # 使用層級布局
    pos = {
        "translation_unit": (0, 3),
        "declaration": (-1, 2),
        "comment: '// 這是註解'": (1, 2),
        "primitive_type: 'int'": (-2, 1),
        "variable_declarator": (-1, 1),
        "punctuation: ';'": (0, 1),
        "identifier: 'score'": (-2, 0),
        "operator: '='": (-1, 0),
        "number_literal: '100'": (0, 0)
    }
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", 
            font_size=10, font_weight="bold", arrows=True, node_shape="s")
    
    plt.title("Tree-sitter Syntax Tree Visualization")
    plt.savefig("tree_visualization.png")

draw_tree()