import json
import re

def update_tex_with_ablation(json_path: str, tex_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    for row in data:
        name = row["Config"]
        if "Baseline" in name and not "Adaptive" in name and not "CoVe" in name:
            key = r"A \(Baseline 纯文本\)"
        elif "A2" in name or ("Baseline" in name and "Adaptive" in name):
            key = r"A2 \(Baseline \+ Adaptive\)"
        elif "A3" in name or ("Baseline" in name and "CoVe" in name):
            key = r"A3 \(Baseline \+ CoVe\)"
        elif "B_Hetero" in name or "B \+Hetero" in name:
            key = r"B \(\+Hetero 异构融合\)"
        elif "C_Adaptive" in name or "C \+Hetero \+ Adaptive" in name:
            key = r"C \(\+Hetero \+ Adaptive\)"
        elif "D_CoVe" in name or "D \(\+CoVe" in name:
            key = r"D \(\+CoVe 全功能系统\)"
        else:
            continue

        f1 = row["F1_Score"]
        nar = row["No_Answer_Rate_Percent"]
        
        pattern = rf"({key} & [^&]+ & [^&]+ & [^&]+ & )([0-9\.]+\\%)( & )([0-9\.]+)( & \[[^\]]+\] \\\\)"
        
        def repl(match):
            # Bold F1 if it's the max, but here we just replace values verbatim
            f1_str = str(f1)
            # If the original was bolded, let's keep it clean or just insert it
            # The pattern captures the & and values.
            return f"{match.group(1)}{nar}\\%{match.group(3)}{f1_str}{match.group(5)}"
            
        content = re.sub(pattern, repl, content)
        
    # Replace N=500 with N=7405 in the table captions/headers
    content = content.replace("N=500", "N=7405")
    content = content.replace("500条", "7405条")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Tex updated successfully with the latest true GPU data (N=7405)!")

if __name__ == "__main__":
    update_tex_with_ablation(
        "/root/snap/ZJU-RAG-Thesis-Turing/data/results/automated_ablation.json", 
        "/root/snap/ZJU-RAG-Thesis-Turing/paper/zjuthesis/body/undergraduate/final/2-body.tex"
    )
