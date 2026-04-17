import math
f1_scores = [7.62, 22.91, 6.98, 24.28, 26.40, 9.17, 19.43, 11.44, 20.38, 22.96]
for f1 in f1_scores:
    p = f1
    se = math.sqrt(p * (100 - p) / 500)
    lower = p - 1.96 * se
    upper = p + 1.96 * se
    print(f"[{lower:.2f}, {upper:.2f}]")
