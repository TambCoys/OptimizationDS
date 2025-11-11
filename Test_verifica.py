# verifichiamo se generate_problem è corretto
from generate_datas import generate_problem

for n, K in [(10, 3), (100, 5), (5000, 50)]:
    Q, q, blocks = generate_problem(n, K, seed=42)
    print(f"n={n}, K={K}")
    print(f"  blocks: {len(blocks)} gruppi")
    print(f"  somme: {[len(b) for b in blocks]}")
    print(f"  totale indici: {sum(len(b) for b in blocks)}")
    print(f"  indici unici: {len(set(i for b in blocks for i in b))}")
    assert sum(len(b) for b in blocks) == n
    assert len(set(i for b in blocks for i in b)) == n
    print("  PASSATO\n")
