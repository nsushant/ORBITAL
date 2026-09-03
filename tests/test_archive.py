"""The archive must hold exactly the feasible non-dominated set, and stay in step
with the schedules it stores -- the depot study needs to know who each front
point serves, not only what it costs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.archive import Archive, dominates

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok: fails.append(name)

print("1. dominance is minimisation on all three")
check("strictly better everywhere dominates",
      dominates(np.array([1.,1.,1.]), np.array([2.,2.,2.])))
check("equal but better on one dominates",
      dominates(np.array([1.,2.,2.]), np.array([2.,2.,2.])))
check("identical does not dominate",
      not dominates(np.array([1.,1.,1.]), np.array([1.,1.,1.])))
check("a trade-off does not dominate",
      not dominates(np.array([1.,3.,2.]), np.array([3.,1.,2.])))

print("\n2. the front keeps trade-offs and evicts the dominated")
A = Archive()
check("first entry is accepted", A.add([10., 3., 5e6], "s1"))
check("a dominated point is rejected", not A.add([12., 4., 6e6], "s2"))
check("  front still holds one", len(A) == 1)
check("a trade-off is accepted", A.add([5., 8., 5e6], "s3"))
check("  front holds two", len(A) == 2)
check("a point dominating both evicts them", A.add([1., 1., 1e6], "s4"))
check("  front collapses to one", len(A) == 1, f"n = {len(A)}")
check("  and it is the dominating one", A.objectives[0, 0] == 1.0)
check("  schedules stayed in step", A.schedules == ["s4"], f"{A.schedules}")

print("\n3. duplicates do not accumulate")
B = Archive()
B.add([1., 2., 3.], "a")
check("an exact duplicate is refused", not B.add([1., 2., 3.], "b"))
check("  front still holds one", len(B) == 1)

print("\n4. eviction keeps objectives and schedules aligned under churn")
rng = np.random.default_rng(0)
C = Archive()
for k in range(400):
    f = rng.random(3) * 100
    C.add(f, f"sched{k}")
check("lengths agree after 400 offers", len(C.schedules) == len(C),
      f"{len(C.schedules)} vs {len(C)}")
f = C.objectives
n_dominated = sum(1 for i in range(len(C)) for j in range(len(C))
                  if i != j and dominates(f[j], f[i]))
check("nothing in the front is dominated by the front", n_dominated == 0)

print("\n5. the two front points the study reads out")
D = Archive()
D.add([100., 1., 9e6], "cheap-few")       # low dV, few vehicles, serves little
D.add([900., 9., 1e6], "rich")            # expensive, many vehicles, serves most
D.add([400., 4., 4e6], "middle")
check("max coverage is the least unrecovered value",
      D.schedules[D.max_coverage()] == "rich")
k = D.knee(nadir=[1000., 10., 1e7])
check("knee is the compromise, not an extreme",
      D.schedules[k] == "middle", f"got {D.schedules[k]}")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
