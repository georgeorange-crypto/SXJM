"""throwaway: why does scheduler.select collapse to 1 channel?"""
import sys
sys.path.insert(0, r"D:\George\SX\Way4\src")
sys.path.insert(0, r"D:\George\SX")

from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.channels import ChannelScheduler, SchedulerMode

belief = BeliefState(n_channels=20)
cert = CertificateManager(n_channels=20)
sched = ChannelScheduler()

q = (0.0, 0.0)
# all 20 unknown, fresh
print("per-channel coverage_gain at (0,0):", cert.coverage_gain(1, q))
print("per-channel value:", sched.channel_value(1, q, belief, cert))

# what does select pick?
plan = sched.select(q, belief, cert, current_channel=1, mode=SchedulerMode.EARLY)
print(f"\nEARLY select at (0,0): {len(plan.channels)} channels: {plan.channels}")
print(f"  value={plan.value:.4f} dwell={plan.dwell_time_s:.1f} ratio={plan.ratio:.6f}")

# manually compute ratio for k=1..20
values = {c: sched.channel_value(c, q, belief, cert)[0] for c in range(1, 21)}
print(f"\nall values equal? {set(round(v,6) for v in values.values())}")
import math
for k in [1, 2, 5, 10, 20]:
    chans = list(range(1, k+1))
    num = sum(values[c] for c in chans)
    ns = k - 1 if 1 in chans else k  # current channel 1 in set
    dwell = 5.0 * k + 1.0 * ns
    print(f"  k={k:2d}: num={num:.4f} dwell={dwell:.1f} ratio={num/dwell:.6f}")
