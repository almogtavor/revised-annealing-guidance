### Auto-(\lambda) (inference-only)

Since clip(cos(v_c,\delta)+1/2, 0, 1) was stuck at 0.5, we'll udpate auto lambda to:
[
\delta_t = v_c - v_u
]

[
\lambda_t
=========

\frac{|\delta_t|}{|v_c|+|v_u|+\varepsilon}
\cdot
\operatorname{clip}!\left(\frac{1+\cos(v_u,v_c)}{2},,0,,1\right)
]

---

### Interpretation (one line)

* first term = **how strong the conditional correction is**,
* second term = **how aligned / safe it is**.

---

### Behavior

* small gap → low (\lambda_t)
* moderate + aligned → high (\lambda_t)
* large but conflicting → reduced (\lambda_t)

---

That’s it — no thresholds, no EMA, fully state-dependent, and directly matches your goal.
