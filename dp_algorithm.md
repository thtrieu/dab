
``` python
# Input: English doc (M lines), Vietnamese doc (N lines):
# en_doc: list of M strings
# vi_doc: list of N strings

# First translate the two docs to the other language.
en2vi_doc = vietai_translate_model_translate_envi(en_doc)
vi2en_doc = vietai_translate_model_translate_vien(vi_doc)
# en2vi_doc: list of M strings
# vi2en_doc: list of N strings

scores = {}  # dict((m, n) -> float)

def match_score(m, n):
  """Returns how 'matched' en_doc[m] and vi_doc[n] is."""
  if (m, n) not in scores:
    en_line, en2vi_line = en_doc[m], en2vi_doc[m]
    vi_line, vi2en_line = vi_doc[n], vi2en_doc[n]
    bleu_score[(m, n)] = bleu(en_line, vi2en_line) + 
                         bleu(vi_line, en2vi_line)

  return scores[(m, n)]
    

# Dynamic Programming for pair matching:
# define: F[m, n] is sum of match scores between all pairs in the best matching 
# between en_doc[:m] and vi_doc[:n], i.e.,
# F[m, n] = sum(score(p, q) for p, q in best_pairs(en_doc[:m], vi_doc[:n])
# then:
for m in range(M):
  for n in range(N):
    match_m_n = match_score(m, n)
    if m == 0:
      F[m, n] = max(F[0, :n], match_m_n)
    elif n == 0:
      F[m, n] = max(F[:m, 0], match_m_n)
    else:
      F[m, n] = max(F[m-1, n-1] + match_m_n,
                    F[m-1, n],
                    F[m, n-1])


# Backtracking:
p, q = M-1, N-1  # python's last index of M-element and N-element lists.
best_pairs = []

while p >= 0 and q >= 0:
 if F[p, q] == F[p-1, q-1] + scores[p, q]:
   best_pairs += [(p, q)]
   p, q = p-1, q-1
 elif F[p, q] == F[p-1, q]:
   p -= 1
 else:
   q -= 1

return best_pairs.reverse()
```

### 
