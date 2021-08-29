
``` python
# Input: English doc (M lines), Vietnamese doc (N lines):
# en_doc: list of M strings
# vi_doc: list of N strings

# First translate the two docs to the other language.
en2vi_doc = vietai_translate_model_translate_envi(en_doc)
vi2en_doc = vietai_translate_model_translate_vien(vi_doc)
# en2vi_doc: list of M strings
# vi2en_doc: list of N strings

# Next compute M x N BLEU scores:
M = len(en_doc)
N = len(vi_doc)
bleu_score = np.zero((M, N))

for m in range(M):
  en_line, en2vi_line = en_doc[m], en2vi_doc[m]
  
  for n in range(N):
    vi_line, vi2en_line = vi_doc[n], vi2en_doc[n]
    
    bleu_score[m, n] = bleu(en_line, vi2en_line) + 
                       bleu(vi_line, en2vi_line)

# Dynamic Programming for pair matching:
# define: F[m, n] is sum of bleu scores between all pairs in the best matching 
# between en_doc[:m] and vi_doc[:n], i.e.,
# F[m, n] = sum(bleu(p, q) for p, q in best_pairs(en_doc[:m], vi_doc[:n])
# then:
for m in range(M):
  for n in range(N):
    F[m, n] = max(F[m-1, n-1] + bleu_score[m, n],
                  F[m-1, n],
                  F[m, n-1])

# Backtracking:
p, q = M, N
best_pairs = []  

while p and q:
 if F[p, q] == F[p-1, q-1] + bleu_score[p, q]:
   best_pairs += [(p, q)]
   p, q = p-1, q-1
 elif F[m, n] == F[m-1, n]:
   p -= 1
 else:
   q -= 1

return best_pairs.reverse()
```

### 
