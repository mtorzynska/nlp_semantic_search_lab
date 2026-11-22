### Setup ewaluacji (ground truth)

Przygotowanie korpusu oparte zostało na unikalnych tekstach odpowiedzi. Dataset dostarcza relację $N:1$ pytanie-odpowiedź, zatem przeprowadzona została deduplikacja dla wydajności i spójności metryk.

```
Query - N*Answer (ta sama odpowiedź) -> Answer - [unique_query_1, unique_query_2, ...]
```

Z puli $87\,599$ odpowiedzi wyodrębnione zostały $18\,891$ unikalne dokumenty.
Powiązanie queries z answers wykonano za pomocą `np.unique` z parametrem `return_inverse=True` (mapowanie).

Ground Truth = **identyczność tekstowa**.

Finalna forma: 1-D wektor odpowiedzi przypisanych dla zapytań:
- $\text{idx} = q$ (identyfikator zapytania $q \in \{0, \dots, |Q|-1\}$)
- $\text{target}[\text{idx}] = a_q$ (identyfikator unikalnego dokumentu stanowiącego poprawną odpowiedź)

Po deduplikacji jedna odpowiedź przypisywana jest średnio do $4{,}64$ pytań. Przy definicji istotności jednakowej z identycznością tekstową oraz unikalności dokumentów, na **KAŻDE zapytanie przypada dokładnie jeden dokument istotny** ($|R_q| = 1$).

Wynikiem tego jest fakt, że $\text{Precision}@k$, $\text{Recall}@k$ i $\text{F1}@k$ mają charakter binarny per zapytanie i można je wyliczać bezpośrednio z 1-D wektora targetów bez budowania macierzy $2\text{D}$. Co więcej, w tym przypadku zachodzi zachowanie:

$$\text{Recall}@k = \text{Hit Rate}@k$$

Progóg podobieństwa nie został zastosowany (similarity threshold), aby uniknąć błędnego koła — ocena modelu opierałaby się wówczas na kombinacji podobieństwa tych samych embeddingów, które poddajemy testowi, co mogłoby zakłamać wyniki. Binarna identyczność jest tańsza obliczeniowo i deterministyczna.

Ograniczeniem tego podejścia jest to, że każda parafraza poprawnej odpowiedzi jest traktowana jako błąd ($\text{miss}$). Zatem uzyskane wyniki (np. $\text{P}@1 = 0{,}659$) należy traktować jako dolną granicę (floor) realnej trafności semantycznej.

---

### Precision@k, Recall@k, F1@k

Dla danego zapytania $q$, niech $\text{Top-}k(q)$ oznacza zbiór $k$ dokumentów zwróconych przez retriever, a $R_q$ zbiór dokumentów istotnych ($|R_q| = 1$). Niech $\mathbb{I}(\cdot)$ oznacza funkcję wskaźnikową (dającą $1$, gdy warunek jest spełniony, oraz $0$ w przeciwnym razie).

#### 1. Precision@k
Stosunek liczby trafionych istotnych dokumentów w top-$k$ do liczby wszystkich zwróconych dokumentów ($k$). Metryka nie uwzględnia kolejności wyników wewnątrz okna.

$$\text{Precision}@k(q) = \frac{|\text{Top-}k(q) \cap R_q|}{k}$$

Dla pojedynczego zapytania przy $|R_q| = 1$:

$$\text{Precision}@k(q) = \begin{cases} \frac{1}{k}, & \text{gdy } \text{target}[q] \in \text{Top-}k(q) \\ 0, & \text{w przeciwnym razie} \end{cases}$$

Średnia dla całego zbioru zapytań $Q$:

$$\text{Precision}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{Precision}@k(q)$$

#### 2. Recall@k
Liczba trafionych istotnych dokumentów w top-$k$ podzielona przez całkowitą liczbę istotnych dokumentów dla zapytania ($|R_q|$).

$$\text{Recall}@k(q) = \frac{|\text{Top-}k(q) \cap R_q|}{|R_q|}$$

Ponieważ w naszym setupie $|R_q| = 1$, wzór upraszcza się do binarnego wskaźnika trafienia ($\text{Hit Rate}@k$):

$$\text{Recall}@k(q) = \mathbb{I}\Big(\text{target}[q] \in \text{Top-}k(q)\Big) \in \{0, 1\}$$

Średnia dla całego zbioru $Q$:

$$\text{Recall}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall}@k(q)$$

#### 3. F1@k
Średnia harmoniczna z $\text{Precision}@k$ oraz $\text{Recall}@k$:

$$\text{F1}@k(q) = 2 \cdot \frac{\text{Precision}@k(q) \cdot \text{Recall}@k(q)}{\text{Precision}@k(q) + \text{Recall}@k(q)}$$

Gdy występuje trafienie w top-$k$ ($\text{Recall}@k = 1$ oraz $\text{Precision}@k = \frac{1}{k}$), F1 upraszcza się do stałej wartości zależnej wyłącznie od $k$:

$$\text{F1}@k(q) = 2 \cdot \frac{\frac{1}{k} \cdot 1}{\frac{1}{k} + 1} = \frac{2}{k + 1}$$

Zapytania bez żadnych istotnych dokumentów ($0/0$) są pomijane przy obliczaniu średniej Precision i F1, ale wliczają $0$ do średniego Recall.

---

### MRR (Mean Reciprocal Rank)

Średnia z odwrotności pozycji ($\frac{1}{\text{rank}}$) pierwszego trafnego dokumentu. Zastosowano wersję okienkową ($\text{windowed}$, $k = 5$). Brak trafnego dokumentu w oknie $k$ daje wynik $0$.

Niech $\text{rank}(q)$ oznacza 1-indexed pozycję poprawnej odpowiedzi $\text{target}[q]$ w zwróconym rankingu.

$$\text{RR}@k(q) = \begin{cases} \frac{1}{\text{rank}(q)}, & \text{gdy } \text{target}[q] \in \text{Top-}k(q) \\ 0, & \text{w przeciwnym razie} \end{cases}$$

$$\text{MRR}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}@k(q)$$

---

### MAP (Mean Average Precision)
*(Nie zaimplementowana)*

Średnia z wartości Average Precision ($\text{AP}$) dla każdego zapytania. $\text{AP}$ to suma $\text{Precision}@j$ na pozycjach $j$, w których pojawia się relewantny dokument, podzielona przez liczbę wszystkich istotnych dokumentów:

$$\text{AP}@k(q) = \frac{1}{|R_q|} \sum_{j=1}^{k} \text{Precision}@j(q) \cdot \text{rel}_j(q)$$

gdzie $\text{rel}_j(q) \in \{0, 1\}$ oznacza binarną istotność dokumentu na pozycji $j$.

$$\text{MAP}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}@k(q)$$

Dla setupu z pojedynczą odpowiedzią ($|R_q| = 1$), jeśli jedyny poprawny dokument znajduje się na pozycji $r \le k$, to $\text{AP}@k(q) = \frac{1}{r}$, co sprawia, że przy $|R_q|=1$ metryka $\text{MAP}@k$ staje się tożsama z $\text{MRR}@k$.

---

### NDCG (Normalized Discounted Cumulative Gain)

Metryka uwzględnia kolejność i wspiera stopniowalną relewantność (w naszym przypadku stosujemy binarne $\text{rel}_i \in \{0, 1\}$).

Dyskontowany skumulowany zysk na pozycji $k$:

$$\text{DCG}@k(q) = \sum_{i=1}^{k} \frac{\text{rel}_i(q)}{\log_2(i + 1)}$$

Przy binarnej istotności i $|R_q| = 1$, jeśli trafienie nastąpi na pozycji $r \le k$:

$$\text{DCG}@k(q) = \frac{1}{\log_2(r + 1)}$$

Idealne DCG ($\text{IDCG}$), czyli wynik dla idealnego rankingu (trafienie na pozycji $r=1$):

$$\text{IDCG}@k(q) = \frac{1}{\log_2(1 + 1)} = 1{,}0$$

Unormowana wartość NDCG dla zapytania:

$$\text{NDCG}@k(q) = \frac{\text{DCG}@k(q)}{\text{IDCG}@k(q)}$$

Średnia NDCG dla całego zbioru $Q$:

$$\text{NDCG}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{NDCG}@k(q) \in [0, 1]$$

---

### Porównanie: indeks vs brak indeksu

W celu weryfikacji poprawności implementacji oraz oceny wydajności biblioteki FAISS, przeprowadzono porównanie wyszukiwania z wykorzystaniem `faiss.IndexFlatIP` (dokładny brute-force inner product) z implementacją bez indeksu, opartą na chunkowanym skanie za pomocą `torch.topk`.

W wariancie bez indeksu zastosowano podział na bloki o rozmiarze 4096, gdzie poszczególne macierze podobieństwa (o wymiarach $4096 \times 18\,891$) w typie `float32` zajmowały ok. 300 MB pamięci. Test przeprowadzono na pełnym zbiorze $87\,599$ zapytań dla parametru $top\_k = 5$.

**Wyniki analizy dokładności:**
*   Dla $100\%$ zapytań uzyskano identyczne zbiory wyników w top-5.
*   Maksymalna różnica w wartościach score'ów wyniosła ok. $10^{-5}$, co wynika z szumu numerycznego typowego dla obliczeń zmiennoprzecinkowych.
*   Metryki jakościowe okazały się identyczne do ostatnich cyfr: $P@5 = 0{,}176235$, $R@5 = 0{,}881174$, $MRR = 0{,}747568$.
*   Zgodność obu metod została potwierdzona asercjami w notebooku, co jest bezpośrednią konsekwencją konstrukcji `IndexFlatIP`, który realizuje pełny skan przestrzeni.

**Wydajność i przepustowość:**

| Metoda | Czas wykonania | Przepustowość |
| :--- | :---: | :---: |
| FAISS (`IndexFlatIP`) | $6{,}90\text{ s}$ | $\approx 12\,700\text{ q/s}$ |
| Chunkowany MatMul (`torch`) | $9{,}80\text{ s}$ | $\approx 8\,900\text{ q/s}$ |

Odnotowano przyspieszenie rzędu $1{,}42\times$ (z wahaniami między uruchomieniami w przedziale $1{,}35\text{--}1{,}52\times$). Różnica ta wynika z optymalizacji backendu (faiss-cpu vs matmul na CUDA), a nie z zastosowania algorytmów aproksymacyjnych.

**Wnioski:**
Przy korpusie liczącym $18\,891$ dokumentów optymalnym wyborem jest indeks `FlatIP`, co jest zgodne z wytycznymi *FAISS Wiki Guidelines* (dla "relatively few points" rekomendowany jest exhaustive search). Zastosowanie indeksów ANN (Approximate Nearest Neighbors) lub kwantyzacji w tym przypadku nie przyniosłoby wymiernych korzyści wydajnościowych, a prowadziłoby do degradacji dokładności wyników. Realna wartość indeksów przybliżonych ujawnia się dopiero przy znacznie większych korpusach lub rygorystycznym budżecie pamięciowym.