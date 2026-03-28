"""
1.1. Gizli Katmansız YSA Eğitme Formülleri — OR Kapısı
========================================================

  ȳᵢᵗ = Σ wᵢⱼ xⱼᵗ              (net girdi)
  s(x) = 1 / (1 + e^(-cx+θ))    (sigmoid, c=1, θ=0)
  Eₜ   = ½ Σ (yᵢᵗ - s(ȳᵢᵗ))²   (hata)
  E²ᵢₜ = (yᵢᵗ - s(ȳᵢᵗ)) · s'(ȳᵢᵗ)
  s'(x) = s(x) · (1 - s(x))     (sigmoid türevi)

  ∂E/∂wᵢⱼ = -Σ (yᵢᵗ - s(ȳᵢᵗ)) · s'(ȳᵢᵗ) · xⱼᵗ

  wᵢⱼ(yeni) = wᵢⱼ(eski) + α · Σ E²ᵢₜ · xⱼᵗ

n = 2 (giriş nöron sayısı)
m = 1 (çıkış nöron sayısı)
p = 4 (eğitme veri sayısı)
"""

import math
import random

# ─── Sigmoid fonksiyonu: s(x) = 1 / (1 + e^(-cx+θ)) ─────────────────────────
# c = 1, θ = 0 alınmıştır (standart sigmoid)

c_param = 1     # sigmoid eğim parametresi
theta   = 0     # sigmoid eşik parametresi

def sigmoid(x):
    """s(x) = 1 / (1 + e^(-c*x + θ))"""
    return 1.0 / (1.0 + math.exp(-c_param * x + theta))

def sigmoid_turev(x):
    """s'(x) = s(x) · (1 - s(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)

# ─── OR kapısı doğruluk tablosu (eğitme veri kümesi, p=4) ────────────────────
#       xⱼᵗ (j=1..n)    yᵢᵗ (i=1..m)

egitim_verisi = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [1]),
]

n = 2   # giriş katmanındaki nöron sayısı
m = 1   # çıkış katmanındaki nöron sayısı
p = 4   # eğitme veri sayısı

# ─── Hiperparametreler ───────────────────────────────────────────────────────

alpha = 0.5         # α : öğrenme oranı
epoch_sayisi = 1000

# ─── Ağırlıklar: wᵢⱼ (i=1..m, j=1..n) ──────────────────────────────────────
# m=1 çıkış, n=2 giriş → w[0][0]=w₁₁, w[0][1]=w₁₂

random.seed(42)
w = [[random.uniform(-0.5, 0.5) for j in range(n)] for i in range(m)]

print("=" * 70)
print("  OR KAPISI — GİZLİ KATMANSIZ YSA EĞİTİMİ")
print("  Formüller: s(x) = 1/(1+e^(-cx+θ)),  s'(x) = s(x)·(1-s(x))")
print("=" * 70)
print(f"\nn = {n} (giriş nöron),  m = {m} (çıkış nöron),  p = {p} (veri sayısı)")
print(f"c = {c_param},  θ = {theta}")
print(f"α = {alpha},  epoch = {epoch_sayisi}")
print(f"\nBaşlangıç ağırlıkları:")
for i in range(m):
    for j in range(n):
        print(f"  w[{i+1}][{j+1}] = {w[i][j]:.5f}")

# ─── Eğitim döngüsü ──────────────────────────────────────────────────────────

for epoch in range(1, epoch_sayisi + 1):

    toplam_E = 0.0
    grad = [[0.0 for j in range(n)] for i in range(m)]

    for t in range(p):
        x_t = egitim_verisi[t][0]    
        y_t = egitim_verisi[t][1]   

        for i in range(m):
            y_bar = sum(w[i][j] * x_t[j] for j in range(n))

            s_out = sigmoid(y_bar)

            hata = y_t[i] - s_out
            toplam_E += 0.5 * hata ** 2

            s_turev = sigmoid_turev(y_bar)
            E2_it = hata * s_turev

            for j in range(n):
                grad[i][j] += E2_it * x_t[j]

  
    for i in range(m):
        for j in range(n):
            w[i][j] = w[i][j] + alpha * grad[i][j]

    if epoch == 1 or epoch % 100 == 0 or epoch == epoch_sayisi:
        w_str = "  ".join(f"w[1][{j+1}]={w[0][j]:.5f}" for j in range(n))
        print(f"Epoch {epoch:>5d}  |  E = {toplam_E:.8f}  |  {w_str}")

# ─── Eğitim sonrası sonuçlar ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EĞİTİM TAMAMLANDI — SONUÇLAR")
print("=" * 70)

print(f"\nSon ağırlıklar:")
for i in range(m):
    for j in range(n):
        print(f"  w[{i+1}][{j+1}] = {w[i][j]:.5f}")

print(f"\n{'x1':>4s} {'x2':>4s} {'Hedef':>7s} {'ȳ (net)':>10s} {'s(ȳ)':>12s} {'Yuvarlak':>10s} {'Sonuç':>7s}")
print("-" * 58)

for t in range(p):
    x_t = egitim_verisi[t][0]
    y_t = egitim_verisi[t][1]
    y_bar = sum(w[0][j] * x_t[j] for j in range(n))
    s_out = sigmoid(y_bar)
    yuvarlak = round(s_out)
    sonuc = "✓" if yuvarlak == y_t[0] else "✗"
    print(f"{x_t[0]:>4d} {x_t[1]:>4d} {y_t[0]:>7d} {y_bar:>10.5f} {s_out:>12.6f} {yuvarlak:>10d} {sonuc:>7s}")

# ─── Son epoch — adım adım hesap detayları ───────────────────────────────────

print("\n" + "=" * 70)
print("  SON EPOCH — ADIM ADIM HESAPLAR")
print("=" * 70)

for t in range(p):
    x_t = egitim_verisi[t][0]
    y_t = egitim_verisi[t][1]

    for i in range(m):
        y_bar = sum(w[0][j] * x_t[j] for j in range(n))
        s_out = sigmoid(y_bar)
        hata  = y_t[i] - s_out
        s_tur = sigmoid_turev(y_bar)
        E2_it = hata * s_tur

        print(f"\n--- t={t+1}: x=[{x_t[0]}, {x_t[1]}] → hedef={y_t[i]} ---")
        print(f"  ȳ  = Σ wᵢⱼ·xⱼ = {w[0][0]:.5f}×{x_t[0]} + {w[0][1]:.5f}×{x_t[1]} = {y_bar:.6f}")
        print(f"  s(ȳ)  = {s_out:.6f}")
        print(f"  hata  = y - s(ȳ) = {y_t[i]} - {s_out:.6f} = {hata:.6f}")
        print(f"  s'(ȳ) = s(ȳ)·(1-s(ȳ)) = {s_out:.6f} × {1-s_out:.6f} = {s_tur:.6f}")
        print(f"  E²ᵢₜ  = hata × s'(ȳ) = {hata:.6f} × {s_tur:.6f} = {E2_it:.6f}")
        for j in range(n):
            print(f"  E²ᵢₜ·x{j+1} = {E2_it:.6f} × {x_t[j]} = {E2_it * x_t[j]:.6f}")