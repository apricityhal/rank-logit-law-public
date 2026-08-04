import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The one-head causal-profile lower bound

This file formalizes the rank-one specialization of the row-subset determinant
bound in `main_paper.tex` and `published_appendix.tex`.

There are `N + 1` tokens and `N` causal boundaries.
`CenteredRankAtMostOne S` states that the score matrix has rank at most one
modulo arbitrary row shifts, which is equivalent over `ℝ` to centered score
rank at most one.  The causal difference profile retains exactly the diagonal
suppression and strict-lower triangle flatness used by the determinant proof.

For `N > 0`, the checked conclusion is

`γ ^ N ≤ 2 * B * δ ^ (N - 1)`.

The proof exposes the rank-one determinant identity directly: the product of
the diagonal first differences is the northeast corner first difference times
the product of the subdiagonal first differences.
-/

open scoped BigOperators

namespace C17
namespace OneHead

/-- A score matrix on `N + 1` tokens. -/
abbrev Score (N : ℕ) := Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ

/-- The arithmetic mean of one score row. -/
noncomputable def rowMean {N : ℕ} (S : Score N) (i : Fin (N + 1)) : ℝ :=
  (∑ j, S i j) / (N + 1 : ℝ)

/-- The score after subtracting its row mean. -/
noncomputable def centered {N : ℕ} (S : Score N) : Score N :=
  fun i j ↦ S i j - rowMean S i

/-- The manuscript's centered logit-radius condition `B(S) ≤ B`. -/
def CenteredRadiusAtMost {N : ℕ} (S : Score N) (B : ℝ) : Prop :=
  ∀ i j, |centered S i j| ≤ B

/-- Rank at most one after quotienting by row shifts.

The witness equation says `S = rowShift + query ⊗ key`.  Subtracting the row
mean kills `rowShift`, so this is the factorized form of centered rank at most
one used in the paper.
-/
def CenteredRankAtMostOne {N : ℕ} (S : Score N) : Prop :=
  ∃ rowShift query key : Fin (N + 1) → ℝ,
    ∀ i j, S i j = rowShift i + query i * key j

/-- The two local first-difference consequences of a causal
`(δ, γ)` profile.

The first clause is suppression across boundary `i`.  The second clause is
prefix flatness: an earlier adjacent difference in a later boundary row has
absolute value at most `δ`.
-/
def CausalDifferenceProfile {N : ℕ} (S : Score N) (δ γ : ℝ) : Prop :=
  (∀ i : Fin N,
      γ ≤ S i.castSucc i.castSucc - S i.castSucc i.succ) ∧
  (∀ i j : Fin N, j < i →
      |S i.castSucc j.castSucc - S i.castSucc j.succ| ≤ δ)

lemma score_sub_score_eq_centered_sub {N : ℕ} (S : Score N)
    (i j k : Fin (N + 1)) :
    S i j - S i k = centered S i j - centered S i k := by
  simp only [centered]
  ring

lemma scoreDifference_le_two_mul_of_centeredRadiusAtMost {N : ℕ}
    {S : Score N} {B : ℝ} (hB : CenteredRadiusAtMost S B)
    (i j k : Fin (N + 1)) :
    |S i j - S i k| ≤ 2 * B := by
  rw [score_sub_score_eq_centered_sub]
  calc
    |centered S i j - centered S i k|
        ≤ |centered S i j| + |centered S i k| := abs_sub _ _
    _ ≤ B + B := add_le_add (hB i j) (hB i k)
    _ = 2 * B := by ring

lemma scoreDifference_factor {N : ℕ} {S : Score N}
    {rowShift query key : Fin (N + 1) → ℝ}
    (h : ∀ i j, S i j = rowShift i + query i * key j)
    (i j k : Fin (N + 1)) :
    S i j - S i k = query i * (key j - key k) := by
  rw [h, h]
  ring

/-- The multiplicative identity behind the rank-one determinant bound. -/
lemma prod_diagonal_eq_corner_mul_subdiagonal {n : ℕ}
    (u v : Fin (n + 1) → ℝ) :
    (∏ i, u i * v i) =
      (u 0 * v (Fin.last n)) *
        ∏ i : Fin n, u i.succ * v i.castSucc := by
  rw [Finset.prod_mul_distrib, Fin.prod_univ_succ,
    Fin.prod_univ_castSucc, Finset.prod_mul_distrib]
  ring

lemma prod_abs_diagonal_eq_corner_mul_subdiagonal {n : ℕ}
    (u v : Fin (n + 1) → ℝ) :
    (∏ i, |u i * v i|) =
      |u 0 * v (Fin.last n)| *
        ∏ i : Fin n, |u i.succ * v i.castSucc| := by
  simpa only [abs_mul] using
    prod_diagonal_eq_corner_mul_subdiagonal
      (fun i ↦ |u i|) (fun i ↦ |v i|)

/-- One-head lower bound, indexed by one less than the number of causal
boundaries to keep every finite index inhabited.

This is exactly

`γ ^ N ≤ 2 * B * δ ^ (N - 1)`

with `N = n + 1`.
-/
theorem oneHeadLowerBoundSucc (n : ℕ) {S : Score (n + 1)}
    {δ γ B : ℝ} (_hδ : 0 ≤ δ) (hγ : 0 ≤ γ) (hB0 : 0 ≤ B)
    (hrank : CenteredRankAtMostOne S)
    (hprofile : CausalDifferenceProfile S δ γ)
    (hradius : CenteredRadiusAtMost S B) :
    γ ^ (n + 1) ≤ 2 * B * δ ^ n := by
  rcases hrank with ⟨rowShift, query, key, hscore⟩
  let u : Fin (n + 1) → ℝ := fun i ↦ query i.castSucc
  let v : Fin (n + 1) → ℝ :=
    fun j ↦ key j.castSucc - key j.succ

  have hfactor (i j : Fin (n + 1)) :
      S i.castSucc j.castSucc - S i.castSucc j.succ = u i * v j := by
    simpa [u, v] using
      scoreDifference_factor hscore i.castSucc j.castSucc j.succ

  have hdiag (i : Fin (n + 1)) : γ ≤ |u i * v i| := by
    have h := hprofile.1 i
    rw [hfactor i i] at h
    exact h.trans (le_abs_self (u i * v i))

  have hsubdiag (i : Fin n) : |u i.succ * v i.castSucc| ≤ δ := by
    have h := hprofile.2 i.succ i.castSucc i.castSucc_lt_succ
    simpa only [hfactor] using h

  have hcorner : |u 0 * v (Fin.last n)| ≤ 2 * B := by
    rw [← hfactor 0 (Fin.last n)]
    exact scoreDifference_le_two_mul_of_centeredRadiusAtMost
      hradius 0 (Fin.last n).castSucc (Fin.last n).succ

  have hdiagProduct :
      γ ^ (n + 1) ≤ ∏ i : Fin (n + 1), |u i * v i| := by
    rw [← Fin.prod_const]
    exact Finset.prod_le_prod (fun _ _ ↦ hγ) (fun i _ ↦ hdiag i)

  have hsubdiagProduct :
      (∏ i : Fin n, |u i.succ * v i.castSucc|) ≤ δ ^ n := by
    rw [← Fin.prod_const]
    exact Finset.prod_le_prod (fun _ _ ↦ abs_nonneg _) (fun i _ ↦ hsubdiag i)

  calc
    γ ^ (n + 1) ≤ ∏ i : Fin (n + 1), |u i * v i| := hdiagProduct
    _ = |u 0 * v (Fin.last n)| *
          ∏ i : Fin n, |u i.succ * v i.castSucc| :=
      prod_abs_diagonal_eq_corner_mul_subdiagonal u v
    _ ≤ (2 * B) * δ ^ n := by
      exact mul_le_mul hcorner hsubdiagProduct
        (Finset.prod_nonneg fun _ _ ↦ abs_nonneg _) (mul_nonneg (by norm_num) hB0)

/-- The same finite bound with the manuscript's boundary count `N`. -/
theorem oneHeadLowerBound {N : ℕ} (hN : 0 < N) {S : Score N}
    {δ γ B : ℝ} (hδ : 0 ≤ δ) (hγ : 0 ≤ γ) (hB0 : 0 ≤ B)
    (hrank : CenteredRankAtMostOne S)
    (hprofile : CausalDifferenceProfile S δ γ)
    (hradius : CenteredRadiusAtMost S B) :
    γ ^ N ≤ 2 * B * δ ^ (N - 1) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hN)
  exact oneHeadLowerBoundSucc n hδ hγ hB0 hrank hprofile hradius

end OneHead
end C17
