---
title: "[논문 리뷰] Skillful joint probabilistic weather forecasting from marginals
  (Alet et al., 2025)"
date: 2026-06-30
---
## 요약

- FGN(Functional Generative Network)은 DeepMind의 확률 앙상블 기상 모델로, 2025년 11월 공개된 WeatherNext 2의 엔진이다.
- 핵심 아이디어: noise를 입·출력이 아니라 신경망 파라미터 공간에 conditional layer normalization을 통해, 저차원(32-dim) 전역 노이즈 하나로 주입한다.
- 학습은 각 격자점의 marginal CRPS만 최소화한다. 그런데 추론 시에는 공간적으로 일관된 joint 분포가 나온다. 이게 제목 그대로 "from marginals"의 의미이다.
- Diffusion 기반 GenCast가 매 step마다 여러 차례의 denoising forward pass를 도는 반면, FGN은 step당 single forward pass다. 그래서 모델이 더 큰데도(180M vs 57M) 8배 빠르다.
- 결과: GenCast와 ENS를 deterministic·probabilistic 지표 전반에서 추월, 열대 저기압 경로 예측에서 유의하게 우수(p<0.05).

---

## 1. 배경: 왜 새 확률 모델인가

기상 예측에서 확률 예보는 필수적이다. 대기 역학은 비선형이고 관측은 부분적이라, 아무리 좋은 모델도 단일 결정론적 예측으로는 이로 인한 불확실성을 고려하기 어렵다. 의사결정자는 가장 그럴듯한 시나리오뿐 아니라 위험한 꼬리 시나리오(극한 폭풍 등)까지 알아야 한다.

ML 확률 예측의 분수령은 GenCast였다. ECMWF의 운영 앙상블 ENS를 고해상도에서 처음으로 분명하게 추월한 ML 모델이다. 다만 GenCast는 diffusion 모델이라, 한 timestep을 만들기 위해 반복적 denoising을 돌아야 해서 다른 ML 모델 대비 느리고, dense gridded data 바깥으로 확장하기에 유연하지 않았다.

FGN은 더 빠르고 더 정확한 확률 모델을 같은 GNN+graph-transformer 백본 위에서 전혀 다른 생성 방식으로 앙상블을 만들어낸다.

---

## 2. 문제 설정

목표는 표준적이다. 직전 두 스텝을 조건으로 다음 스텝의 분포에서 샘플링하는 2차 마르코프 분해다.

$$  
p(X^{1:T} \mid X^0, X^{-1}) = \prod_{t=1}^{T} p\big(X^t \mid X^{t-2:t-1}\big)  
$$

각 상태 $X^t$ 는 6개 대기 변수 × 13개 기압면 + 6개 지표 변수를 0.25° 격자 위에 담는다. 6시간 timestep으로 15일 예측을 타깃한다. 앙상블은 이 autoregressive 샘플링을 여러 번 반복해 만든다.

여기까지는 GenCast와 동일하다. 차이는 "한 스텝의 분포에서 어떻게 샘플을 뽑느냐" 에 있다.

---

## 3. 핵심 아이디어: 불확실성을 파라미터 공간에 넣는다

FGN은 불확실성을 두 종류로 깔끔하게 분리한다.

### Epistemic uncertainty — deep ensemble

모델 자체에 대한 불확실성(유한한 데이터·불완전한 학습)이다. 이는 deep ensemble(혹은 random seed ensemble)로 처리한다. 독립적으로 초기화·학습한 모델 $J=4$ 개를 두고, 각 모델이 앙상블 멤버의 일부를 생성한다. 베이즈 사후예측분포

$$  
p(X^{1:T}\mid X^{\leq 0}, \mathcal{D}) = \int p(X^{1:T}\mid X^{\leq 0}, \mathcal{M}),p(\mathcal{M}\mid\mathcal{D}),d\mathcal{M}  
$$

를 4개 시드로 근사하는 셈이다.

### Aleatoric uncertainty — learned functional perturbation

진짜 새로운 부분은 여기다. 미해상(unresolved) 현상에서 오는 환원 불가능한 randomness를, FGN은 신경망 파라미터를 샘플링해서 표현한다.

$$  
X_i^t = f_{\theta_i^t}(X^{t-2:t-1}), \qquad \theta_i^t \sim \mathcal{P}_\mathcal{M}(\theta)  
$$

즉 앙상블 멤버 $i$ 마다, timestep $t$ 마다 *서로 다른 신경망*을 뽑아서 예측을 만든다. 이게 "Functional" Generative Network라는 이름의 출처다. 노이즈를 입력 필드나 출력 필드에 더하는 대신, 함수 그 자체를 확률변수로 본다.

왜 이게 더 나은가? Alet(2022)의 논거를 빌리면, 파라미터 공간의 변동은 입력·출력 공간의 변동보다 더 구조화된(structured) 변화를 만든다. 파라미터를 공유하는 레이어(예: 공간 차원에 걸쳐 같은 가중치를 쓰는 conv/GNN)에서는, 샘플링된 파라미터가 모든 공간 위치에 *일관되게* 재사용되기 때문이다. 이것이 뒤에서 다룰 "marginal → joint" 마법의 토대다. 수치 모델의 stochastic physics(파라미터를 흔들어 앙상블을 만드는 기법)와 직접적으로 닮아 있다.

### Reparameterization trick

파라미터 분포 $\mathcal{P}_\mathcal{M}(\theta)$ 는 잘 알려진 reparameterization trick으로 학습한다.

$$  
\theta = \theta^* + \Delta \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)  
$$

- $\theta^*$ : 평균 파라미터 — 결정론적 함수에 해당
- $\Delta$ : 노이즈를 파라미터 섭동으로 매핑하는 (구조화될 수 있는) 행렬 — 공분산을 통제
- 모델 $\mathcal{M} := {\theta^*, \Delta}$

추론 시에는 두 불확실성을 조합한다. 먼저 4개 모델 중 하나 $\mathcal{M}_j$ 를 뽑고(epistemic), 그 안에서 $\epsilon_i^t$ 를 뽑아(aleatoric)

$$  
\theta_i^t = \theta_j^* + \Delta_j \cdot \epsilon_i^t  
$$

를 만든다.

---

## 4. 어떻게 주입하나: Conditional Layer Norm + 32차원 전역 노이즈

이론은 "파라미터를 샘플링한다"지만, 180M 파라미터를 통째로 샘플링하진 않는다. FGN의 영리한 구현 선택은 conditional layer normalization을 perturbation 통로로 재활용하는 것이다.

GenCast에서 conditional norm은 diffusion noise level $\sigma$ 를 조건으로 주입하는 데 쓰였다. FGN은 같은 메커니즘을 다르게 쓴다.

> 전역 노이즈 벡터 $z \sim \mathcal{N}(0,1)^{32}$ 를 네트워크의 모든 conditional layer-norm 레이어에 흘려보낸다. 앙상블 멤버마다 다른 $z$ 를 뽑는 것이, 곧 앙상블의 분산을 만드는 유일한 원천이다.

이 설계의 두 가지 특징이 핵심이다.

1. 저차원 노이즈 — 단 32차원. 무한히 많은 자유도가 아니라, 빡빡한 병목.
2. 전역 적용 + 공간 공유 — 이 $z$ 가 모든 레이어에 동일하게 들어가고, conditional norm 파라미터는 공간 차원(mesh·grid 노드)에 걸쳐 공유된다.

두 특징이 합쳐지면, 모델은 공간적으로 제멋대로인 노이즈를 만들 수 없게 된다. 변동을 주려면 전 지구에 걸쳐 일관된 방식으로 줄 수밖에 없다. 바로 이 제약이 다음 절의 마법을 가능케 한다.

> 참고로 AIFS-CRPS(Lang et al., 2024)도 conditional norm으로 확률성을 주입하지만, *위치별(location-specific)* 조건을 써서 고주파 공간 성분을 더한다. FGN은 의도적으로 반대 방향, 즉 전역·저주파·일관성을 택했다.

---

## 5. 이 논문의 핵심: 왜 marginal로 학습했는데 joint가 나오는가

FGN은 fair CRPS(fCRPS) 를 손실로 쓴다.

$$  
\text{fCRPS}(x^{1:N}, y) := \frac{1}{N}\sum_n |x^n - y| - \frac{1}{2N(N-1)}\sum_{n,n'}|x^n - x^{n'}|  
$$

학습 시에는 $N=2$ 샘플만으로, 모든 위치·변수·기압면에 대해 평균한다. CRPS는 단변량(univariate) 분포에 대해서만 strictly proper scoring rule이다. 즉, FGN의 손실은 각 격자점의 *marginal* 분포만 제약한다.

여기서 자연스러운 의문이 생긴다.

> marginal만 맞추면, 공간 상관구조(joint)는 엉망이어도 손실이 0이 될 수 있지 않나? 각 픽셀이 독립적으로 옳은 분포를 내놓으면서 서로 아무 상관이 없는, 물리적으로 말이 안 되는 앙상블도 marginal 점수는 만점이다.

FGN의 답은 "우리 아키텍처는 그런 앙상블을 만들 수 없다" 이다. 유일한 randomness 원천이 32차원 전역 $z$ 이고, 그게 공간 공유된 conditional norm을 통해서만 들어가므로, 픽셀별 독립 노이즈는 *구조적으로 불가능*하다. 모델이 marginal CRPS를 낮추려고 변동을 만들면, 그 변동은 어쩔 수 없이 전 지구에 걸쳐 일관된 — 즉 그럴듯한 joint 구조를 가진 — 형태가 된다.

정리하면 이렇다.

$$  
\underbrace{\text{marginal-only 손실 (최적화 쉬움)}}*{\text{fCRPS}} ;+; \underbrace{\text{저차원·전역 파라미터 노이즈 (강한 inductive bias)}}*{\text{conditional LN, } z\in\mathbb{R}^{32}} ;\Longrightarrow; \text{skillful joint}  
$$

고차원에서 joint를 직접 학습하는 건 어렵고 비싸다. FGN은 그 어려운 목적함수를 쉬운 marginal 손실 + 영리한 inductive bias로 우회한다. joint 구조는 따로 학습한 게 아니라 제약에서 "공짜로" 떨어진다. 이게 논문 전체를 관통하는 한 방이다.

---

## 6. 아키텍처와 학습

### 백본

GenCast denoiser와 거의 동일한 구조다. lat/lon 격자를 GNN encoder로 6번 정련한 구면 정이십면체 mesh의 latent 공간으로 보내고, 그 노드 위에서 graph-transformer processor가 작동한 뒤, decoder로 되돌린다.


|  | GenCast | FGN |
| ---------------- | ------------------------ | --------------------------- |
| 생성 방식 | diffusion (반복 denoising) | single forward pass |
| 파라미터 | ~57M (전체) | ~180M (시드당) × 4 시드 |
| latent dim | 512 | 768 |
| processor layers | 16 | 24 |
| timestep | 12h | 6h |
| noise 주입 | diffusion $\sigma$ | 32-dim $z$ → conditional LN |
| 학습 손실 | diffusion (joint 직접) | marginal fCRPS |


### Autoregressive 학습 — 직전 글과의 연결

FGN은 먼저 single-step 손실로 학습한 뒤, 마지막 단계에서 AR rollout finetuning을 한다. 매 학습 스텝에서 모델이 여러 스텝을 autoregressive하게 굴리고, 손실을 전체 rollout에 평균하며, gradient를 rollout 전체로 역전파한다. rollout은 최대 8 스텝까지. FGN 규모(180M×4, 0.25°)에서 8 스텝이라는 숫자는, BPTT의 메모리 한계가 어디쯤인지를 보여주는 실측치이기도 하다. 흥미논문은 AR 학습이 도움은 되지만 필수는 아니라고 못 박는다 — AR 없이도 skillful joint가 나온다는 것. joint 구조의 공로가 AR rollout이 아니라 §5의 inductive bias에 있음을 강조하는 대목이다.

### 데이터

ERA5로 사전학습(2016년 단일 NWP 기반의 대규모 재분석), HRES-fc0로 미세조정(2016~현재 운영 NWP 분석, 실시간 가용). 2022년을 검증, 2023년을 테스트로 동결 평가했다.

---

## 7. 결과

- 종합 성능: deterministic·probabilistic 지표 벤치마크 전반에서 GenCast와 ENS를 포괄적으로 추월. 보정(calibration)은 기존 모델과 비슷하거나 더 낫고, 극한값 예측도 동등 이상.
- Joint 구조: 검토한 대부분의 경우에서 더 나은 공간 상관구조. (marginal로만 학습했다는 점을 생각하면 이게 논문의 자랑이다.)
- 열대 저기압 경로: 평균 궤적과 경로 확률 모두에서 유의하게 우수(p<0.05).
- 효율: 단일 TPU v5p에서 15일 예측 1개를 1분 미만에 생성하며, 앙상블 멤버는 병렬 생성. 더 큰 모델이고 6시간 step이라 프레임이 2배인데도 GenCast 대비 8배 빠르다. 이유는 단순하다 — diffusion의 반복 refinement 없이 step당 forward pass 한 번이면 끝나기 때문.
- WeatherNext 2 차원에서는 이전 WeatherNext를 변수·리드타임의 99.9% 에서 능가한다고 보고된다.

---

### 참고 문헌

- Alet, F., Price, I., El-Kadi, A., Masters, D., Markou, S., Andersson, T. R., Stott, J., Lam, R., Willson, M., Sanchez-Gonzalez, A., & Battaglia, P. (2025). *Skillful joint probabilistic weather forecasting from marginals.* arXiv:2506.10772. [https://arxiv.org/abs/2506.10772](https://arxiv.org/abs/2506.10772)
- Price, I. et al. (2025). *GenCast: Diffusion-based ensemble forecasting for medium-range weather.* Nature. — diffusion 기반 baseline
- Lang, S. et al. (2024). *AIFS-CRPS.* — conditional norm 기반 stochasticity(위치별 조건)
- Lakshminarayanan, B. et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS. — epistemic 처리
- Zamo, M., & Naveau, P. (2018). *Estimation of the Continuous Ranked Probability Score with limited information.* — fair CRPS
- Google DeepMind (2025). *WeatherNext 2.* [https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/)

