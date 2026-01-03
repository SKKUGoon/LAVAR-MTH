# ACADEMIC NOTES 

요약: LAVAR-MTH 는 

1. AutoEncoder 인공신경망을 이용하여 비선형적으로 데이터 $x_t$에 대한 Latent Factor(잠재 변수) $z_t$를 찾아낸 후
2. $z_t$ 시계열을 선형 시계열 모델인 VAR($p$)로 매핑합니다. 
3. 이렇게 찾아낸 $x_t ~ z_t$ AutoEncoder를 다양한 다변량 예측 타겟(Multi Target Head, TH)에 접목시켜 Fitting, 예측값 도출

하는 방법으로 작동합니다. 

## 1) 모델: 무엇을 학습하고 어떻게 사용하는가

우리는 다음과 같은 다변량 시계열을 모델링합니다:

- 비-공급(비-타깃) 공변량 $x_t \in \mathbb{R}^{D_x}$
- 공급 타깃 $y_t \in \mathbb{R}^{D_y}$

LAVAR는 저차원 잠재 상태(latent state)를 도입합니다:

$
z_t \in \mathbb{R}^{k}
$

구현은 문제를 다음으로 분해합니다:

- **비선형 관측(관측치) 매핑** (오토인코더):

$
z_t = e_\theta(x_t), \qquad \hat{x}_t = d_\phi(z_t)
$

- **선형 잠재 시계열** (VAR($p$)):

$
\hat{z}_t
  = c + \sum_{i=1}^{p} A_i z_{t-i}
$

여기서 $A_i \in \mathbb{R}^{k \times k}$, $c \in \mathbb{R}^k$ 입니다.

### 코드로의 매핑

- 인코더/디코더는 MLP: `network.MLP`
- VAR 동역학은 결정론적: `dynamics.VARDynamics`
- 스테이지-1 전체 모델: `models.LAVAR`

## 2) “잠재 변수는 어떻게 찾는가?”

잠재 상태는 인코더 네트워크에 의해 **결정론적으로** 계산됩니다:

$
z_t = e_\theta(x_t)
$

학습(스테이지 1) 동안, 윈도우 내 각 시점은 **독립적으로** 인코딩됩니다

구현 디테일: 인코더는 $x$에서 정확히 0인 값들을 작은 $\varepsilon$로 치환합니다(다운스트림 연산에서 죽은 그래디언트(dead gradients)를 피하기 위함):

$
x \leftarrow \begin{cases}
\varepsilon & \text{if } x = 0 \\
x & \text{otherwise}
\end{cases}
$

`models.py`의 `LAVAR.replace_zeros()` 및 `LAVAR.encode()`를 참고하세요.

## 3) 스테이지 1 학습 목적 ( $e_\theta, d_\phi, A_{1:p}, c$ 학습)

### 트레이너가 사용하는 윈도잉(windowing)

데이터셋은 각 인덱스 $t$에 대해 다음을 제공합니다:

$
x_{\text{past}} = [x_{t-p}, \dots, x_t] \in \mathbb{R}^{(p+1)\times D_x}
$

$
x_{\text{future}} = [x_{t+1}, \dots, x_{t+H}] \in \mathbb{R}^{H\times D_x}
$

$
y_{\text{future}} = [y_{t+1}, \dots, y_{t+H}] \in \mathbb{R}^{H\times D_y}
$

`dataset.RollingXYDataset`를 참고하세요.

### 윈도우별(per-window) 계산

`x_{\text{past}}`의 각 스텝을 인코딩합니다:

$
z_{t-i} = e_\theta(x_{t-i}) \quad (i = 0,\dots,p)
$

잠재 VAR 예측은 과거 잠재들 $z_{t-p:t-1}$을 사용해 $z_t$를 예측합니다:

$
\hat{z}_t = c + \sum_{i=1}^{p} A_i z_{t-i}
$

재구성(reconstruction)은 $z_t$를 사용합니다:

$
\hat{x}_t = d_\phi(z_t)
$

### `train_stage1.py`의 손실 항

스테이지-1 목적함수는 평균제곱오차(MSE)의 가중합입니다:

- **재구성 손실**:

$
\mathcal{L}_{\text{recon}}
  = \| \hat{x}_t - x_t \|_2^2
$

- **1-스텝 Latent Factor VAR 손실**:

$
\mathcal{L}_{\text{dyn}}
  = \| \hat{z}_t - z_t \|_2^2
$

- **선택적 다중-스텝 잠재 감독** (`cfg.multi_step_latent_supervision=True`):
  - 과거 잠재들로부터 $H$ 스텝 앞까지 Latent Factor VAR Fitting 값을 롤아웃(roll out)합니다:

$
\hat{z}_{t+1:t+H} = \text{rollout}([z_{t-p+1},\dots,z_t], H)
$

  - 실제 미래 관측을 인코딩하여 타깃 미래 잠재를 얻습니다:

$
z_{t+1:t+H} = e_\theta(x_{t+1:t+H})
$

  - 차이를 패널티로 줍니다:

$
\mathcal{L}_{\text{ms}}
  = \| \hat{z}_{t+1:t+H} - z_{t+1:t+H} \|_2^2
$

전체적으로(정확한 가중치는 `train_stage1.py`와 일치):

$
\mathcal{L}_{\text{stage1}} = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{ms}} \;\; (\text{if enabled})
$

## 4) 스테이지 2: “공급(supplies)은 어떻게 찾는가?”

스테이지 2에서 공급은 다음 방식으로 예측됩니다:

1. 가장 최근 $p$개 공변량 스텝을 잠재 히스토리로 인코딩
2. 학습된 VAR로 미래 잠재를 롤아웃
3. 각 미래 잠재 $z_{t+h}$를 작은 “supply head”로 $y_{t+h}$에 매핑

구체적으로:

$
z_{t-p+1:t} = e_\theta(x_{t-p+1:t})
$

$
\hat{z}_{t+1:t+H} = \text{rollout}(z_{t-p+1:t}, H)
$

$
\hat{y}_{t+h} = s_\psi(\hat{z}_{t+h})
$

구현: `models.LAVARWithSupply.forward()`.

### 파라미터는 고정되는가?

스테이지 2에서는 **LAVAR 전체가 고정(frozen)** 되고, 공급 헤드만 학습됩니다.

`train_stage2.train_supply_head_indexed()`에서:

- `for p in model.lavar.parameters(): p.requires_grad = False`
- 옵티마이저는 `model.supply_head.parameters()`만으로 구성됩니다

따라서:

- **스테이지 1**: 인코더 + 디코더 + VAR 파라미터가 학습됨
- **스테이지 2**: 인코더 + 디코더 + VAR 파라미터는 고정; 공급 헤드만 학습됨

## 5) “여러 개의 공급 헤드(multiple supply heads)는 어떻게 만들었나?”

이 저장소는 공급 차원 $j \in \{1,\dots,D_y\}$을 **밀도(density)로 분할** 하는 방식으로 “multiple heads”를 구현합니다.

### 밀도 지표(metric)

훈련 윈도우에서 타깃별 0이 아닌 비율(nonzero rate)을 계산합니다:

$
r_j = \mathbb{E}[\mathbf{1}\{y_{t,j} > 0\}]
$

구현: `train_stage2.compute_nonzero_rate_from_loader()`.

### 버킷(buckets)

임계값 `dense_nonzero_rate_thr` 및 `ultra_nonzero_rate_thr`를 사용하여:

- dense: $r_j \ge \tau_{\text{dense}}$
- ultra-sparse: $r_j \le \tau_{\text{ultra}}$
- sparse: 그 외

구현: `train_stage2.split_indices_by_density()`.

### 이것이 코드에서 “multiple heads”가 되는 방식

학습 코드는 이후 서로 다른 head 타입으로 설정된 **세 개의 `LAVARWithSupply` 모델을 각각 따로** 학습하며, 각 모델은 자신에게 해당하는 차원 부분집합에 대해서만 학습합니다:

- **dense 버킷**: `head_type="delta_mse"`
- **sparse 버킷**: `head_type="delta_mse"`
- **ultra 버킷**: `head_type="zinb"`

각 호출은 `train_stage2.train_supply_head_indexed(...)`로 수행됩니다.

예측 결과는 `train_stage2.stitch_bucket_predictions(...)`를 사용해 원래의 $D_y$ 순서로 다시 결합할 수 있습니다.

## 6) 어떤 확률분포(probability distribution)를 사용했나?

잠재 모델(인코더/디코더 + VAR)은 **결정론적** 입니다(명시적인 잠재 노이즈 분포 없음).

공급에 대해서는 다음을 지원합니다:

- **Negative Binomial (NB)**: `network.SupplyHeadNB` + `losses.negative_binomial_nll`
- **Zero-Inflated Negative Binomial (ZINB)**: `network.SupplyHeadZINB` + `losses.zinb_nll`

또한 결정론적인 증분 회귀 헤드도 지원합니다:

- **Δy + MSE**: `network.SupplyHeadMSE` (적분/누적합 단계 포함)

### NB 파라미터화 (평균/분산)

NB 헤드는 다음을 예측합니다:

- $\mu > 0$ (mean)
- $\theta > 0$ (dispersion / “total_count”)

그리고 음의 로그우도(negative log-likelihood)는 다음을 사용합니다:

$
\log p(y \mid \mu, \theta) := \log \Gamma(y+\theta) - \log \Gamma(\theta) - \log \Gamma(y+1) + \theta \big(\log \theta - \log(\theta+\mu)\big) + y \big(\log \mu - \log(\theta+\mu)\big)
$

손실은 배치/시간/타깃에 대해 평균낸 $ -\mathbb{E}[\log p(\cdot)] $ 입니다.

### ZINB: 0에 대한 점질량(point-mass) + NB의 혼합(mixture)

ZINB 헤드는 다음을 예측합니다:

- $\pi \in (0,1)$: *구조적(structural)* 0일 확률
- $\mu>0, \theta>0$: NB 파라미터

$y=0$일 때:

$
p(y=0) = \pi + (1-\pi)\,p_{\text{NB}}(0\mid \mu,\theta)
$

$y>0$일 때:

$
p(y) = (1-\pi)\,p_{\text{NB}}(y\mid \mu,\theta)
$

구현은 다음을 사용합니다:

$
p_{\text{NB}}(0\mid \mu,\theta) = \left(\frac{\theta}{\theta+\mu}\right)^{\theta}
$

`losses.zinb_nll()`을 참고하세요.

### dense/sparse 버킷에서 사용하는 Δy (증분) 헤드

dense/sparse 타깃에 대해 이 저장소는 증분을 예측하는 결정론적 헤드를 사용합니다:

$
\Delta y_{t+1} = y_{t+1} - y_t,\quad
\Delta y_{t+h} = y_{t+h} - y_{t+h-1}
$

MSE로 학습합니다:

$
\mathcal{L}_{\Delta}
  = \| \widehat{\Delta y}_{t+1:t+H} - \Delta y_{t+1:t+H} \|_2^2
$

증분을 레벨(level) 예측으로 바꾸기 위해 누적합을 사용합니다:

$
\hat{y}_{t+h} = \max\left(0,\; y_t + \sum_{i=1}^{h} \widehat{\Delta y}_{t+i}\right)
$

구현: 적분/클램프는 `LAVARWithSupply.forward()`에, 델타 타깃 구성은 `train_stage2.train_supply_head_indexed()`에 있습니다.

## 7) 코드 포인터(요약)

- 스테이지 1 핵심 forward pass: `models.LAVAR.forward()`
- 잠재 롤아웃: `models.LAVAR.rollout_latent()`
- VAR 정의: `dynamics.VARDynamics`
- 공급 헤드들: `network.SupplyHeadMSE`, `network.SupplyHeadNB`, `network.SupplyHeadZINB`
- 공급 NLL들: `losses.negative_binomial_nll`, `losses.zinb_nll`
- 스테이지 1 트레이너: `train_stage1.stage1_train_lavar`
- 스테이지 2 밀도 분할 + 고정: `train_stage2.stage2_train_supply_density_split`


