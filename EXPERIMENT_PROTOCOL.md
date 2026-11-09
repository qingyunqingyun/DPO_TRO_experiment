# DPO / TRO 实验纲要

版本：2026-11-09-v6-posterior-confidence  
状态：框架已整理；先验与首轮搜索网格为下文列出的建议值，尚未冻结或启动新方案实验。  
执行进度、实际取值和结果链接统一记录在 [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)。

本文件是后续协作的主纲要。框架负责人维护研究问题、模型和比较原则；实验执行者负责实现、训练、评测和如实回传。修改方法或参数范围时更新本文件及版本记录，具体运行绑定一个不可变的 Git commit。

## 1. 研究问题与两个实验部分

固定一个问题及其两个回答 $(x_i,y_{i1},y_{i2})$。真实偏好概率未知，实际只收集少量投票，且每题人数不同。我们检验：在保留经验偏好比例的同时，使用逐题后验可信程度控制鲁棒部分，是否能改善生成表现并减少相对 oracle 的性能短缺。

两部分实验都需要完成：

1. **统一人数增长实验**：所有题统一为 $N=1,5,11,51,101$，比较 empirical soft DPO 与 oracle soft DPO，观察生成表现与 oracle 的差距随人数如何变化。
2. **少量异质投票实验**：每题 $N_i\in\{1,3,5\}$，比较 hard DPO、empirical soft DPO、公共半径标签 DRO、posterior-confidence TRO 和 oracle soft DPO。

oracle 概率由冻结 RM 定义，是模拟环境的真值，不宣称等同真实人类偏好。oracle policy 是用精确概率训练的策略模型，不是 RM，也不是生成奖励的理论上界。结果允许不支持 TRO，不预设方法排名。

本轮只模拟独立二项投票的抽样误差，不额外翻转标签或污染 oracle 概率。更多票降低抽样误差，但不能据此声称能够纠正系统性偏差或相关标注。

## 2. 数据、模型和信息隔离

| 项目 | 继承的设置 |
|---|---|
| 回答对 | HuggingFaceH4/ultrafeedback_binarized 的现成回答对 |
| 策略 / 参考模型 | meta-llama/Llama-3.2-1B-Instruct；各方法同初始化，参考策略冻结 |
| 教师与评测 RM | Skywork/Skywork-Reward-V2-Llama-3.1-8B，同一冻结版本 |
| 首轮规模 | 2,000 个训练回答对、500 个开发问题；最终确认测试目标 500 个未用于探索的问题 |
| 主场景人数 | $\{1,3,5\}$ 尽量等量随机分配，独立于文本、RM 分数和真实概率 |
| 训练长度 | 2048 tokens：聊天模板、prompt 加一侧完整回答；两侧取最大值判断 |
| 训练预算 | 2 epochs；有效 batch 128 个回答对；learning rate $5\times10^{-7}$；DPO $\beta=0.01$ |
| 其余训练设置 | BF16、FP32 logprob/loss 累积、gradient checkpointing、ZeRO-2、AdamW；betas=(0.9,0.95)、weight decay=0、warmup=0.1、clip norm=1 |
| 生成评测 | 每题 10 条回答，temperature=0.7、top-p=0.9、max_new_tokens=1024、context=4096 |

训练与 RM 使用各自正确的模板，但应看到同样完整的文本。超过 2048 的训练回答对在冻结清单前统一过滤，不能用完整回答的标签训练截断前缀。资源不足先调整 microbatch 与梯度累积，保持有效 batch 和输入规范一致。

按 prompt 隔离 train/dev/test，并保存清单与 hash。旧 test_diagnostic 已用于探索，不能重新命名为新确认集；这不等于已发现训练集与测试集重叠。从未使用的合格 prompt 中预先保留确认集；不足时记录实际数量与处理办法，不悄悄复用旧测试问题。

### 2.1 oracle 与模拟投票

固定两个回答的顺序，RM 分别给分 $r_{i1},r_{i2}$：

$$
p_i^*=\sigma\!\left(\frac{r_{i1}-r_{i2}}{T_{\mathrm{pref}}}\right),\qquad
K_i\sim\mathrm{Binomial}(N_i,p_i^*),\qquad h_i=\frac{K_i}{N_i}.
$$

$T_{\mathrm{pref}}$ 使用最终训练清单非零绝对分差的中位数，一次冻结。旧归档的值为 9.2548828125，仅在 RM、模板、文本清单、原始分数及规则均不变时复用；不能直接沿用到改变后的数据或教师。

所有方法在同一重复内共享回答对、人数、实际投票和训练顺序规则。普通方法的标签与 $\lambda_i$ 只读取 $N_i,K_i,h_i$；$p_i^*$、RM 原始分数不进入普通方法训练权重。oracle trainer 使用 $p_i^*$，评测可使用匹配 oracle 的生成结果。先验不使用真实 $p_i^*$ 拟合。

外层每个回答对权重均为 $1/M$，不按投票人数加权。模拟投票只抽取选择，不重新生成回答；模型生成发生在评测阶段。

## 3. DPO、公共半径 DRO 与 TRO

令

$$
z_i(\theta)=\beta\left[
\log\frac{\pi_\theta(y_{i1}\mid x_i)}{\pi_{\mathrm{ref}}(y_{i1}\mid x_i)}
-\log\frac{\pi_\theta(y_{i2}\mid x_i)}{\pi_{\mathrm{ref}}(y_{i2}\mid x_i)}
\right].
$$

soft DPO 单题损失为

$$
\ell_i(\theta,p)=p\,\mathrm{softplus}(-z_i)+(1-p)\,\mathrm{softplus}(z_i)
=\mathrm{softplus}(z_i)-pz_i.
$$

公共半径的标签不确定集合：

$$
\mathcal U_i(\rho)=
\{p\in[0,1]:d_{\mathrm{KL}}(\mathrm{Ber}(h_i)\Vert\mathrm{Ber}(p))\le\rho\}.
$$

KL 方向固定为经验概率到候选概率，使用自然对数。每题中心不同，但所有题使用同一个 $\rho$；不设 $\rho_i$，也不对题目外层权重做对抗重加权。

$$
J_i^{\mathrm{DRO}}(\theta;\rho)=\max_{p\in\mathcal U_i(\rho)}\ell_i(\theta,p),
$$

$$
\boxed{
J^{\mathrm{TRO}}(\theta)=\frac1M\sum_i
\left[\lambda_i\ell_i(\theta,h_i)+(1-\lambda_i)J_i^{\mathrm{DRO}}(\theta;\rho)\right].
}
$$

$\lambda_i$ 始终表示保留经验项的可信权重；$\lambda_i=1$ 是 empirical soft DPO，$\lambda_i=0$ 是公共半径 DRO。$\rho=0$ 时两项相同，均退化为 empirical soft DPO。新方案不改变名义标签或球的中心：二者仍为 $h_i$，不是后验均值。

## 4. 新方案：后验可信概率作为信任权重

### 4.1 固定先验，按票数计算后验

给所有题使用同一个固定、对称的参考先验：

$$
p_i\sim\mathrm{Beta}(a_0,b_0),\qquad a_0=b_0>0.
$$

独立二项投票下：

$$
p_i\mid N_i,K_i\sim\mathrm{Beta}(a_i,b_i),\qquad
a_i=a_0+K_i,\quad b_i=b_0+N_i-K_i.
$$

主先验建议为 $a_0=b_0=0.5$，敏感性对照建议为 $a_0=b_0=1$。二者均是显式建模选择；Jeffreys 先验有 Fisher 信息下参数变换不变性的依据，但并非“没有假设”。每题只有 1～5 票时，先验影响可能明显，不能因为权重排序符合直觉就宣称校准正确。

### 4.2 可信事件、公式与含义

全局 $\varepsilon$ 定义“比例误差多大以内算足够准确”：

$$
G_i=\{|p_i-h_i|\le\varepsilon\},\qquad
\boxed{\lambda_i=\Pr(G_i\mid N_i,K_i).}
$$

设 $p_i^-=\max(0,h_i-\varepsilon)$、$p_i^+=\min(1,h_i+\varepsilon)$，则

$$
\boxed{
\lambda_i=
\frac{\displaystyle\int_{p_i^-}^{p_i^+}t^{a_i-1}(1-t)^{b_i-1}\,dt}
{\displaystyle\int_0^1t^{a_i-1}(1-t)^{b_i-1}\,dt}
=I_{p_i^+}(a_i,b_i)-I_{p_i^-}(a_i,b_i).
}
$$

$I_x(a,b)$ 是 Beta 累积分布函数。积分变量不是真实 oracle 值；计算只需要每题票数、全局先验和 $\varepsilon$，与 $\rho$、当前模型参数及训练 loss 无关。

例如 $h_i=0.6,\varepsilon=0.1$ 时，$\lambda_i$ 表示后验认为真实比例落在 $[0.5,0.7]$ 的概率。它不是“第一个回答获胜的概率”、不是“没有污染的概率”，也不是投票者正确率。

不再采用之前讨论过的 CP 区间权重或 $\min(1,\varepsilon/\sqrt{\text{后验MSE}})$。这些是不同候选定义，不能把数值或理论保证混到本版中。

### 4.3 数值参考与计算

下表使用建议先验 $\mathrm{Beta}(0.5,0.5)$，用于实现核验，不表示参数已经选优。反向票型具有相同权重。

| 人数 | 票型 | $\varepsilon=0.1$ | $\varepsilon=0.2$ | $\varepsilon=0.3$ |
|---:|---:|---:|---:|---:|
| 1 | 1:0 | 0.395819 | 0.549815 | 0.660746 |
| 3 | 2:1 | 0.316162 | 0.607254 | 0.837663 |
| 3 | 3:0 | 0.592916 | 0.772547 | 0.873130 |
| 5 | 3:2 | 0.373012 | 0.686868 | 0.894243 |
| 5 | 4:1 | 0.464404 | 0.815393 | 0.912287 |
| 5 | 5:0 | 0.707482 | 0.874533 | 0.947337 |

不同票型之间，不要求人数较多者必然有更大权重；不能根据表格重新定义“可信”来追求预期排序。所有单票题只能得到同一个权重，这是只使用人数和比例所能区分的信息范围。

参考计算如下；生产实现还需纳入本项目的数据与训练验收：

~~~python
from scipy.special import betainc

def posterior_trust(n, k, epsilon, a0=0.5, b0=0.5):
    if int(n) != n or int(k) != k or n < 1 or not 0 <= k <= n:
        raise ValueError("Require integer n >= 1 and 0 <= k <= n")
    if not 0 <= epsilon <= 1 or not (a0 > 0 and b0 > 0):
        raise ValueError("Invalid tolerance or prior")
    h = k / n
    a, b = a0 + k, b0 + n - k
    lower, upper = max(0.0, h - epsilon), min(1.0, h + epsilon)
    value = betainc(a, b, upper) - betainc(a, b, lower)
    return float(min(1.0, max(0.0, value)))  # only roundoff clipping
~~~

建议以 float64 离线计算并缓存。极小尾概率需检查 CDF 相减的数值稳定性，必要时用互补 CDF 或稳定区间积分；不要通过任意平滑或阈值截断改变权重定义。

### 4.4 更新时机与能够证明的结论

固定投票数据下，后验和权重在训练前计算一次。新增独立投票 $(\Delta N_i,\Delta K_i)$ 后才更新后验参数：

$$
a_i\leftarrow a_i+\Delta K_i,\qquad
b_i\leftarrow b_i+\Delta N_i-\Delta K_i.
$$

同时更新经验比例，再计算 $\lambda_i$。同一批票不能在不同 epoch 重复计数；新增相反意见可能使可信概率下降。

令 $T_i=\mathbf1_{G_i}$，在所声明的后验下有

$$
\lambda_i=\arg\min_{q\in[0,1]}\mathbb E[(q-T_i)^2\mid N_i,K_i].
$$

这说明它是可信事件的最优后验概率预测，不证明把该概率用于 TRO 后能最小化完整训练风险或提高 RM 奖励。该后验解释也不是对所有固定真实概率都成立的频率学覆盖保证。

固定正的先验参数和 $\varepsilon>0$、独立同分布投票下，人数增长时后验集中、经验比例趋近真实比例，因此 $\lambda_i\to1$。这不保证神经网络训练参数一致，也不保证有限训练预算下生成奖励逐点单调。

直接对真实 soft loss 取后验期望只得到后验均值标签：

$$
\mathbb E[\ell_i(\theta,p_i)\mid N_i,K_i]
=\ell_i\!\left(\theta,\frac{a_0+K_i}{a_0+b_0+N_i}\right).
$$

因此本方案是“用后验量化经验标签可信程度的 TRO”，不是把 posterior soft DPO 换一个名字。将可信概率作为混合系数仍是待检验的训练设计。对于对 $p$ 线性的损失，本 TRO 也等价于在收缩集合 $\lambda_i h_i+(1-\lambda_i)\mathcal U_i(\rho)$ 上做 DRO，不能声称超出所有可能的数据依赖 DRO 集合。

## 5. 实验组别与必要对照

| 方法名 | 单题训练目标 | 用途 |
|---|---|---|
| hard_dpo | $\ell_i(\theta,\mathbf1_{K_i>N_i/2})$ | 多数票硬标签基线 |
| soft_dpo | $\ell_i(\theta,h_i)$ | 经验比例基线 |
| label_dro_common | $J_i^{\mathrm{DRO}}(\theta;\rho)$ | 调优后的公共半径标签 DRO |
| posterior_confidence_tro | $\lambda_i\ell_i(\theta,h_i)+(1-\lambda_i)J_i^{\mathrm{DRO}}$ | 本版主候选 |
| oracle_soft_dpo | $\ell_i(\theta,p_i^*)$ | 同架构、同预算的信息充分参照 |

主场景 1、3、5 票无多数票平局。若未来加入偶数人数，hard 的平局规则预先固定并记录，不删除样本。

正式论证逐题可信程度的作用时，至少增加以下机制对照，保持相同数据、先验、$\varepsilon$、$\rho$、训练预算和重复：

- **matched_mean_lambda**：用训练集 $\bar\lambda=M^{-1}\sum_i\lambda_i$ 替代所有逐题权重，控制平均鲁棒比例。这不等同于已搜索出的最佳常数权重；若要比较最优常数权重，另在 dev 搜索并报告预算。
- **shuffled_lambda**：以预先记录的随机种子在训练题之间打乱 $\lambda_i$，保留整组权重分布；不只在相同 $(N_i,K_i)$ 组内打乱，也不改变标签、人数或 $\rho$。

打乱后直接使用置换后的题目—权重映射，不能再按原题的 $N_i,K_i$ 重算而撤销置换。若主方法最终选中 soft/DRO 端点，这些对照会退化为相同目标，应复用并实报；不能据退化结果声称逐题可信程度有效。

建议在主实验之后补充两类解释性对照，并单列预算：旧人数权重 $N_i/(N_i+c)$；使用相同先验的 posterior-mean soft DPO。前者帮助区分新旧可信规则，后者检验收益是否仅由更简单的后验软标签带来。未执行时不声称胜过这些对照。

## 6. 全局参数与首轮搜索建议

当前已经确定的是参数职责；下列具体先验和搜索数值为首轮建议，须在运行前写入冻结配置并绑定 commit，不能描述成已运行或理论最优。

| 参数 | 首轮建议 | 作用 |
|---|---|---|
| 主先验 $(a_0,b_0)$ | $(0.5,0.5)$，先固定 | 从投票到后验的参考假设 |
| 先验敏感性 | $(1,1)$ | 检查少票下结论是否依赖主先验 |
| $\varepsilon$ | $\{0.1,0.2,0.3\}$ | 全局比例精度容忍，越大越容易判定可信 |
| $\rho$ | $\{0.005,0.05,0.1\}$ | 全局 DRO 保守程度 |
| 退化端点 | $\varepsilon=0$、$\varepsilon=1$、$\rho=0$ | 分别复用对应 DRO 或 soft 目标 |

主要搜索轴为 $(\varepsilon,\rho)$；不用旧参数 $c$，也不用 CP 覆盖参数 $\delta$。先验仍是模型设定，不应说整个方法完全只有两个超参数。DPO $\beta$ 和公共训练超参数固定在各方法之间。

先验敏感性比较建议固定主方案在开发集选出的 $\varepsilon,\rho$，只改变先验后重新计算权重并训练，以隔离先验影响。若另一先验也单独调参，另列搜索预算和结论；不能看最终测试后选择表现较好的先验改称主方案。

### 6.1 九个内部 TRO 组合

| run_id | $\rho$ | $\varepsilon$ |
|---|---:|---:|
| PTRO-01 | 0.005 | 0.1 |
| PTRO-02 | 0.005 | 0.2 |
| PTRO-03 | 0.005 | 0.3 |
| PTRO-04 | 0.05 | 0.1 |
| PTRO-05 | 0.05 | 0.2 |
| PTRO-06 | 0.05 | 0.3 |
| PTRO-07 | 0.1 | 0.1 |
| PTRO-08 | 0.1 | 0.2 |
| PTRO-09 | 0.1 | 0.3 |

### 6.2 开发 A / B 与端点选择

开发 A、B 是同一开发 prompt 清单上的两次独立投票/重训/生成重复，不是两份不同开发集。

1. **A 全网格**：soft 1 + 非零 DRO 3 + 内部 TRO 9 + hard 1 + oracle 1，共 15 个不同训练目标。显式记录 soft 与各 DRO 所代表的退化端点，不额外重复训练相同目标。
2. **按家族入围**：DRO 家族候选为 soft 加 3 个非零 DRO；TRO 家族候选为 soft、3 个 DRO 和 9 个内部 TRO。按开发短缺各取前两名，不能只在非零 DRO 或内部 TRO 中选择。
3. **B 确认**：两个入围集合取并集、相同目标去重，最多 4 个候选加 1 个匹配 oracle，共最多 5 个训练任务。soft 或 DRO 端点入围时必须同样确认。
4. **冻结选择**：对 B 确认并集中所有已完成的目标计算 A/B 短缺均值，按所属家族确定最终配置。DRO 家族包含其中全部 soft/DRO 目标；TRO 家族包含其中全部 soft、DRO 和内部 TRO 目标，包括因 DRO 家族入围而完成 B 的端点。不能仅比较各家族在 A 阶段自行选出的两个目标，不能把只有 A 的指标与 A/B 平均混排，也不能宣称全部网格都做过两次重复。
5. 若最优 TRO 家族目标退化为 soft 或 DRO，如实报告；可以另报内部 TRO 最优候选，但不能用它替代家族真实选择结果。

主选择指标为开发集逐题 oracle 短缺 $S_O$，平均奖励 $R$ 同时报出。并列时先选较高 $R$，再选较小平均有效区间宽度 $(1-\lambda_i)(b_i^{\mathrm{KL}}-a_i^{\mathrm{KL}})$，再按较小 $\rho$ 和预定 run_id 排序。这里 KL 端点的符号与 Beta 参数不同。

推荐沿用可追踪种子：A=101/1101/2101，B=102/1102/2102；分别为 vote/train/generation。最终三重复建议使用 201/301/401、202/302/402、203/303/403。使用其他实际种子时提前记录，不能把同一模型换生成 seed 当作重新训练。

### 6.3 最终重复、复用与预算

固定配置后，异质场景至少 3 次独立投票与重训，每次五组方法共享同一投票及生成设置，最多 15 个模型作业。各方法同一重复使用匹配 oracle。新确认测试集不参与选参。

统一人数实验为五档 soft 加一组 oracle，三重复共 18 个模型；oracle 不随人数变化。满足相同文本、$p^*$、初始化、训练 seed、长度、实现和预算时，可与主实验复用 oracle。

首轮预算上限为开发 A 15 + B 5 + 最终 15 + 人数增长 18 = 53 个训练任务；复用三个 oracle 时为 50 个。退化目标及已核验的旧产物可进一步去重。这不包括失败重跑、机制消融和先验敏感性，也不意味着立即启动全部任务。

旧归档的日志显示 18 个收敛模型训练完成；checkpoint 为外置产物，其实际可获取性和 hash 待核验。优先核对实现、清单、投票前缀、训练配置与模型 hash，再判断是否直接补评测；不要只因文件名相同就复用，也不要无理由全部重训。

## 7. 统一人数增长实验的执行细节

每个训练题、每个重复生成最长 101 票的独立 Bernoulli 序列，$N=1,5,11,51,101$ 使用同一序列的前 $N$ 票。保存序列或可重建的逐题随机键；仅声称各档使用相同 seed 不足以证明嵌套前缀。

固定训练回答对数、文本、$p^*$、初始化和训练预算。每个 $N$ 从同一初始模型独立训练，只比较 empirical soft DPO 和 oracle；不能在上一档 checkpoint 上继续训练。评测仍每题生成 10 条回答，不把投票人数 $N$ 当成生成数量。

必须交付五档完整指标表，以及 $N$ 对 $R$、$N$ 对 $S_O$ 的两张图，展示三次重复及波动。标签 MSE 仅作模拟器诊断；不能用标签 MSE 的下降替代生成表现的实验结果，也不要求每次曲线严格单调。

## 8. 两个主评测指标与统计比较

同一冻结 RM 对模型在每个问题 $x$ 生成的 $G=10$ 条回答评分，先求题内均值：

$$
\bar r_x(\theta)=\frac1G\sum_{j=1}^{G}r_\phi(x,y_{xj}^{\theta}).
$$

对 $Q$ 个评测问题：

$$
\boxed{R(\theta)=\frac1Q\sum_x\bar r_x(\theta)\quad\text{越高越好},}
$$

$$
\boxed{
S_O(\theta)=\frac1Q\sum_x\max\{\bar r_x(\theta_O)-\bar r_x(\theta),0\}
\quad\text{越低越好}.
}
$$

模型在某题超过 oracle 时，该题短缺为 0；不足的题保留差值。先题内平均，再逐题取正部，最后题间平均。不能对全数据均值之差最后才截零，不能对原始负 RM 分数截零，也不把两模型第 $j$ 条独立生成强行配对。

两个指标共同构成结果报告，开发选参仍按预先规定的 $S_O$。不临时用测试集改成加权综合分。$S_O$ 是输出表现短缺，不是参数或生成分布的距离。

每次训练重复，以 prompt 为单位做 2,000 次配对 bootstrap，直接报告 TRO−soft、TRO−最佳公共 DRO 在 $R$ 和 $S_O$ 上的差值及 95% 区间；相同 bootstrap 索引同时抽取两方法及共同 oracle，保留题内生成的相关性。各方法自己的区间不能替代差值区间。另报告三次重训的均值和标准差，不能让 prompt bootstrap 替代重训。

有限生成次数会产生评测噪声。若差异接近噪声水平，可预先登记 oracle 独立生成重复等噪声检查，所有方法使用一致规则。不得不断追加直到显著。固定回答对上的概率、交叉熵等仅作可选诊断，不替代这两个主指标。

## 9. 实现验收与交付

### 9.1 权重与 loss 验收

- 对 $N=1,3,5$ 全票型核对本文件数值表，并检查标签交换对称、$0\le\lambda_i\le1$、对 $\varepsilon$ 的单调性、$\varepsilon=0/1$ 端点。
- 改变 $\rho$ 不得改变后验或 $\lambda_i$；固定投票跨 epoch 的权重不得变动。
- 验证 $h=0/1$ 的 KL 端点：分别为 $[0,1-e^{-\rho}]$、$[e^{-\rho},1]$；内部点用 float64 求端点。
- 因损失关于 $p$ 线性，$z_i>0$ 取 KL 下端点、$z_i<0$ 取上端点；$z_i=0$ 可取 $h_i$ 对应的合法次梯度。
- 最坏端点与权重停止梯度。有效概率为 $\lambda_i h_i+(1-\lambda_i)p_i^{\mathrm{adv}}$，logit 梯度应为 $\sigma(z_i)$ 减去该有效概率。
- 在实际 PyTorch trainer 中核验值和梯度、reference 冻结、回答 token mask、padding/EOS、不同 microbatch 与梯度累积、尾批处理。标量数学参考通过不等于训练器验收通过。
- 旧工程若存储的是鲁棒项权重，明确迁移方向；新 $\lambda_i$ 始终乘经验项。不能只改字段名。

### 9.2 复现材料

每个运行记录：run_id、阶段、纲要 commit、实现 commit、数据及投票 hash、模型/RM/tokenizer revision 或本地内容 hash、模板、先验、$\varepsilon,\rho$、vote/train/generation seeds、长度、真实 DP 并行度、microbatch、累积、有效 batch、优化器/scheduler、checkpoint、GPU 时间、失败原因及指标路径。

实际有效 batch = 每 DP rank 的回答对数 × DP world size × 梯度累积次数。归档方案的例子为 $1\times2\times64=128$；资源调整后记录实际值，不能用机器总 GPU 数替代实际 DP 数量。

回传完整候选表、每次重复的指标、逐题评分及配对差值区间。完整生成和大模型权重可外置，但应提供持久位置、大小、hash 和访问说明；核心 loss/trainer、投票生成代码及可复核的轻量指标不要只放在个人服务器路径。

## 10. 执行顺序与版本维护

1. **补齐旧实验可复核材料**：实际 trainer/loss、投票生成与观测、逐题评分；核实旧收敛数据的前缀构造。
2. **补完旧任务**：核验后评测已有收敛 checkpoint；补 soft 端点的开发 B 确认或明确记录旧协议偏离；补方法间配对统计。保留原结果，不追溯改名。
3. **冻结新轮配置**：在状态表记录主先验、$\varepsilon/\rho$ 网格、数据清单、种子、新确认测试集及预算。先完成数值与真实训练器验收，再做 100 对 smoke。
4. **新轮开发与最终实验**：按 A/B 含端点规则选参，冻结配置，做三次重复和机制对照，再执行独立确认评测。
5. **回传并更新**：每阶段更新状态与产物链接。若公式、标签中心、参数范围或评测改变，更新纲要版本和原因；旧任务仍引用当时 commit。

状态表区分“待执行、进行中、训练完成、生成完成、评分完成、分析完成、材料待补、失败”。一份总表不能把训练完成自动写成实验完成。

## 11. 来源与变更记录

- [Beta 后验与贝叶斯推断](https://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)
- [Jeffreys 先验的二项模型推导](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture7.pdf)
- [Beta CDF / 正则化不完全 Beta 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html)
- [2026-11-04 历史实验报告](artifacts/ALL_EXPERIMENTS_REPORT.md)

| 日期 | 变更 |
|---|---|
| 2026-11-09 | 在仓库根目录建立长期维护的纲要与状态表；主候选改为后验可信概率权重；保留经验标签、公共 KL 半径、两个输出指标及必做人数增长实验；修正选参端点规则并区分旧产物与新方案。 |
