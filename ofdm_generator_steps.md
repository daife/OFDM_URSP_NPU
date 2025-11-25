OFDM 发送端（生成端）主要处理步骤如下：

---

- [ ] **1. 比特流分组与 LDPC 编码**
  - 将原始数据转换为比特流，并分组。
  - 对每组比特流进行 LDPC 编码，提升抗噪声能力。

---

- [ ] **2. QAM 调制**
  - 将 LDPC 编码后的比特流映射为 QAM 调制符号（如 64QAM），得到复数符号序列。

---

- [x] **3. 导频插入**
  - 按照预设的导频模式，将导频符号插入到指定的子载波位置，其余为数据符号。
  - 关键代码（_ofdm_generator.py）：
    ```python
 pilot_indices = torch.tensor([11, 25, 39, 53], dtype=torch.long, device=data_freq.device)
    pilot_value = torch.tensor([1.0, 0.0], dtype=data_freq.dtype, device=data_freq.device)
    out = data_freq.clone()
    # 向量化赋值
    out[:, :, pilot_indices] = pilot_value.view(2, 1).expand(2, pilot_indices.size(0))
    ```
遇到的一些问题:一开始使用了scatter算子，该算子未命中高优先级算子库（可能是在host侧实现的）
```

[PID: 11127] 2025-11-15-10:47:10.434.860 Operator_Missing_High-Priority_Performance(W11001): Op [node_select_scatter_1] does not hit the high-priority operator information library, which might result in compromised performance.
[PID: 11127] 2025-11-15-10:47:10.439.091 Operator_Missing_High-Priority_Performance(W11001): Op [node_select_scatter] does not hit the high-priority operator information library, which might result in compromised performance.
[PID: 11127] 2025-11-15-10:47:10.445.254 Operator_Missing_High-Priority_Performance(W11001): Op [node_select_scatter_2] does not hit the high-priority operator information library, which might result in compromised performance.
```
解决方法：参考[改图优化](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/practicalcases/GeneralPerformanceIssue/toolsample6_191.html)使用[netron](https://netron.app/)查询到scatter算子所在的代码段落。修改为向量化赋值。



---

- [x] **4. IDFT 变换**
  - 将一帧（一个 OFDM 符号）的所有子载波上的复数符号做逆快速傅里叶变换（IDFT），得到时域 OFDM 符号。优化空间：使用FFT递归处理，降低时间复杂度。

---

- [ ] **5. 添加循环前缀（CP）（暂时没考虑）**
  - 在每个时域 OFDM 符号前添加循环前缀，便于接收机区分响铃的ofdm符号。

---

- [ ] **6. 帧拼接与发送（暂时未考虑）**

---

**数据流存储说明：**
- 原始比特流：`torch.Tensor` 或 `numpy.ndarray`，0/1
- LDPC 编码后比特流：同上
- QAM 调制符号：复数类型，形状为 `[batch_size, n_subcarriers]`
- IFFT 后时域符号：同上
- 添加 CP 后信号：同上，长度增加为 `[batch_size, n_subcarriers + cp_len]`

---

**总结流程：**
比特流分组 → LDPC 编码 → QAM 调制 → 导频插入 → IFFT → 添加 CP → 拼接帧 → 发送信号

如需某一步详细代码，可进一步指定。
