import numpy as np
from py3gpp.nrLDPCEncode import nrLDPCEncode

def run_py3gpp_ldpc(input_data):
    """
    使用py3gpp库进行LDPC编码
    input_data: shape (1, 8448), dtype np.int32, 元素为0或1
    返回: shape (1, 25344), dtype np.int32
    """
    K = 8448
    bgn = 1
    # py3gpp要求输入shape为(K, 1)，且类型为int8
    assert input_data.shape == (1, K)
    x = input_data.reshape((K, 1)).astype(np.int8)
    y = nrLDPCEncode(x, bgn, algo='thangaraj')  # 输出shape (25344, 1)
    y = y[:, 0].reshape(1, -1).astype(np.int32) # 转为(1, 25344)
    return y

if __name__ == "__main__":
    np.random.seed(42)
    batch_size = 1
    K = 8448
    output_len = 25344
    # 输入为INT32，shape为(1, 8448)，且只包含0或1
    x = np.random.randint(0, 2, (K,)).astype(np.int32)
    x_batch = x.reshape((1, K))  # (1, 8448)

    # LDPC编码
    t1 = None
    import time
    start = time.time()
    y_batch = run_py3gpp_ldpc(x_batch)
    t1 = time.time() - start

    print(f"py3gpp LDPC编码 1个batch总时间: {t1*1000:.2f} ms")
    print(f"输入shape: {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"输出shape: {y_batch.shape}, dtype: {y_batch.dtype}")
