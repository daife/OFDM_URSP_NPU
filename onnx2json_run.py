import sys
from onnx2json import convert
import json

def main():
    onnx_path = "ofdm_process_noprefix_sim.onnx"
    json_path = "ofdm_process_noprefix_sim.json"
    # 调用官方库进行转换
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(convert(onnx_path), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
#图形化分析可以去网站netron，借用ai进行分析则使用本代码导出json 