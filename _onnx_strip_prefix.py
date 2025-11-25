import onnx

def strip_op_type_prefix(model_path, output_path):
    model = onnx.load(model_path)
    for node in model.graph.node:
        # 只保留最后一个'::'后的内容
        if '::' in node.op_type:
            node.op_type = node.op_type.split('::')[-1]
    onnx.save(model, output_path)

if __name__ == "__main__":
    strip_op_type_prefix("ofdm_process.onnx", "ofdm_process_noprefix.onnx")
    strip_op_type_prefix("ofdm_generator.onnx", "ofdm_generator_noprefix.onnx")
    strip_op_type_prefix("dft256.onnx", "dft256_noprefix.onnx")
    strip_op_type_prefix("dft256_mat.onnx", "dft256_mat_noprefix.onnx")