import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import numpy as np
import time

sys.path.append(os.getcwd())
import pdist_custom # pyright: ignore[reportMissingImports]

torch.npu.config.allow_internal_format = False

class TestCustomPdist(TestCase):

    def custom_verify_result(self, output_npu, output_cpu):
        """
        自定义精度对比函数
        Args:
            output_npu: NPU算子输出的Tensor
            output_cpu: CPU标杆输出的Tensor
        """
        # 1. 转为numpy并展平，统一使用float32进行计算以避免numpy计算时的溢出
        output = output_npu.cpu().numpy().astype(np.float32).reshape(-1)
        golden = output_cpu.cpu().numpy().astype(np.float32).reshape(-1)

        # 2. 定义容差 (FP16精度较低，建议 rtol=1e-3, atol=1e-3)
        # 如果是 FP32，建议使用 rtol=1e-4, atol=1e-4 或更严
        relative_tol = 1e-3
        absolute_tol = 1e-3
        error_tol_ratio = 0.001  # 允许的错误点占比 (例如千分之一)

        print(f"\n[Verify] Shape: {output.shape}, Dtype: {output_npu.dtype}")
        
        # 3. 使用 np.isclose 进行对比
        different_element_results = np.isclose(output,
                                               golden,
                                               rtol=relative_tol,
                                               atol=absolute_tol,
                                               equal_nan=True)
        
        # 4. 获取不相等元素的索引
        different_element_indexes = np.where(different_element_results == False)[0]
        diff_count = len(different_element_indexes)
        total_count = len(golden)

        # 5. 打印详细错误信息 (最多打印前100个)
        if diff_count > 0:
            print(f">>> Found {diff_count} mismatches ({diff_count/total_count:.2%})")
            print(f"{'Index':<10} | {'Expected (CPU)':<15} | {'Actual (NPU)':<15} | {'Rel Diff':<10}")
            print("-" * 60)
            
            for index in range(len(different_element_indexes)):
                real_index = different_element_indexes[index]
                golden_data = golden[real_index]
                output_data = output[real_index]
                
                # 计算相对误差 (处理分母为0的情况)
                if abs(golden_data) < 1e-9:
                    rdiff = abs(output_data - golden_data)
                else:
                    rdiff = abs(output_data - golden_data) / abs(golden_data)

                print(f"{real_index:<10d} | {golden_data:<15.6f} | {output_data:<15.6f} | {rdiff:<10.6f}")
                
                if index >= 20:
                    print(f"... and {diff_count - 20} more mismatches ...")
                    break

        # 6. 计算错误率并断言
        error_ratio = float(diff_count) / total_count
        print(f"[Result] Error ratio: {error_ratio:.4f}, Tolerance threshold: {error_tol_ratio}")

        # 如果你需要严格的一致性，这里可以直接判断 diff_count == 0
        if error_ratio > error_tol_ratio:
            self.fail(f"Test Failed! Error ratio {error_ratio:.4f} > {error_tol_ratio}")
        else:
            print("Test Pass!")

    def test_pdist_custom_ops(self):
        shape = [100, 400]
        p = float('inf')  # 1.0, 2.0, float('inf')
        dtype = torch.float32  

        x = torch.rand(shape, device='cpu', dtype=dtype)
        x_npu = x.npu()

        # 运行 NPU 算子
        # 如果想对比 NPU 时间，可以在这里也加计时 (注意 NPU 需要同步)
        start_time = time.time()
        output_npu = pdist_custom.run_pdist_custom(x_npu, p)
        end_time = time.time()
        npu_time_ms = (end_time - start_time) * 1000
        
        print(f"\n[Performance] torch.pdist (NPU) time: {npu_time_ms:.4f} ms")
        
        output_cpu = None
        
        # ================== 统计 CPU 时间开始 ==================
        
        if (dtype == torch.float16):
            start_time = time.time()
            # 如果是 float16，这里包含了 转float -> 计算 -> 转half 的总耗时
            output_cpu = torch.pdist(x.float(), p).half()
            end_time = time.time()
        else:
            start_time = time.time()
            output_cpu = torch.pdist(x_npu, p)
            end_time = time.time()
            
        cpu_time_ms = (end_time - start_time) * 1000
        print(f"\n[Performance] torch.pdist (CPU) time: {cpu_time_ms:.4f} ms")
        # ================== 统计 CPU 时间结束 ==================

        # 调用自定义对比函数
        self.custom_verify_result(output_npu, output_cpu)



if __name__ == "__main__":
    run_tests()