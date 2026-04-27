#新增#
#以下为直接加载启动的方式
#如果在过程中需要安装新的三方库，请在安装完成后，重新运行指导文档中的3.模型包文件生成参考，以生成最新依赖库列表并下载离线包。
import json
import time
from vllm import LLM, SamplingParams
from transformers import AutoModel

from prompt import *

class Competition:
    def __init__(self):
        #加载获取参数量
        model = AutoModel.from_pretrained(
            "Qwen2.5-3B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Total parameters: {model.num_parameters() / 1e9:.2f}B")
        del model

        #以VLLM加载
        self.llm = LLM(model="Qwen2.5-3B-Instruct")  #注意相对引用路径
        self.sampling_params = {
            "choice": SamplingParams(max_tokens=2048, temperature=0.8, top_p=0.95),
            "code-generate": SamplingParams(n=3, max_tokens=2048, temperature=0.8, top_p=0.95),
            "generic-generate": SamplingParams(max_tokens=2048, temperature=0.8, top_p=0.95),
            "math": SamplingParams(max_tokens=2048, temperature=0.8, top_p=0.95)
        }
        self.examples = {
            "math": [],
            "code-generate" : [],
            "generic-generate" : [],
            "choice" : [],
        }
        with open ("data/few_shot_2.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                type_ = item['type']
                if type_ == "choice":
                    shot = {"input": f"{item['prompt']}\nA){item['choices']['A']}\nB){item['choices']['B']}\nC){item['choices']['C']}\nD){item['choices']['D']}", "output": item['content']}
                else:
                    shot = {"input": item['prompt'], "output": item['content']}
                self.examples[type_].append(shot)
        
    def load_data(self, data_file="test.jsonl"):
        a_datas = []
        with open(data_file, 'r', encoding="utf-8") as f:
            for line in f:
                a_data = json.loads(line)
                a_datas.append(a_data)
        self.a_datas = a_datas
        
    def get_results(self, jsondata_list):
        # --- 批处理核心修改 ---
        # vllm.generate() 一次调用可以接受一个SamplingParams对象。
        # 如果一个批次内的数据类型不同，它们的采样参数也可能不同，这会产生冲突。
        # 最稳健的并行处理方式是按类型对数据进行分组，然后对每个组进行批处理。
        
        # 1. NEW: 按任务类型对输入数据进行分组
        batched_requests = {}
        for data in jsondata_list:
            type_ = data.get("type")
            if type_ not in batched_requests:
                batched_requests[type_] = []
            batched_requests[type_].append(data)

        # 2. NEW: 存储所有任务的最终结果
        all_results = []

        # 3. NEW: 对每个同类型的批次进行并行推理
        for type_, data_batch in batched_requests.items():
            print(f"Processing batch of type '{type_}' with {len(data_batch)} requests...")
            
            prompts_batch = []
            # 准备当前批次的所有prompts
            for a_data in data_batch:
                template = PROMPT_TEMPLATES[type_]
                question = ""
                if type_ == "choice":
                    choices = a_data["choices"]
                    question = a_data["prompt"] + f"\nA) {choices['A']}\nB) {choices['B']}\nC) {choices['C']}\nD) {choices['D']}\n"
                else:
                    question = a_data["prompt"]
                ex = self.examples[type_]
                prompt = template.format(system=SYSTEM[type_], question=question, eq0=ex[0]["input"], ea0=ex[0]["output"], eq1=ex[1]["input"], ea1=ex[1]["output"])
                prompts_batch.append(prompt)

            # 关键！一次性将整个批次的prompts列表传给generate函数
            sampling_params_for_batch = self.sampling_params[type_]
            outputs_batch = self.llm.generate(prompts_batch, sampling_params_for_batch)

            # 按顺序处理返回的结果
            for original_data, output in zip(data_batch, outputs_batch):
                id_ = original_data.get("id")
                generated_text = []
                for o in output.outputs:
                    generated_text.append(o.text)
                
                # 根据n的值来决定是返回列表还是单个字符串
                generated_text = generated_text[0] if len(generated_text) == 1 else generated_text
                all_results.append({"id": id_, "content": generated_text})
        
        # 4. NEW: 按照原始ID排序（可选，但推荐），以保证输出顺序的确定性
        # id_to_index = {data['id']: i for i, data in enumerate(jsondata_list)}
        # all_results.sort(key=lambda x: id_to_index[x['id']])
        
        return {"result": {"results": all_results}}

if __name__ == "__main__":
    comp = Competition()
    comp.load_data(data_file="data/A-data.jsonl") #load_data函数加载本地数据集用于验证跑通流程，可使用A榜数据集测试。
    res = comp.get_results(comp.a_datas[:10]) #测试前10条,在ModelArts中能顺利跑出结果后即可进行提交。
    if len(comp.a_datas) >= 1:
        test_batch = comp.a_datas 
        print(f"--- Running parallel inference on a batch of {len(test_batch)} items ---")
        
        # ADDED: 记录开始时间
        start_time = time.time()

        res = comp.get_results(test_batch)

        # ADDED: 记录结束时间
        end_time = time.time()
        duration = end_time - start_time

        print("\n--- Batch Inference Result ---")
        # print(json.dumps(res, indent=4, ensure_ascii=False))

        # ADDED: 打印总耗时和性能指标
        print("\n--- 性能统计 ---")
        print(f"处理 {len(test_batch)} 条数据总耗时: {duration:.4f} 秒")
        if duration > 0:
            throughput = len(test_batch) / duration
            print(f"吞吐率 (Throughput): {throughput:.2f} items/sec")
        
    else:
        print("Test data has less than 4 items, please check test.jsonl")
    #判题调用方式
    # for item in [测试集1，测试集2.....测试集n]：
    #     res = comp.get_results(item)

