from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# 模型路径（不用改）
model_name = "/mnt/data/chatglm3-6b"

# 所有测试问题，一次性全部放进去
questions = [
    "请说出以下两句话区别在哪里? 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上，请解释这两句话的不同含义",
    " 他知道我知道你知道他不知道吗？这句话里，到底谁不知道",
    " 明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    " 领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就  不好意思了。小明：是我不好意思。请问：以上意思分别是什么意思。"
]

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

# 自动依次回答所有问题
print("=" * 60)
print(" 开始依次回答所有测试问题...")
print("=" * 60)

for i, prompt in enumerate(questions, 1):
    print(f"\n\n【第 {i} 个问题】")
    print(f"问题：{prompt}")
    print("-" * 50)
    print("模型回答：")
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(input_ids=inputs, streamer=streamer, max_new_tokens=400)

print("\n" + "="*60)
print(" 所有问题回答完毕！作业完成 ✅")
print("="*60)