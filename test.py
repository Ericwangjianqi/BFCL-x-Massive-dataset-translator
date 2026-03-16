import google.generativeai as genai

# 配置你的 API Key
genai.configure(api_key="AIzaSyCULquIODf4HX4EDVB2el9t-wwAub9ZRkA")

# 遍历并打印模型名称
for m in genai.list_models():
    print(m.name)