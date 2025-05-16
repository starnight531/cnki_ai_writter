import re

def extract_numbers(s):
    return re.findall(r'\d+', s)

# 示例用法
s = "1, 2, 44, 3"
print(extract_numbers(s))  # 输出: ['1', '2', '3']s