from pathlib import Path

# 加载提示词的函数
def load_prompt(name: str):
    path = Path(__file__).parents[2] / 'prompt' / f'{name}.prompt'
    return path.read_text(encoding='utf-8')

if __name__ == '__main__':
    print(load_prompt('correct_sql'))