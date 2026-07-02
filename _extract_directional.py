"""Extract directional strategy info from notebook"""
import json
with open(r'C:\Users\pc\Downloads\stock hist data\full_return_model_pipeline.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src.lower() for kw in ['directional', 'strategy definition', 'long-only', 'long short',
                                          'threshold', 'market timing', 'combined', 'def backtest']):
        print(f'--- Cell {i} ({cell["cell_type"]}) ---')
        lines = src.split('\n')
        for l in lines[:60]:
            print(l)
        print()
