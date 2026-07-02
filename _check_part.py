import os
BASE = r'C:\Users\pc\Downloads\stock hist data\comprehensive_data'
main_files = {f for f in os.listdir(BASE) if f.endswith('.csv')}
shoper = [f for f in main_files if 'SHOPERSTOP' in f]
print(f'SHOPERSTOP in main: {shoper}')
part = os.listdir(os.path.join(BASE, '_part'))
print(f'_part files: {part}')
# Check if _part files are already in main
for pf in part:
    sym = pf.replace('.csv', '')
    has_main = any(sym in f for f in main_files)
    print(f'  {pf}: has_main={has_main}')
