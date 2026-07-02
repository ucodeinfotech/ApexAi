import pickle, sys
BASE = r'C:\Users\pc\Downloads\stock hist data'
sys.path.insert(0, BASE)

class SafeLoader(pickle.Unpickler):
    def find_class(self, mod, name):
        return super().find_class(mod, name)

with open(BASE + r'\return_prediction_report_v6\results_v6.pkl', 'rb') as f:
    v6 = SafeLoader(f).load()

print(f'Keys: {list(v6.keys())}')
print(f'Contains "models": {"models" in v6}')
print(f'Contains "cost_rt": {"cost_rt" in v6}')
print(f'Contains "bt": {"bt" in v6}')
print(f'Contains "rd": {"rd" in v6}')
