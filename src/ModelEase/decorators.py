import time
from functools import wraps

from . import table


def cost_record(explain: str = 'Method', record: str | bool = False):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            s = f'Class[{args[0].__class__.__name__}] {explain}'
            print(f'\033[34m{s} ...\033[0m', end='\r')
            start = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(f'\033[31mError:', e, f'\033[0m')
                return None
            cost = time.time() - start
            print(f'\033[32m{s}: {cost} s\033[0m')
            if record is not False:
                attr = f'{record}_Cost' if isinstance(record, str) else f'{explain}_Cost'
                table[args[0].name][attr] = cost
                if hasattr(args[0], attr):
                    setattr(args[0], attr, cost)
            return result

        return wrapper

    return decorator
