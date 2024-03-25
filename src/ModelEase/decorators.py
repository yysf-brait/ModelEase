import time
from functools import wraps

from . import model_list


def cost_record(explain: str = 'Method', record: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'\033[34m{explain} ({func.__name__}) ...\033[0m', end='\r')
            start = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(f'\033[31mError:', e, f'\033[0m')
                return None
            cost = time.time() - start
            print(f'\033[32m{explain} ({func.__name__}) : {cost} s\033[0m')
            if record is not None and hasattr(args[0], record):
                setattr(args[0], record, cost)
                model_list[args[0].name][record] = cost
            return result

        return wrapper

    return decorator
