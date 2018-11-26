import psutil

def available_ram():
    vm = psutil.virtual_memory()
    return 'Free RAM: %.1f/%.1f GB' % (vm[1] / 10**9, vm[0] / 10**9)
