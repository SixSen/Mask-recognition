import threading
import time

def test():

    for i in range(5):
        print('test ',i)
        time.sleep(1)


thread = threading.Thread(target=test)
thread.start()

for i in range(5):
    print('main ', i)
    time.sleep(1)
