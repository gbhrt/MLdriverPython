import time

def run(shared):
    while(1):
        if shared.exit_program == True:
            break
        print("main prog")
        time.sleep(1)