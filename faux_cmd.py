import subprocess

print("I figured out how to interact with the Terminal using python!")
print("(Triple colon implies terminal call.)")
print("- - - - -\n")

while True:
    subprocess.Popen(input("::: "), shell=True).wait()
