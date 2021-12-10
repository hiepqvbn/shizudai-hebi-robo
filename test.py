import keyboard

stt=True

while stt:

    print("run")
    # Collect events until released
    if keyboard.is_pressed('h'):
        print("griped")
    if keyboard.is_pressed('j'):
        print("left")
    if keyboard.is_pressed('esc'):
        stt = False
    
print("complete")


