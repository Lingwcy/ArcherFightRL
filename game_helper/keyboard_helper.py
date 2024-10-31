from pynput.keyboard import Controller,Key
# kbd = Controller()
# while True:
#     time.sleep(0.3)
#     kbd.press('d')  # 模拟按下'a'键
# # # 特殊按键 如 'right' 'left' 需要 Key.[name]
# # # 普通按键 如 'a','b','c' 可以直接press


keyboard = Controller()
# ["w", "a", "s", "d","wa","wd","ds","sa"]
def key_down(key):
    if key == '': return
    elif key == "wa":
        keyboard.press('w')
        keyboard.press('a')
    elif key == "wd":
        keyboard.press('w')
        keyboard.press('d')
    elif key == "ds":
        keyboard.press('d')
        keyboard.press('s')
    elif key == "sa":
        keyboard.press('s')
        keyboard.press('a')
    else:keyboard.press(key)

def key_up(key):
    if key == '': return
    elif key == "wa":
        keyboard.release('w')
        keyboard.release('a')
    elif key == "wd":
        keyboard.release('w')
        keyboard.release('d')
    elif key == "ds":
        keyboard.release('d')
        keyboard.release('s')
    elif key == "sa":
        keyboard.release('s')
        keyboard.release('a')
    else:keyboard.release(key)