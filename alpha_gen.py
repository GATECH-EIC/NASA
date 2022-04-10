
arch = [
'k3_e6',
'k5_e3',
'k3_e3',
'add_k5_e3',
'add_k5_e3',
'k5_e3',
'add_k5_e3',
'add_k3_e3',
'add_k3_e3',
'k5_e6',
'add_k3_e6',
'add_k3_e6',
'shift_k3_e3',
'k3_e6',
'shift_k3_e6',
'shift_k3_e3',
'k3_e3',
'k5_e6',
'shift_k3_e3',
'shift_k5_e6',
'shift_k5_e6',
'shift_k5_e6',
]

alpha = []
for a in arch:
    list = [0]*19
    if a == 'k3_e1':
        list[0] = 1
    elif a == 'k3_e3':
        list[1] = 1
    elif a == 'k3_e6':
        list[2] = 1
    elif a == 'k5_e1':
        list[3] = 1
    elif a == 'k5_e3':
        list[4] = 1
    elif a == 'k5_e6':
        list[5] = 1
    elif a == 'add_k3_e1':
        list[6] = 1
    elif a == 'add_k3_e3':
        list[7] = 1
    elif a == 'add_k3_e6':
        list[8] = 1
    elif a == 'add_k5_e1':
        list[9] = 1
    elif a == 'add_k5_e3':
        list[10] = 1
    elif a == 'add_k5_e6':
        list[11] = 1
    elif a == 'shift_k3_e1':
        list[12] = 1
    elif a == 'shift_k3_e3':
        list[13] = 1
    elif a == 'shift_k3_e6':
        list[14] = 1
    elif a == 'shift_k5_e1':
        list[15] = 1
    elif a == 'shift_k5_e3':
        list[16] = 1
    elif a == 'shift_k5_e6':
        list[17] = 1
    elif a == 'skip':
        list[18] = 1
    alpha.append(list)
print(alpha)
