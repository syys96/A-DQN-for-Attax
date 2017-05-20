def action_pos():
    count = 0
    action_pos_dict = {}
    for i in range(7):
        for j in range(7):
            for a in range(-2,3):
                for b in range(-2,3):
                    if (a==0 and b==0) or i+a<0 or i+a>=7 or j+b<0 or j+b>=7:
                        continue
                    action = {}
                    action['x0'] = i+a
                    action['y0'] = j+b
                    action['x1'] = i
                    action['y1'] = j
                    action_pos_dict[count] = action
                    count += 1
    return action_pos_dict
def getaction_pos(action,action_dict):
    for i in range(len(action_dict)):
        if action_dict[i] == action:
            return i

