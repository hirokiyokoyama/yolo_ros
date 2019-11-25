def box_overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = max(l1, l2)
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = min(r1, r2)
    return right - left

def box_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    w = box_overlap(ax, aw, bx, bw)
    h = box_overlap(ay, ah, by, bh)
    if w < 0 or h < 0:
        return 0
    return w*h

def box_union(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return u

def box_iou(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return float(i)/u

def obj_iou(obj1, obj2):
    return box_iou((obj1.left + obj1.right)/2,
                   (obj1.top + obj1.bottom)/2,
                   obj1.right - obj1.left,
                   obj1.bottom - obj1.top,
                   (obj2.left + obj2.right)/2,
                   (obj2.top + obj2.bottom)/2,
                   obj2.right - obj2.left,
                   obj2.bottom - obj2.top)
