from libs import *

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

def group_objects(objects, margin=30):
    groups = []
    for obj in objects:
        best_group = None
        best_bi = None
        for g in groups:
            for o in g:
                w1 = o.right-o.left + 2*margin
                h1 = o.bottom-o.top
                w2 = obj.right-obj.left + 2*margin
                h2 = obj.bottom-obj.top
                bi = box_intersection(o.left-margin, o.top, w1, h1,
                                      obj.left-margin, obj.top, w2, h2)
                if bi > best_bi:
                    best_bi = bi
                    best_group = g
        if best_bi:
            best_group.append(obj)
        else:
            groups.append([obj])
    return groups

def reduce_objects(objects, iou_threshold=.5, merge_boxes=True):
    result = []
    for obj in objects:
        matched = False
        l1 = obj.left
        r1 = obj.right
        t1 = obj.top
        b1 = obj.bottom
        for i, obj2 in enumerate(result):
            l2,r2,t2,b2 = obj2.left, obj2.right, obj2.top, obj2.bottom
            iou = box_iou(l1, t1, r1-l1, b1-t1,
                          l2, t2, r2-l2, b2-t2)
            if iou > iou_threshold:
                if merge_boxes:
                    box = (min(l1,l2), max(r1,r2), min(t1,t2), max(b1,b2))
                    result[i].left, result[i].right, result[i].top, result[i].bottom = box
                matched = True
                break
        if not matched:
            result.append(obj)
    return result
