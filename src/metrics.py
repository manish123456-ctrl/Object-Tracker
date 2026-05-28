def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

gt_data = [
    {"ids": [1,2],   "boxes": [[100,150,180,320],[300,140,380,310]]},
    {"ids": [1,2],   "boxes": [[110,150,190,320],[310,140,390,310]]},
    {"ids": [1,2,3], "boxes": [[120,150,200,320],[320,140,400,310],[500,160,570,330]]},
]
pred_data = [
    {"ids": [1,2],   "boxes": [[102,153,182,322],[298,138,379,309]]},
    {"ids": [1,2],   "boxes": [[111,152,191,321],[309,139,389,308]]},
    {"ids": [1,2,3], "boxes": [[121,151,201,321],[318,139,398,309],[501,161,571,331]]},
]

total_gt    = 0
matched     = 0
id_switches = 0
prev_ids    = {}

for gt, pred in zip(gt_data, pred_data):
    total_gt += len(gt["ids"])
    for gid, gbox in zip(gt["ids"], gt["boxes"]):
        for pid, pbox in zip(pred["ids"], pred["boxes"]):
            if calculate_iou(gbox, pbox) > 0.5:
                matched += 1
                if gid in prev_ids and prev_ids[gid] != pid:
                    id_switches += 1
                prev_ids[gid] = pid
                break

mota = (matched - id_switches) / total_gt * 100
idf1 = (2 * matched) / (total_gt + matched) * 100

print("=" * 40)
print("  Object Tracking Metrics")
print("=" * 40)
print(f"  Total GT objects : {total_gt}")
print(f"  Matched          : {matched}")
print(f"  ID Switches      : {id_switches}")
print(f"  MOTA             : {mota:.1f}%")
print(f"  IDF1             : {idf1:.1f}%")
print("=" * 40)