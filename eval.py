import json

gtPath = "gt_general.json"
predPath = "pred_general.json"


def matchSim(gt, pred, type="WZ"):
    """
    匹配相似类别的gt与pred数组
    """
    if type == "WZ":
        return (gt["WZlabel"] == pred["WZlabel"]) and \
            (gt["WZcoordinate"] == pred["WZcoordinate"])
    elif type == "PIC":
        return ((gt["PIClabel"] == pred["PIClabel"]) and (gt["PICcoordinate"] == pred["PICcoordinate"])) or \
               ((gt["PIClabel"] == pred["PIClabel"] == "NotFound") and (pred["PICcoordinate"] == "null"))


with open(gtPath, "rb") as f:
    gtdata = json.load(f)
with open(predPath, "rb") as f:
    preddata = json.load(f)

cnt = 0
total = 0

for i in range(len(gtdata)):
    for j in range(len(preddata)):
        if gtdata[i]["imagePath"] != preddata[j]["imagePath"]:
            continue
        else:
            for GT_each in gtdata[i]["gt"]:
                for PD_each in preddata[j]["preds"]:
                    # 如果两个文字标签匹配上，则total计数
                    if matchSim(GT_each, PD_each, type="WZ"):
                        # 如果两个图片标签匹配上，则cnt计数
                        if matchSim(GT_each, PD_each, type="PIC"):
                            cnt += 1
                        total += 1
print("-----------------"*2)
print("Match Accuracy: {}% ({}/{})".format(round(cnt/total,4)*100.0, cnt, total))
print("-----------------"*2)
