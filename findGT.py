import os
import json
from time import sleep
from PIL import Image
import numpy as np
from aip import AipOcr
import shutil
from rich.progress import track


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

GT_Merged = []
cnt = 0
non_wz_label_list = ['wz1', 'wz3', 'wz1-l', 'wz3-l', 'wz2', 'wz4',
    'd1', 'h1', 'v1', 'range', 'mask', 'Ne', 'Ns', 'k1', 'pair', 'g1']


def genOcrText(img):
    """
    OCR识别
    :param img: 图片路径
    """
    """ 你的 APPID AK SK """
    APP_ID = '25093489'
    API_KEY = 'oXdM0yrF5b6dwiFL5SCQPzI0'
    SECRET_KEY = 'U6gxWimo38tSr0DTgSicbAC6GA9Nu4TR'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    text = client.basicAccurate(get_file_content(img))
    sleep(0.3)
    if "words_result" not in text.keys():
        return "QPS limit reached"
    else:
        return text["words_result"]


def genImageInfo(pointlh, pointrd, img):
    """
    用于得到图所对应文字信息
    :param pointlh: 左上角坐标
    :param pointrd: 右下角坐标
    :param img: PIL打开的图片
    """
    global cnt
    img2 = img.crop((pointlh[0], pointlh[1], pointrd[0], pointrd[1]))
    img2.save(os.path.join(path, "img" + str(cnt) + ".png"))
    txtInfo = genOcrText(os.path.join(path, "img" + str(cnt) + ".png"))
    print(txtInfo)
    return txtInfo


def del_labels(data, label2rm=['range', 'pair']):
    """
    用于删除data原有的range标签
    """
    c = 0
    # label2rm = ['range','pair'] #要删除的标签
    for i in range(len(data['shapes'])):
        if data['shapes'][i-c]['label'] in label2rm:
            data['shapes'].remove(data["shapes"][i-c])
            c += 1


def centerPoint(pointlh, pointrd):
    """
    中心点计算函数
    :param pointlh: 左上角坐标
    :param pointrd: 右下角坐标
    """
    return (int((pointlh[0] + pointrd[0]) / 2), int((pointlh[1] + pointrd[1]) / 2))


def search_wzline(lh, rd, data):
    """
    用于查找wzline并返回
    :param lh: 左上角坐标
    :param rd: 右下角坐标
    :param data: json数据
    return wzline_shape的另一头端点坐标
    """
    lh = (lh[0], lh[1])
    rd = (rd[0], rd[1])
    for i in range(len(data['shapes'])):
        if data['shapes'][i]['label'] in ['wz1-l', 'wz3-l']:
            xl = data['shapes'][i]['points'][0][0]
            xr = data['shapes'][i]['points'][1][0]
            yl = data['shapes'][i]['points'][0][1]
            yr = data['shapes'][i]['points'][1][1]
            # 如果在wz的方框范围内寻找到了wz-l，则返回wz-l的shape
            if (lh[0] < xl < rd[0] and lh[1] < yl < rd[1]):
                # 如果线的左端点落入wz方框内，返回wzl的shape右端点
                return (xr, yr)
            elif (lh[0] < xr < rd[0] and lh[1] < yr < rd[1]):
                return (xl, yl)
    # 如果没有找到wz-l，则返回None
    return None


def searchNearShape(wzpoint, data, already_paired, wztxt="info", mode="main", scale=50):
    """
    搜索最邻近标签（暂时未引入语义增强）
    :param wzpoint: wz标签区域的中心点坐标
    :param wztxt:（未完善）wz区域识别出的语义信息
    :param data: json数组
    :param already_paired: 已经匹配过的wz标签
    :param mode: 模式，默认为main，主程序模式，test为测试模式
    :param scale: 搜索尺度,后续需要可变功能，根据image的w，h来设定
    """
    xmin = wzpoint[0] - scale
    xmax = wzpoint[0] + scale
    ymin = wzpoint[1] - scale
    ymax = wzpoint[1] + scale
    if mode == "test":
        # draw_out_rectangle写入临时json，用于观察中心点框的坐标
        draw_out_rectangle = {"label": "range", "shape_type": "rectangle",
                              "points": [[xmin, ymin], [xmax, ymax]], "flags": {}, "group_id": None}
        data["shapes"].append(draw_out_rectangle)
        with open(data["imagePath"].split(".")[0]+".json", 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    choiceList = []
    for i, shape in enumerate(data["shapes"]):
        if shape["label"] not in ["mask", "Ne", "Ns"] and shape["shape_type"] not in ["point", "line"]:
            lh, rd = shape["points"][0], shape["points"][1]  # lh代表左上角，rd代表右下角
            ct_this = centerPoint(lh, rd)
            # 如果该标签框的中心点在搜索范围内，则进一步进行label的选择
            if xmin <= ct_this[0] <= xmax and ymin <= ct_this[1] <= ymax:
                # 搜索的该标签不可以是wz标签
                if shape["label"] not in non_wz_label_list:
                    choiceList.append([shape, ct_this])
    # 如果搜索范围内有多个标签，则选择最靠近wz标签的标签; 如果最靠近wz标签的标签以及已经匹配过，则选择下一个标签
    result = []
    if len(choiceList) >= 1:
        choiceList.sort(key=lambda x: (
            x[1][0]-wzpoint[0])**2+(x[1][1]-wzpoint[1])**2, reverse=False)
        i = 0
        while choiceList[i][0] in already_paired:
            if (i+1) < len(choiceList):
                i += 1
            else:
                return None
        result = choiceList[i]  # 返回最靠近的标签
        return result
    else:
        try:
            raise Exception("没有找到合适的标签!")
        except:
            return None


def mainProc(path, files, gtPath, search_scale=100, ocr_flag=False):
    """
    图文匹配主程序函数
    :param path: 图片-json路径
    :param files: 文件名列表
    :param ocr_flag: 是否使用OCR识别，默认为False
    """
    global cnt
    for i in track(range(len(files)), description="Processing"):
        if files[i].endswith(".json"):
            findGT(path, files[i], gtPath, search_scale, ocr_flag)
    with open(os.path.join(gtPath, "gtJson", "gt_general.json"), "w", encoding="utf-8") as f:
        json.dump(GT_Merged,f,ensure_ascii=False,indent=4)


def findGT(path, file, gtPath, search_scale=100, ocr_flag=False):
    """
    寻找groundTruth
    :param path: 图片-json路径
    :param file: 文件名
    :param gtPath: groundTruth路径
    :param search_scale: 搜索尺度
    :param ocr_flag: 是否进行ocr识别
    """
    GroundTruth = {"imagePath": "", "gt": []}
    GroundTruth["imagePath"] = file.split(".")[0]+".png"
    already_paired = []
    oriscale = search_scale  # 存储搜索尺度初值
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        del_labels(data)  # 删除无用标签
        img = Image.open(os.path.join(path, data['imagePath']))
        for i, shape in enumerate(data["shapes"]):
            if shape["label"] in ["wz1", "wz3"]:
                # lh代表左上角，rd代表右下角
                lh, rd = shape["points"][0], shape["points"][1]
                ct_wz = centerPoint(lh, rd)  # 文字区域中心点
                # wlshape代表wz1或wz3的-line标签shape
                line_endPoint = search_wzline(lh, rd, data)
                # line_endPoint = None

                ##--------------------------------ocr识别--------------------------------------##
                if ocr_flag:
                    txt = genImageInfo(lh, rd, img)  # 文字信息
                else:
                    txt = "non-usable-ocr-result"

                ##---------------------------寻找最近的标签shape--------------------------------##
                nearest_shape = None
                while nearest_shape is None:
                    if line_endPoint is not None:
                        nearest_shape = searchNearShape(
                            line_endPoint, data, already_paired, txt, mode="main", scale=search_scale)
                        if search_scale > 200:
                            res_label = "NotFound"
                            res_point = line_endPoint
                            break
                    else:
                        nearest_shape = searchNearShape(
                            ct_wz, data, already_paired, txt, mode="test", scale=search_scale)
                        if search_scale > 500:
                            res_label = "NotFound"
                            res_point = line_endPoint
                            break
                    search_scale += 20

                search_scale = oriscale
                if nearest_shape is None:
                    GroundTruth["gt"].append({"WZlabel": shape['label'], "PIClabel": res_label,
                                   "WZcoordinate": shape['points'], "PICcoordinate": res_point})
                    continue

                ##------------------------------绘制出匹配线段-----------------------------------##
                ct_pic = nearest_shape[1]

                # pair_line写入临时json，用于连接匹配上的文字区域和图片区域
                pair_line = {"label": "pair", "shape_type": "line", "lineType": "LL",
                             "points": [[ct_wz[0], ct_wz[1]], [ct_pic[0], ct_pic[1]]], "flags": {}, "group_id": None}

                data["shapes"].append(pair_line)
                with open(os.path.join(path, data["imagePath"].split(".")[0]+".json"), 'w', encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                # GroundTruth存储结构化信息
                GroundTruth["gt"].append({"WZlabel": shape['label'], "PIClabel": nearest_shape[0]['label'],
                               "WZcoordinate": shape['points'], "PICcoordinate": nearest_shape[0]['points']})

                already_paired.append(nearest_shape[0])  # 防止重复匹配
    name = "gt_"+file.split(".")[0]+".json"
    with open(os.path.join(gtPath,"gtJson",name), 'w', encoding="utf-8") as f:
        json.dump(GroundTruth, f, ensure_ascii=False, indent=4)
        GT_Merged.append(GroundTruth)

def showResult(path, dir, dst):
    """
    清空除文字和连接线之外的所有元素，展示结果
    :param path: json文件目录
    :param dir: json文件名集合
    :param dst: 结果保存路径
    """
    if not os.path.exists(dst):
        os.mkdir(dst)
    for file in dir:
        if os.path.exists(os.path.join(path, file.split(".")[0]+".png")):
            shutil.copy(os.path.join(path, file.split(".")[
                        0]+".png"), os.path.join(dst, file.split(".")[0]+".png"))
        # 如果存在同名json则删除无用标签
        if os.path.exists(os.path.join(path, file.split(".")[0]+".json")):
            with open(os.path.join(path, file.split(".")[0]+".json"), 'rb') as f:
                data2=json.load(f)
                del_labels(data2, label2rm=[
                           "mask", "wz1-l", "wz3-l", "range"])  # 删除无用标签
            json.dump(data2, open(os.path.join(dst, file.split(".")[
                      0]+".json"), 'w', encoding="utf-8"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    path = "C:\\Users\\19586\\Downloads\\pairWZ\\data"  # 有图有json的文件夹
    files = os.listdir(path)
    # findGT(path,"43.json",search_scale=100, ocr_flag=False)
    # showResult(path,["43.json"],dst="C:\\Users\\19586\\Downloads\\pairWZ\\result")
    genPath="C:\\Users\\19586\\Downloads\\pairWZ\\result_gt"
    mainProc(path, files, gtPath=genPath, search_scale=100, ocr_flag=False)
    showResult(path, files, dst=genPath)

