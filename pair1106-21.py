import os
import json
from time import sleep
from PIL import Image
import numpy as np
from aip import AipOcr
import shutil

path = "C:\\Users\\19586\\Downloads\\pairWZ\\data" #有图有json的文件夹
files = os.listdir(path)
cnt = 0
non_wz_label_list = ['wz1','wz3','wz1-l','wz3-l','wz2','wz4','d1','h1','v1','range','mask','Ne','Ns','k1','pair']

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

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
    text = client.basicGeneral(get_file_content(img))
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

def del_labels(data, label2rm=['range','pair']):
    """
    用于删除data原有的range标签
    """
    c = 0
    # label2rm = ['range','pair'] #要删除的标签
    for i in range(len(data['shapes'])):
        if data['shapes'][i-c]['label'] in label2rm:
            data['shapes'].remove(data["shapes"][i-c])
            c+=1

def centerPoint(pointlh, pointrd):
    """
    中心点计算函数
    :param pointlh: 左上角坐标
    :param pointrd: 右下角坐标
    """
    return (int((pointlh[0] + pointrd[0]) / 2), int((pointlh[1] + pointrd[1]) / 2))

def search_wzline(lh,rd, data):
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
        if data['shapes'][i]['label'] in ['wz1-l','wz3-l']:
            xl = data['shapes'][i]['points'][0][0]
            xr = data['shapes'][i]['points'][1][0]
            yl = data['shapes'][i]['points'][0][1]
            yr = data['shapes'][i]['points'][1][1]
            # 如果在wz的方框范围内寻找到了wz-l，则返回wz-l的shape
            if (lh[0] < xl < rd[0] and lh[1] < yl < rd[1]):
                #如果线的左端点落入wz方框内，返回wzl的shape右端点
                return (xr,yr)
            elif (lh[0] < xr < rd[0] and lh[1] < yr < rd[1]):
                return (xl,yl)
    # 如果没有找到wz-l，则返回None
    return None

def findChoice(shapeMe, data, mode="main",scale=50):
    """
    搜索最邻近标签list（暂时未引入语义增强）
    :param wzpoint: wz标签区域的中心点坐标
    :param wztxt:（未完善）wz区域识别出的语义信息
    :param data: json数组
    :param mode: 模式，默认为main，主程序模式，test为测试模式
    :param scale: 搜索尺度,后续需要可变功能，根据image的w，h来设定
    return: 返回choiceDict
    """
    
    lh,rd = shapeMe["points"][0],shapeMe["points"][1]
    wzpoint = centerPoint(lh,rd)
    choiceDict = {"self":[shapeMe,wzpoint], "nearlabel":[]}

    xmin = wzpoint[0] - scale
    xmax = wzpoint[0] + scale
    ymin = wzpoint[1] - scale
    ymax = wzpoint[1] + scale
    if mode == "test":
        # draw_out_rectangle写入临时json，用于观察中心点框的坐标
        draw_out_rectangle = {"label": "range","shape_type": "rectangle",
                              "points": [[xmin,ymin],[xmax,ymax]],"flags": {},"group_id": None}
        data["shapes"].append(draw_out_rectangle)
        with open(data["imagePath"].split(".")[0]+".json",'w',encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)

    for i,shape in enumerate(data["shapes"]):
        if shape["label"] not in ["mask","Ne","Ns"] and shape["shape_type"] not in ["point", "line"]:
            
            lh,rd = shape["points"][0],shape["points"][1] # lh代表左上角，rd代表右下角
            ct_this = centerPoint(lh,rd)
            # 如果该标签框的中心点在搜索范围内，则进一步进行label的选择
            if xmin <= ct_this[0] <= xmax and ymin <= ct_this[1] <= ymax:
                # 搜索的该标签不可以是wz标签
                if shape["label"] not in non_wz_label_list:
                    choiceDict["nearlabel"].append([shape,ct_this])
    # 如果搜索范围内有多个标签，则选择最靠近wz标签的标签; 如果最靠近wz标签的标签以及已经匹配过，则选择下一个标签
    if len(choiceDict["nearlabel"]) >= 2:
        choiceDict["nearlabel"].sort(key=lambda x:(x[1][0]-wzpoint[0])**2+(x[1][1]-wzpoint[1])**2, reverse=False)
        return choiceDict
    else:
        return None

def mainProc(path,files, search_scale=100, ocr_flag=False):
    """
    图文匹配主程序函数
    :param path: 图片-json路径
    :param files: 文件名列表
    :param ocr_flag: 是否使用OCR识别，默认为False
    """
    global cnt
    for file in files:
        if file.endswith(".json"):
            print("#"*20+ file + "#"*20)
            singleMatch(path, file, search_scale, ocr_flag)

def singleMatch(path, file, search_scale=100, ocr_flag=False):
    """
    测试用单个图片测试函数
    :param path: 图片-json路径
    :param file: 文件名
    :param search_scale: 搜索尺度
    :param ocr_flag: 是否进行ocr识别
    """
    oriscale = search_scale# 存储搜索尺度初值
    TotalList = [] # 存储全部choiceDics,方便统一管理
    with open(os.path.join(path,file),'r',encoding='utf-8') as f:
        data = json.load(f)
        del_labels(data) # 删除无用标签
        img = Image.open(os.path.join(path,data['imagePath']))
        for i,shape in enumerate(data["shapes"]):
            if shape["label"] in ["wz1","wz3"]:
                lh,rd = shape["points"][0],shape["points"][1] # lh代表左上角，rd代表右下角
                ct_wz = centerPoint(lh,rd) # 文字区域中心点
                #line_endPoint = search_wzline(lh,rd,data) # wlshape代表wz1或wz3的-line标签shape
                # line_endPoint = None
                ##---------------------------寻找最近的标签shape--------------------------------##
                choiceDics = None
                while choiceDics is None:
                    choiceDics = findChoice(shape,data,mode="test",scale=search_scale)
                    if search_scale > 1050:
                        res_label="NotFound"
                        res_point=None
                        break
                    search_scale += 200
                search_scale = oriscale
                if choiceDics is None:
                    if ocr_flag:
                        txt = genImageInfo(lh,rd,img) # 文字信息
                    else:
                        txt = "non-usable-ocr-result"
                    print("WZlabel: {:<10s} coordinate: {} ".format(shape['label'],shape['points']))
                    print('--------------'*2+"||"+"--------------"*2)
                    print('--------------'*2+"\/"+"--------------"*2)
                    print("Piclabel: {:<9s} coordinate: {} ".format(res_label,res_point))
                    print(" ")
                    continue
                TotalList.append(choiceDics)
        FindBestPair(TotalList, data, ocr_flag, img)

def distance2(x1,y1,x2,y2):
    return (x1-x2)**2+(y1-y2)**2

def compareDis(c1,c2):
    """
    返回距离大。需要修改为第二最近距离的choicDic
    """
    d1 = distance2(c1["self"][1][0],c1["self"][1][1],c1["nearlabel"][0][1][0],c1["nearlabel"][0][1][1])
    d2 = distance2(c2["self"][1][0],c2["self"][1][1],c2["nearlabel"][0][1][0],c2["nearlabel"][0][1][1])
    if d1 >= d2:
        return c1
    else:
        return c2

def FindBestPair(TotalList, data, ocr_flag, img):
    """
    寻找最优匹配方案
    :param TotalList:全部choiceDics的集合
    TotalList = [{"self":[shapeWZ1, ct_wz1], "nearlabel":[[shape1,ct_this1],...,[shapen,ct_thisn]]},
                 {"self":[shapeWZ2, ct_wz2], "nearlabel":[[shape1,ct_this1],...,[shapen,ct_thism]]},
                 ...
                 {"self":[shapeWZn, ct_wzn], "nearlabel":[[shape1,ct_this1],...,[shapen,ct_thisz]]}
                ]
    :param data: 图片jsondict
    """
    passlist = [] # 放置中心点坐标
    conflictlist = [] # 放置有冲突的choiceDic
    resList = []
    for i in range(len(TotalList)):
        ct_this = TotalList[i]["nearlabel"][0][1]
        if ct_this not in passlist:
            passlist.append(ct_this) # 把每个nearlabel的ct_this中心点都加到一个list中
        else:
            conflictlist.append(TotalList[i])

    if len(conflictlist) == 0:
        for choiceDic in TotalList:
            ct_wz = choiceDic["self"][1]
            ct_pic = choiceDic["nearlabel"][0][1]
            ##--------------------------------ocr识别--------------------------------------##
            if ocr_flag:
                txt = genImageInfo(choiceDic["self"][0]["points"][0],choiceDic["self"][0]["points"][1],img) # 文字信息
            else:
                txt = "non-usable-ocr-result"
            ## pair_line写入临时json，用于连接匹配上的文字区域和图片区域
            pair_line = {"label":"pair", "shape_type": "line", "lineType": "LL", 
                        "points":[[ct_wz[0],ct_wz[1]],[ct_pic[0],ct_pic[1]]],"flags": {},"group_id": None}
                    
            data["shapes"].append(pair_line)
            with open(os.path.join(path, data["imagePath"].split(".")[0]+".json"),'w',encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=4)
                
            ##---------------------------------打印信息--------------------------------------##    
            print("WZlabel: {:<10s} coordinate: {} ".format(choiceDic['self'][0]["label"],choiceDic['self'][0]["points"]))
            print('--------------'*2+"||"+"--------------"*2)
            print('--------------'*2+"\/"+"--------------"*2)
            print("Piclabel: {:<9s} coordinate: {} ".format(choiceDic["nearlabel"][0][0]['label'],choiceDic["nearlabel"][0][0]['points']))
            print(" ")
        
    else:
        # 寻找出含有conflictlabel的所有choiceDic,比较距离远近,修改TotalList
        for each in conflictlist:
            for i in range(len(TotalList)):
                if (TotalList[i]["nearlabel"][0][1] == each["nearlabel"][0][1]) and (TotalList[i] != each):
                    choice = compareDis(TotalList[i], each) #choice即为选出的选出的需要修改choiceDic
                    for j in range(len(TotalList)):
                        if choice == TotalList[j]:
                            # 更新TotalList
                            if len(TotalList[j]["nearlabel"])>1:
                                TotalList[j]["nearlabel"][0] = TotalList[j]["nearlabel"][1]
                    break
        
        for choiceDic in TotalList:
            ct_wz = choiceDic["self"][1]
            ct_pic = choiceDic["nearlabel"][0][1]

            if ocr_flag:
                txt = genImageInfo(choiceDic["self"][0]["points"][0],choiceDic["self"][0]["points"][1],img) # 文字信息
            else:
                txt = "non-usable-ocr-result"
            ## pair_line写入临时json，用于连接匹配上的文字区域和图片区域
            pair_line = {"label":"pair", "shape_type": "line", "lineType": "LL", 
                        "points":[[ct_wz[0],ct_wz[1]],[ct_pic[0],ct_pic[1]]],"flags": {},"group_id": None}
                    
            data["shapes"].append(pair_line)
            with open(os.path.join(path, data["imagePath"].split(".")[0]+".json"),'w',encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=4)
                
            ##---------------------------------打印信息--------------------------------------##    
            print("WZlabel: {:<10s} coordinate: {} ".format(choiceDic['self'][0]["label"],choiceDic['self'][0]["points"]))
            print('--------------'*2+"||"+"--------------"*2)
            print('--------------'*2+"\/"+"--------------"*2)
            print("Piclabel: {:<9s} coordinate: {} ".format(choiceDic["nearlabel"][0][0]['label'],choiceDic["nearlabel"][0][0]['points']))
            print(" ")
                                

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
        if os.path.exists(file.split(".")[0]+".png"):
            shutil.copy(os.path.join(path,file.split(".")[0]+".png"),os.path.join(dst,file.split(".")[0]+".png"))
        if os.path.exists(file.split(".")[0]+".json"): # 如果存在同名json则删除无用标签
            with open(os.path.join(path,file.split(".")[0]+".json"),'rb') as f:
                data2 = json.load(f)
                del_labels(data2, label2rm=["mask","wz1-l","wz3-l","range"]) # 删除无用标签
            json.dump(data2,open(os.path.join(dst,file.split(".")[0]+".json"),'w',encoding="utf-8"),ensure_ascii=False,indent=4)
        

if __name__ == "__main__":

    #singleMatch(path,"43.json",search_scale=50, ocr_flag=False)
    # showResult(path,["43.json"],dst="C:\\Users\\19586\\Downloads\\pairWZ\\result")
    mainProc(path, files, search_scale=100, ocr_flag=False)
    showResult(path, files, dst="C:\\Users\\19586\\Downloads\\pairWZ\\result")



