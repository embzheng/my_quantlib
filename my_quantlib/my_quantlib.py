import pandas as pd

STRUCT_ID = {'None': 0, 'top_d': 1,'top_50': 2,'top_100': 3, 'bottom_d': -1,'bottom_50': -2,'bottom_100': -3}
STRUCT_NAME = {0:'None', 1:'top_d',2:'top_50',3:'top_100',-1:'bottom_d',-2:'bottom_50',-3:'bottom_100'}
    
'''
[获取前 begin 到 end (不包含end) 周期内的最低值]
@Param          [list]     [list]          [数据列表]
@Param          [begin]    [int]           [开始周期]
@Param          [end]      [int]           [结束周期]
'''	
def LLV(list, begin, end):
    count = len(list)
    if count == 0:
        return 0
    elif count == 1 or count <= begin:
        return list[-1]
        
    begin_id = count - begin
    end_id = count - end
    low = list[begin_id]
    
    #print("LLV begin_id %d end_id %d" % (begin_id, end_id))
    for i in list[begin_id:end_id]:
        if i < low:
            low = i
        #print("low %f i %f" % (low, i))
    return low
    
    
'''
[获取前 begin 到 end (不包含end) 周期内的最高值]
@Param          [list]     [list]          [数据列表]
@Param          [begin]    [int]           [开始周期]
@Param          [end]      [int]           [结束周期]
'''	
def HHV(list, begin, end):
    count = len(list)
    if count == 0:
        return 0
    elif count == 1 or count <= begin:
        return list[-1]
    
    begin_id = count - begin
    end_id = count - end
    hight = list[begin_id]
    
    #print("HHV begin_id %d end_id %d" % (begin_id, end_id))
    for i in list[begin_id:end_id]:
        if i > hight:
            hight = i
        #print("hight %f i %f" % (hight, i))
    return hight    
    
'''
[获取EMA]
@Param          [plist]     [list]          [数据列表]
@Param          [N]         [int]           [周期数]
@return                     [np.array]      [ema的列表]
'''	
def EMA(plist,N):
    ema_list = []
    ema = plist[0]
    ema_list.append(ema)
    for i in range(1,len(plist)):
        ema=(2*plist[i]+(N-1)*ema)/(N+1)
        ema_list.append(ema)
    return np.array(ema_list)   
    
'''
功能：MACD函数
用法：
        MACD( data['close']             )
        MACD( data['close'].tolist()    )
        MACD( np.array(data['close'])   )
输入：pd.Series、list、np.array
输出：
    (np.array, np.array, np.array)
    (DIFF, DEA, MACD)
'''  
def MACD(close_list, fastperiod=12, slowperiod=26, signalperiod=9):    
    fast = EMA(close_list, fastperiod)
    slow = EMA(close_list, slowperiod)
    DIF = fast - slow
    DEA  = EMA(DIF, signalperiod)
    MACD = 2*( DIF - DEA)
    return DIF, DEA, MACD   
    
'''
[获取股票移动平均值]
@Param          [stock]     [string]        [股票代码]
@Param          [days]      [int]           [天数]
@Param          [offset]    [int]           [偏移量。例如：当offset为1时，取昨天的移动平均值]
'''	
def MA(stock,days,offset = 0):
    offset = abs(int(offset))
    stockAvgList = attribute_history(security=stock,count=days+offset,unit='1d',fields=['close'])
    ma = stockAvgList.close[:days].mean()
    return ma

'''
[获取参数移动平均值 获取失败返回0]
@Param          [list]      [list]          [列表]
@Param          [days]      [int]           [天数]
@Param          [offset]    [int]           [偏移量。例如：当offset为1时，取昨天的移动平均值]
'''	
def LISTMA(list,days,offset = 0): 
    arrayin = np.array(list)
    offset = abs(int(offset))    
    count = days + offset
    if len(arrayin) < count:
        return 0
        
    array_tmp = arrayin[-count:]
    ma = array_tmp[:days].mean()
    # for dataa in array_tmp:
    #     print('dataa *** %f' % (dataa))  
    return ma    
        
'''
[获取股票成交量平均值]
@Param          [stock]     [string]        [股票代码]
@Param          [days]      [int]           [天数]
@Param          [offset]    [int]           [偏移量。例如：当offset为1时，取昨天的成交量平均值]
'''	
def MAVOL(stock,days,offset = 0):
    offset = abs(int(offset))
    stockAvgList = attribute_history(security=stock,count=days+offset,unit='1d',fields=['money'])
    mavol = stockAvgList.money[:days].mean()
    return mavol   

'''
随机指标KDJ一般是用于股票分析的统计体系，根据统计学原理，通过一个特定的周期（常为9日、9周等）内出现过的
最高价、最低价及最后一个计算周期的收盘价及这三者之间的比例关系，来计算最后一个计算周期的未成熟随机值RSV，
然后根据平滑移动平均线的方法来计算K值、D值与J值，并绘成曲线图来研判股票走势。
输出：字典(dict),键(key)为股票代码，值(value)为数据。
来源：聚宽函数库，作者joinquant-PM [回测专用]
修改部分：
kValue = np.array([SMA_CN(kValue[:x], slowk_period) for x in range(1, len(kValue) + 1)])
dValue = np.array([SMA_CN(kValue[:x], fastd_period) for x in range(1, len(kValue) + 1)])
'''
#注意代码要加上下面的import部分
import numpy as np
import talib
from functools import reduce
def KDJ(security_list, fastk_period=5, slowk_period=3, fastd_period=3) :
# KDJ指标计算，输入股票代码列表，默认参数5、3、3
    def SMA_CN(close, timeperiod) :
        close = np.nan_to_num(close)
        return reduce(lambda x, y: ((timeperiod - 1) * x + y) / timeperiod, close)

    # 修复传入为单只股票的情况
    if isinstance(security_list, str):
        security_list = [security_list]
    # 计算 KDJ
    n = max(fastk_period, slowk_period, fastd_period)
    k = {}; d = {}; j = {}
    for stock in security_list:

        security_data = attribute_history(stock, n*2,'1d',fields=['high', 'low', 'close'], df=False)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            print("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            k[stock] = np.nan
            d[stock] = np.nan
            j[stock] = np.nan
        else:
            high = security_data['high']
            low = security_data['low']
            close = security_data['close']
            kValue, dValue = talib.STOCHF(high, low, close, fastk_period, fastd_period, fastd_matype=0)
            #修改下面两行，以便在python3下运行
            kValue = np.array([SMA_CN(kValue[:x], slowk_period) for x in range(1, len(kValue) + 1)])
            dValue = np.array([SMA_CN(kValue[:x], fastd_period) for x in range(1, len(kValue) + 1)])
            jValue = 3 * kValue - 2 * dValue

            func = lambda arr : np.array([0 if x < 0 else (100 if x > 100 else x) for x in arr])

            k[stock] = func(kValue)
            d[stock] = func(dValue)
            j[stock] = func(jValue)
    return k, d, j

def get_max(a, b):
    if a > b:
        return a
    else:
        return b

def get_min(a, b):
    if a < b:
        return a
    else:
        return b

import datetime
def date_compare(date1, date2, fmt='%Y-%m-%d'):
    """
    比较两个真实日期之间的大小，date1 > date2 则返回True
    :param date1:
    :param date2:
    :param fmt:
    :return:
    """

    zero = datetime.datetime.fromtimestamp(0)

    try:
        d1 = datetime.datetime.strptime(str(date1), fmt)
    except:
        d1 = zero

    try:
        d2 = datetime.datetime.strptime(str(date2), fmt)
    except:
        d2 = zero
    return d1 > d2

def date_same(date1, date2, fmt='%Y-%m-%d'):
    """
    比较两个真实日期是否一样，date1 = date2 则返回True
    :param date1:
    :param date2:
    :param fmt:
    :return:
    """

    zero = datetime.datetime.fromtimestamp(0)

    try:
        d1 = datetime.datetime.strptime(str(date1), fmt)
    except:
        d1 = zero

    try:
        d2 = datetime.datetime.strptime(str(date2), fmt)
    except:
        d2 = zero
    return d1 == d2

# A1:=C>REF(C,4);
# A2:=C<REF(C,4);
# T1:=A2 AND REF(A1,1);
# T2:=A2 AND REF(T1,1);
# T3:=A2 AND REF(T2,1);
# T4:=A2 AND REF(T3,1);
# T5:=A2 AND REF(T4,1);
# T6:=A2 AND REF(T5,1);
# T7:=A2 AND REF(T6,1);
# T8:=A2 AND REF(T7,1);
# T9:=A2 AND REF(T8,1);
# T10:=A2 AND REF(T9,1);
def get_jiuzhuan_bottom_a1(close_list):
    if len(close_list) < 6:
        return False
    c1 = close_list[-1] 
    c2 = close_list[-2] 
    c5 = close_list[-5] 
    c6 = close_list[-6] 
    if c1 < c5 and c2 > c6:
        return True
    else:
        return False

def get_jiuzhuan_top_b1(close_list):
    if len(close_list) < 6:
        return False
    c1 = close_list[-1] 
    c2 = close_list[-2] 
    c5 = close_list[-5] 
    c6 = close_list[-6] 
    if c1 > c5 and c2 < c6:
        return True
    else:
        return False

def get_jiuzhuan_bottom(close_list):
    if len(close_list) < 15:
        return 0

    if get_jiuzhuan_bottom_a1(close_list):
        return 1
            
    for i in range(0, 9):
        c1 = close_list[-1 - i] 
        c5 = close_list[-5 - i]
        if  c1 < c5:
            if get_jiuzhuan_bottom_a1(close_list[:-1 - i]):
                return i + 2
        else:
            return 0
    return 0

def get_jiuzhuan_top(close_list):
    if len(close_list) < 15:
        return 0

    if get_jiuzhuan_top_b1(close_list):
        return 1
            
    for i in range(0, 9):
        c1 = close_list[-1 - i] 
        c5 = close_list[-5 - i]
        if  c1 > c5:
            if get_jiuzhuan_top_b1(close_list[:-1 - i]):
                return i + 2
        else:
            return 0
    return 0

'''
功能：获取九转序列函数
用法：
        JIUZHUAN( data['close']             )
        JIUZHUAN( data['close'].tolist()    )
        JIUZHUAN( np.array(data['close'])   )
输入：pd.Series、list、np.array
输出：-10 ~ +10
    0 没有序列
    小于0 底部序列
    大于0 顶部序列
''' 
def JIUZHUAN(close_list):
    top = get_jiuzhuan_top(close_list)
    bottom = get_jiuzhuan_bottom(close_list)

    if top != 0:
        return top
    if bottom != 0:
        return -bottom
    return 0

'''
功能：获取趋势仓位
输入：high_list, low_list, close_list, current_cangwei 当前仓位
输出：计算后的最新仓位0 ~ 10
''' 
#根据总仓位获取长短线仓位
def get_ls_position(current_position):
    if current_position == 10:
        current_position_l = 6    
        current_position_s = 4
    elif current_position == 6:
        current_position_l = 6    
        current_position_s = 0
    elif current_position == 4:
        current_position_l = 0    
        current_position_s = 4
    elif current_position == 0:
        current_position_l = 0    
        current_position_s = 0
    else:
        current_position_s = 0
        current_position_l = 0
        print("仓位异常 %.2f" % (current_position)) 
    return current_position_s, current_position_l

def QS(high_list, low_list):
    qs = {'s_h':0, 's_l':0, 'l_h':0, 'l_l':0} 
    if len(high_list) < 89:
        return qs
    
    #计算趋势线
    EMA26_H = EMA(high_list, 26)
    EMA26_L = EMA(low_list, 26)    
    EMA89_H = EMA(high_list, 89)
    EMA89_L = EMA(low_list, 89)
    qs['s_h'] = EMA26_H[-1]
    qs['s_l'] = EMA26_L[-1]
    qs['l_h'] = EMA89_H[-1]
    qs['l_l'] = EMA89_L[-1]
    return qs

######################################################## MACD结构 ########################################################
#获取前 begin 到 end (不包含end) 周期内macd为正的周期数
def get_macd_plus_count( macd, begin, end):
    
    count = len(macd)
    begin_id = count - begin
    end_id = count - end
    count = 0 
    #print("begin_id %d end_id %d" % (begin_id, end_id))
    for i in macd[begin_id:end_id]:
        #print("count %d i %f" % (count, i))
        if i > 0:
            count += 1
    return count
    
#获取前 begin 到 end (不包含end) 周期内macd为负的周期数
def get_macd_minus_count( macd, begin, end):
    count = len(macd)
    begin_id = count - begin
    end_id = count - end
    count = 0 
    for i in macd[begin_id:end_id]:
        if i < 0:
            count += 1
        #print("count %d i %f" % (count, i))
    return count  

#获取上一次金叉到ref周期前的周期数，ref=0代表当前周期，ref=5，即上一次金叉到5周期前的周期数
def get_barlast_gold(macd_dif, macd_dea, ref=0):
    count = len(macd_dif)
    last_id = count - ref
    i = last_id #list中索引包含头，不包含尾，比如[1:5]指第二个到第五个
    #print("count %d last_id %d" % (count, last_id))
    while i >= 0:
        if get_macd_gold(macd_dif[:i], macd_dea[:i]):
            #print("金叉 i %d" % (i))
            break
        else:
            i -= 1
    return last_id - i + 1
    
def get_barlast_dead(macd_dif, macd_dea, ref=0):
    count = len(macd_dif)
    last_id = count - ref
    i = last_id #list中索引包含头，不包含尾，比如[1:5]指第二个到第五个
    #print("count %d last_id %d" % (count, last_id))
    while i >= 0:
        if get_macd_dead(macd_dif[:i], macd_dea[:i]):
            #print("死叉 i %d" % (i))
            break
        else:
            i -= 1
    return last_id - i + 1  
    
def get_macd_gold( macd_dif, macd_dea):
    '''
    判断是否 MACD 金叉
    return 1 or 0
    '''
    #print("macd_dif[-1] %f macd_dea[-1] %f" % (macd_dif[-1], macd_dea[-1]))
    #print("macd_dif[-2] %f macd_dea[-2] %f" % (macd_dif[-2], macd_dea[-2]))
    if len(macd_dif) < 2:
        return 0
        
    if macd_dif[-2] < macd_dea[-2] and macd_dif[-1] >= macd_dea[-1]:
        return 1
    else:
        return 0

def get_dif_gold( macd_dif):
    '''
    判断dif拐头向上
    return 1 or 0
    '''
    #print("macd_dif[-1] %f macd_dea[-1] %f" % (macd_dif[-1], macd_dea[-1]))
    #print("macd_dif[-2] %f macd_dea[-2] %f" % (macd_dif[-2], macd_dea[-2]))
    if len(macd_dif) < 2:
        return 0
        
    if macd_dif[-1] > macd_dif[-2]:
        return 1
    else:
        return 0

def get_macdval_gold( macd_macd):
    '''
    判断macd能量柱变大
    return 1 or 0
    '''
    if len(macd_macd) < 2:
        return 0
        
    if macd_macd[-1] > macd_macd[-2]:
        return 1
    else:
        return 0
    
def get_macd_dead( macd_dif, macd_dea):
    '''
    判断是否 MACD 死叉
    return 1 or 0
    '''
    #print("macd_dif[-1] %f macd_dea[-1] %f" % (macd_dif[-1], macd_dea[-1]))
    if len(macd_dif) < 2:
        return 0
    
    if macd_dif[-2] > macd_dea[-2] and macd_dif[-1] <= macd_dea[-1]:
        return 1
    else:
        return 0

def get_dif_dead( macd_dif):
    '''
    判断dif拐头向上
    return 1 or 0
    '''
    #print("macd_dif[-1] %f macd_dea[-1] %f" % (macd_dif[-1], macd_dea[-1]))
    #print("macd_dif[-2] %f macd_dea[-2] %f" % (macd_dif[-2], macd_dea[-2]))
    if len(macd_dif) < 2:
        return 0
        
    if macd_dif[-1] < macd_dif[-2]:
        return 1
    else:
        return 0

def get_macdval_dead( macd_macd):
    '''
    判断能量柱减少
    return 1 or 0
    '''
    if len(macd_macd) < 2:
        return 0
        
    if macd_macd[-1] < macd_macd[-2]:
        return 1
    else:
        return 0    


'''
功能：获取顶部结构
输入：close_list, high_list, high_list可为空
输出：struct = {'struct_type':'None', 'state':'None', 'is_multiple':False, 'ch1':0} 
        struct_type:top_d(顶部钝化)，top（顶部结构）
        state:100（结构形成100%），50（结构形成50%，即dif拐头但未死叉）
        is_multiple：是否多重结构
        ch1：结构形成时的价格创新高的这个新高值
''' 
def TOP_STRUCT( close_list, macd_dif, macd_dea, macd_macd):
    '''
    输入close_list
    输出 MacdStruct 实例
    '''
    struct = {'struct_type':'None', 'state':'None', 'is_multiple':False, 'ch1':0} 
    if len(close_list) < 10:
        return struct
    
    # 获取金叉死叉区间
    n1 = get_barlast_gold(macd_dif, macd_dea, 0)
    n2 = get_barlast_gold(macd_dif, macd_dea, n1)
    n3 = get_barlast_gold(macd_dif, macd_dea, n2 + n1)
    n1_l = n1
    n2_l = n1 + n2
    n3_l = n1 + n2 + n3

    d1 = get_barlast_dead(macd_dif, macd_dea, 0)
    d2 = get_barlast_dead(macd_dif, macd_dea, d1)  
    if d1 < n1:
        d3 = get_barlast_dead(macd_dif, macd_dea, d2 + d1)
        d1 = d2
        d2 = d3
    d1_l = d1 # 死叉到当前周期数，死叉当周期算1
    d2_l = d1 + d2
    
    #print("n1 %d, n2 %d, n3 %d" % (n1, n2, n3))
    
    ch1 = HHV(close_list, n1_l, 0)
    ch2 = HHV(close_list, n2_l, d1_l)
    ch3 = HHV(close_list, n3_l, d2_l)
    #print("ch1 %f, ch2 %f, ch3 %f" % (ch1, ch2, ch3))
    
    difh1 = HHV(macd_dif, n1_l, 0)
    difh2 = HHV(macd_dif, n2_l, d1_l)
    difh3 = HHV(macd_dif, n3_l, d2_l)
    #print("difh1 %f, difh2 %f, difh3 %f" % (difh1, difh2, difh3))
    
    macd_plus_count1 = get_macd_plus_count(macd_macd, n1_l, 0)
    macd_plus_count2 = get_macd_plus_count(macd_macd, n2_l, n1_l)
    macd_plus_count3 = get_macd_plus_count(macd_macd, n3_l, n2_l)
    #print("macd_plus_count1 %d, macd_plus_count2 %d, macd_plus_count3 %d" % (macd_plus_count1, macd_plus_count2, macd_plus_count3))
    
    macd_minus_count1 = get_macd_minus_count(macd_macd, n1_l, 0)
    macd_minus_count2 = get_macd_minus_count(macd_macd, n2_l, n1_l)
    macd_minus_count3 = get_macd_minus_count(macd_macd, n3_l, n2_l)
    #print("macd_minus_count1 %d, macd_minus_count2 %d, macd_minus_count3 %d" % (macd_minus_count1, macd_minus_count2, macd_minus_count3))
    
    if (((ch1 > ch2 and difh1 < difh2) or (ch1 > get_max(ch3, ch2) and difh1 < get_max(difh3, difh2)))
        and macd_plus_count1 >= 2 # 至少两个红角线才有钝化，结构之前必有钝化
        and macd_dif[-1] > 0): # 自定义：钝化和结构的时候dif必须是正的
        #print("macd_dif[-1] %f macd_dif[-2] %f" % (macd_dif[-1], macd_dif[-2]))
        # 结构形成 
        if macd_minus_count1 == 0: # 还未死叉
            if macd_plus_count1 == 2:
                struct['struct_type'] = 'top_d'
                struct['ch1'] = ch1
            else:
                if macd_dif[-1] < macd_dif[-2]:
                    struct['struct_type'] = 'top_50'   
                    struct['ch1'] = ch1
                else:
                    struct['struct_type'] = 'top_d'
                    struct['ch1'] = ch1
        elif macd_minus_count1 == 1: # 刚刚死叉
            struct['struct_type'] = 'top_100'   
            struct['ch1'] = ch1  
        #else:
            #已经死叉过
    return struct

'''
功能：获取底部结构
输入：close_list, low_list, low_list可为空
输出：struct = {'struct_type':'None', 'state':'None', 'is_multiple':False, 'ch1':0} 
        struct_type:bottom_d(底部钝化)，bottom（底部结构）
        state:100（结构形成100%），50（结构形成50%，即dif拐头但未金叉）
        is_multiple：是否多重结构
        ch1：结构形成时的价格创新低的这个新低值
''' 
def BOTTOM_STRUCT(close_list, macd_dif, macd_dea, macd_macd):
    struct = {'struct_type':'None', 'state':'None', 'is_multiple':False, 'ch1':0} 

    if len(close_list) < 10:
        return struct
    
    # 获取股票的收盘价
    d1 = get_barlast_dead(macd_dif, macd_dea, 0)
    d2 = get_barlast_dead(macd_dif, macd_dea, d1)
    d3 = get_barlast_dead(macd_dif, macd_dea, d2 + d1)
    d1_l = d1
    d2_l = d1 + d2
    d3_l = d1 + d2 + d3
    #print("d1 %d, d2 %d, d3 %d" % (d1, d2, d3))
    n1 = get_barlast_gold(macd_dif, macd_dea, 0)
    n2 = get_barlast_gold(macd_dif, macd_dea, n1)    
    if n1 < d1:
        n3 = get_barlast_gold(macd_dif, macd_dea, n2 + n1)
        n1 = n2
        n2 = n3
    n1_l = n1
    n2_l = n1 + n2
    
    cl1 = LLV(close_list, d1_l, 0)
    cl2 = LLV(close_list, d2_l, n1_l)
    cl3 = LLV(close_list, d3_l, n2_l)
    #print("cl1 %f, cl2 %f, cl3 %f" % (cl1, cl2, cl3))
    
    difl1 = LLV(macd_dif, d1_l, 0)
    difl2 = LLV(macd_dif, d2_l, n1_l)
    difl3 = LLV(macd_dif, d3_l, n2_l)
    #print("difl1 %f, difl2 %f, difl3 %f" % (difl1, difl2, difl3))
    
    macd_plus_count1 = get_macd_plus_count(macd_macd, d1_l, 0)
    macd_plus_count2 = get_macd_plus_count(macd_macd, d2_l, d1_l)
    macd_plus_count3 = get_macd_plus_count(macd_macd, d3_l, d2_l)
    #print("macd_plus_count1 %d, macd_plus_count2 %d, macd_plus_count3 %d" % (macd_plus_count1, macd_plus_count2, macd_plus_count3))
    
    macd_minus_count1 = get_macd_minus_count(macd_macd, d1_l, 0)
    macd_minus_count2 = get_macd_minus_count(macd_macd, d2_l, d1_l)
    macd_minus_count3 = get_macd_minus_count(macd_macd, d3_l, d2_l)
    #print("macd_minus_count1 %d, macd_minus_count2 %d, macd_minus_count3 %d" % (macd_minus_count1, macd_minus_count2, macd_minus_count3))
    
    #print("macd_dif[-1] %f macd_dif[-2] %f" % (macd_dif[-1], macd_dif[-2]))
    if (((cl1 < cl2 and difl1 > difl2) or (cl1 < get_min(cl3, cl2) and difl1 > get_min(difl3, difl2)))
        and macd_minus_count1 >= 2 # 至少两个绿角线才有钝化，结构之前必有钝化
        and macd_dif[-1] < 0): # 自定义：钝化和结构的时候dif必须是负的
        #print("macd_dif[-1] %f macd_dif[-2] %f" % (macd_dif[-1], macd_dif[-2]))
        # 结构形成 
        if macd_plus_count1 == 0: # 还未金叉
            if macd_minus_count1 == 2:
                struct['struct_type'] = 'bottom_d'
                struct['ch1'] = cl1
            else:
                if macd_dif[-1] > macd_dif[-2]:
                    struct['struct_type'] = 'bottom_50'   
                    struct['ch1'] = cl1
                else:
                    struct['struct_type'] = 'bottom_d'
                    struct['ch1'] = cl1
        elif macd_plus_count1 == 1: # 刚刚金叉
            struct['struct_type'] = 'bottom_100'   
            struct['ch1'] = cl1  
        #else:
            #已经金叉过
    return struct

'''
功能：获取定量结构
输入：close_list
输出：struct = {'struct_type':'None', 'state':'None', 'is_multiple':False, 'ch1':0} 
        struct_type:bottom_d(底部钝化)，bottom（底部结构）
        state:100（结构形成100%），50（结构形成50%，即dif拐头但未金叉）
        is_multiple：是否多重结构
        ch1：结构形成时的价格创新低的这个新低值
''' 
def MACD_STRUCT(close_list):
    macd_dif, macd_dea, macd_macd = MACD(close_list)
    struct = TOP_STRUCT(close_list, macd_dif, macd_dea, macd_macd)
    if struct['struct_type'] == 'None':
        struct = BOTTOM_STRUCT(close_list, macd_dif, macd_dea, macd_macd) 
    return struct

class StructStat:
    def __init__(self, stock, type):
        self.stock = stock
        self.type = type #哪种类型数据 '1m' 代表一分钟
        self.top_50     = False
        self.top_100    = False
        self.bottom_50  = False
        self.bottom_100 = False
        self.top_50_count     = 0
        self.top_100_count    = 0
        self.bottom_50_count  = 0
        self.bottom_100_count = 0

        self.top_50_fail_count     = 0
        self.top_100_fail_count    = 0
        self.bottom_50_fail_count  = 0
        self.bottom_100_fail_count = 0

        self.top_count_tmp      = 0
        self.bottom_count_tmp   = 0
        self.top_zz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 暂定最大失败10次，第一个代表成功一次就失败的次数，第二个代表成功两次就失败的次数
        self.bottom_zz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 暂定最大失败10次，第一个代表成功一次就失败的次数，第二个代表成功两次就失败的次数
        
'''
功能：获取结构形成后的将金不金信号
输入：close_list，ref（查询范围，如默认值30，代表从当前周期往前30个周期范围内）
输出：True/False
'''    
def MACD_STRUCT_JJBJ(close_list, struct):
    if len(close_list) < 10:
        return False

    macd_dif, macd_dea, macd_macd = MACD(close_list)    
    # 获取金叉死叉区间
    g1 = get_barlast_gold(macd_dif, macd_dea, 0) 
    d1 = get_barlast_dead(macd_dif, macd_dea, 0)   
    #print("n1 %d, n2 %d, n3 %d" % (n1, n2, n3))

    ch_gold = HHV(close_list, g1, 0)
    ch_dead = HHV(close_list, d1, 0)    

    if struct['struct_type'] != 'top_100':
        return False 
    if d1 > 20 or d1 < 3:
        return False
    if  ch_gold != ch_dead:
        return False   
    if macd_macd[-1] > 0:
        return False   
    if macd_dif[-2] > macd_dif[-3] and macd_dif[-1] < macd_dif[-2]:
        return True        
    return False

'''
功能：获取结构形成后的将死不死信号
输入：close_list，ref（查询范围，如默认值30，代表从当前周期往前30个周期范围内）
输出：True/False
'''    
def MACD_STRUCT_JSBS(close_list, struct):
    if len(close_list) < 10:
        return False

    macd_dif, macd_dea, macd_macd = MACD(close_list)    
    # 获取金叉死叉区间
    g1 = get_barlast_gold(macd_dif, macd_dea, 0) 
    d1 = get_barlast_dead(macd_dif, macd_dea, 0)   
    #print("n1 %d, n2 %d, n3 %d" % (n1, n2, n3))

    ch_gold = LLV(close_list, g1, 0)
    ch_dead = LLV(close_list, d1, 0)    

    if struct['struct_type'] != 'bottom_100':
        return False 
    if g1 > 20 or g1 < 3:
        return False
    if  ch_gold != ch_dead:
        return False   
    if macd_macd[-1] < 0:
        return False  
    if macd_dif[-2] < macd_dif[-3] and macd_dif[-1] > macd_dif[-2]:
        return True        
    return False
        
def MACD_STRUCT_SIGNAL(close_list):
    struct = BARLAST_MACD_STRUCT(close_list)
    jjbj = MACD_STRUCT_JJBJ(close_list, struct) 
    jsbs = MACD_STRUCT_JSBS(close_list, struct) 
    if jjbj:
        return(-1) 
    elif jsbs:
        return(1) 
    else:
        return(0) 
'''
功能：获取上一次定量结构到现在的周期数
输入：close_list，ref（查询范围，如默认值30，代表从当前周期往前30个周期范围内）
输出：上一次定量结构到现在的周期数对应的结构，当前周期结构周期为0
'''         
        
def BARLAST_MACD_STRUCT(close_list, ref=24):
    struct_none = {'struct_type':'None', 'state':'None', 'barlast':-1, 'is_multiple':False, 'ch1':0} 
    #macd至少要10个周期
    if len(close_list) < 40:
        return struct_none    
    
    # 代表没有钝化和结构
    has_struct_none = False
    struct_50 = None
    struct_100 = None
    macd_dif, macd_dea, macd_macd = MACD(close_list)
    
    for i in range(0, ref + 1):      #需要找出结构前的钝化，所以需要加1
        if i == 0:
            struct = MACD_STRUCT(close_list)
            struct['barlast'] = -1
            struct_current = struct
        else:
            struct = MACD_STRUCT(close_list[:-i])
        #判断结构有效型
        if struct['struct_type'] == 'top_d':
            #print("顶部钝化 i %d" %(i))
            #钝化之后有结构
            if struct_100 != None:
                struct_100['barlast'] = i - 1
                return struct_100
            elif struct_50 != None:
                struct_50['barlast'] = i - 1
                return struct_50
            else:
                if has_struct_none == False:
                    struct['barlast'] = i - 1
                    return struct_current
                else:
                    struct['barlast'] = -1
                    return struct_current
        elif struct['struct_type'] == 'bottom_d':
            #print("底部钝化 i %d" %(i))
            #钝化之后有结构
            if struct_100 != None:
                struct_100['barlast'] = i - 1
                return struct_100
            elif struct_50 != None:
                struct_50['barlast'] = i - 1
                return struct_50
            else:
                if has_struct_none == False:
                    struct['barlast'] = i - 1
                    return struct_current
                else:
                    struct['barlast'] = -1
                    return struct_current

        elif struct['struct_type'] == 'top_50':            
            #顶部结构之后必须没有金叉
            barlast_gold = get_barlast_gold(macd_dif, macd_dea)
            if barlast_gold > i:
                has_struct_50 = True     
                struct_50 = struct
            else:
                print("50结构失败%s %s barlast_gold %d i %d" %(struct['struct_type'], struct['state'], barlast_gold, i))
       
        elif struct['struct_type'] == 'top_100':            
            #顶部结构100之后必须没有金叉
            barlast_gold = get_barlast_gold(macd_dif, macd_dea)
            if barlast_gold > i:
                has_struct_100 = True
                struct_100 = struct
            else:
                print("100结构失败%s %s barlast_gold %d i %d" %(struct['struct_type'], struct['state'], barlast_gold, i))
        elif struct['struct_type'] == 'bottom_50':            
            #结构之后100必须没有死叉
            barlast_dead = get_barlast_dead(macd_dif, macd_dea)
            if barlast_dead > i:
                has_struct_50 = True 
                struct_50 = struct
            else:
                print("结构失败%s %s barlast_dead %d i %d" %(struct['struct_type'], struct['state'], barlast_dead, i))                           
        elif struct['struct_type'] == 'bottom_100':           
            #结构之后100必须没有死叉
            barlast_dead = get_barlast_dead(macd_dif, macd_dea)
            if barlast_dead > i:
                has_struct_100 = True
                struct_100 = struct
            else:
                print("100结构失败%s %s barlast_dead %d i %d" %(struct['struct_type'], struct['state'], barlast_dead, i))                           
        else:
            has_struct_none = True
            #print("没有结构%s %s" %(struct['struct_type'], struct['state']))
                
    #没有找到结构，返回当前结构情况
    return (struct_current)

# 获取结构列表
def MACD_STRUCT_LIST(close_list, list_len = 30):  
    if list_len == 0:
        return None 
    struct_list = []
    for i in range(list_len):
        if i == 0:                
            struct = MACD_STRUCT(close_list)
            struct_list.append(struct)
        else:
            struct = MACD_STRUCT(close_list[:-i])
            struct_list.append(struct)
    #对列表的元素进行反向排序
    struct_list.reverse()
    return struct_list

def date_same(date1, date2):
    cmp = date1 - date2
    if cmp.days == 0 and cmp.seconds == 0:
        return True
    else:
        return False    


import threading
import queue

# 每个任务线程
class WorkThread(threading.Thread):
    def __init__(self, work_queue):
        super().__init__()
        self._stop_event = threading.Event()
        self.work_queue = work_queue
        self.daemon = True

    def run(self):
        while True:
            func, *args = self.work_queue.get()
            self._stop_event.clear()
            func(*args)
            self.stop()
            self.work_queue.task_done()
    
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

# 线程池
class ThreadPoolManger():
    def __init__(self, thread_number):
        #线程锁
        self.lock = threading.Lock()
        self.thread_number = thread_number
        self.work_queue = queue.Queue()
        self.thread_list = []
        for i in range(self.thread_number):     # 生成一些线程来执行任务
            thread = WorkThread(self.work_queue)
            thread.start()
            self.thread_list.append(thread)

    def thread_lock(self):
        self.lock.acquire()

    def thread_unlock(self):
        self.lock.release()

    def add_work(self, func, *args):
        self.work_queue.put((func, *args))  
    
    def stop_all_work(self):
        for t in self.thread_list:
            if not t.stopped():
                t.stop()          


'''
根据时间，获取分时列表
'''
def get_market_fenshi(dt_new):
    fenshi_list = ['1m']
    if dt_new.minute % 5 == 0:
        fenshi_list.append('5m')
    if dt_new.minute % 15 == 0:
        fenshi_list.append('15m')
    if dt_new.minute % 30 == 0:
        fenshi_list.append('30m')
    if dt_new.hour == 10 and dt_new.minute == 30:
        fenshi_list.append('60m')
    elif dt_new.hour == 11 and dt_new.minute == 30:
        fenshi_list.append('60m')
        fenshi_list.append('120m')
    elif dt_new.hour == 14 and dt_new.minute == 0:
        fenshi_list.append('60m')
    elif dt_new.hour == 15 and dt_new.minute == 0:
        fenshi_list.append('60m')
        fenshi_list.append('120m')
        fenshi_list.append('1d')
    return fenshi_list                

# 获取趋势仓位
# 输入：包含 close、high、low 的df
# 输出：仓位列表
def GET_XU_POS_QUSHI(src_data_df):
    src_data_df['ema26h'] = talib.EMA(src_data_df['high'], 26)
    src_data_df['ema26l'] = talib.EMA(src_data_df['low'], 26)    
    src_data_df['ema89h'] = talib.EMA(src_data_df['high'], 89)
    src_data_df['ema89l'] = talib.EMA(src_data_df['low'], 89)
    # 填充大值，确保无数据时仓位为0
    src_data_df['ema26h'].fillna(100000, inplace=True)
    src_data_df['ema26l'].fillna(100000, inplace=True)
    src_data_df['ema89h'].fillna(100000, inplace=True)
    src_data_df['ema89l'].fillna(100000, inplace=True)
    #print(src_data_df)
    position = []
    for i in range(src_data_df.shape[0]):
        if i == 0:
            position.append(0)
            continue
        #print(src_data_df['ema89h'][i])
        current_position_s, current_position_l = get_ls_position(position[-1]*10)
        current_price = src_data_df['close'][i]
        position_s = current_position_s
        position_l = current_position_l
        if current_price >= src_data_df['ema26h'][i]: #短上轨
            position_s = 4
        elif current_price < src_data_df['ema26l'][i]: #短下轨
            position_s = 0        
    
        if current_price >= src_data_df['ema89h'][i]: #长上轨
            position_l = 6
        elif current_price < src_data_df['ema89l'][i]: #长下轨
            position_l = 0
        position.append((position_s+position_l)/10)   
    
    position_dict = {'pos_qushi':position}
    position_df = pd.DataFrame(position_dict,index=src_data_df.index)
    position_df = pd.concat([src_data_df, position_df], axis=1) # 将signal_df和threshold_df进行合并    
    return position_df

# 总策略：趋势为王，结构修边
# 趋势仓位：
# 一、0成 ，股价在长短期趋势以下
# 修边原则：
# 离短期趋势近，60分钟及以上底部结构，买4层，多个结构，以大周期为准 
# 结构失效原则：
#     50的结构：钝化消失
#     100的结构：有效期结束、短期趋势突破、结构消失（dif小于dea）
# 二、4成 ，股价在短期结构上，长期结构下
# 修边原则：
# 60分钟及以上顶部结构，减4层，多个结构，以大周期为准 
# 结构失效原则：
#     50的结构：钝化消失
#     100的结构：有效期结束、短期趋势破位、结构消失（dif大于dea）
# 三、6成 ，股价在长期结构上，短期结构下
# 修边原则：
# 60分钟及以上底部结构，买4层，多个结构，以大周期为准 
# 结构失效原则：
#     50的结构：钝化消失
#     100的结构：有效期结束、短期趋势突破、结构消失（dif小于dea）
# 四、10成 
# 修边原则：
# 离短期趋势近，60分钟及以上顶部结构，减4层，多个结构，以大周期为准 
# 结构失效原则：
#     50的结构：钝化消失
#     100的结构：有效期结束、短期趋势破位、结构消失（dif大于dea）
# curr 当前仓位，满仓10
# struct 结构类型，传ID（STRUCT_NAME可以转换）
def GET_XU_POS_VAL(curr, close, sh, sl, lh, ll, struct=0, struct_barlast=-1, jiuzhuang=0, struct_valid_num=24):    
    current_position_s, current_position_l = get_ls_position(curr)
    current_price = close  

    #仓位策略    
    #print("当前 短线仓位 %.2f 长线仓位 %.2f" % (current_position_s, current_position_l))
    position_s = current_position_s
    position_l = current_position_l
    if current_price >= sh: #短上轨
        position_s = 4
    elif current_price < sl: #短下轨
        position_s = 0    
    
    #长期趋势加仓的前提是短期趋势已突破，且股价大于短上轨
    # if current_price >= EMA89_H[-1] and current_position_s != 0 and current_price >= EMA26_H[-1]: #长上轨
    #     position_l = 6
    # #长期趋势减仓的前提是短期趋势已破位，且股价小于短下轨
    # elif current_price < EMA89_L[-1] and current_position_s == 0 and current_price < EMA26_L[-1]: #长下轨
    #     position_l = 0
    if current_price >= lh: #长上轨
        position_l = 6
    elif current_price < ll: #长下轨
        position_l = 0
    print("收盘价 %6.2f [短趋势(%d):%8.2f %8.2f 长趋势(%d):%8.2f %8.2f]" 
        % (current_price, position_s, sh, sl, position_l, lh, ll))
        
    cangwei_qs = position_s + position_l
    cangwei_current = cangwei_qs
    #macd结构俢边 
    #结构在有效周期内
    cangwei_xiubian = 0             
    if struct_barlast >= 0 and struct_barlast < struct_valid_num:
        #当前周期形成结构
        if STRUCT_NAME[struct] == 'top_50' or STRUCT_NAME[struct] == 'top_100':
            cangwei_xiubian = -4
        elif STRUCT_NAME[struct] == 'bottom_50' or STRUCT_NAME[struct] == 'bottom_100':
            cangwei_xiubian = 4      
    
    #计算俢边仓位
    if cangwei_xiubian < 0:
        if cangwei_current == 10:            
            cangwei_current -= 10
        else:
            cangwei_current = 0
    elif cangwei_xiubian > 0:
        if cangwei_current == 0:            
            cangwei_current += 10
        else:
            cangwei_current = 10
              
    print("结构:%10s 趋势仓位:%d 最终仓位:%d" %(STRUCT_NAME[struct], cangwei_qs, cangwei_current))                                
    return cangwei_current
