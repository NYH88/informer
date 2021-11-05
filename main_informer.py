import argparse
import datetime
import json
import os

import pandas as pd
import torch

from exp.exp_informer import Exp_Informer
from utils.visualization import *
from pyecharts.globals import CurrentConfig, OnlineHostType



parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='chicken', help='data them')
parser.add_argument('--root_path', type=str, default='./data/chicken/', help='数据文件的根路径（root path of the data file）')
# parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根路径（root path of the data file）')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_path', type=str, default='日均价.csv', help='data file')
parser.add_argument('--features', type=str, default='S', help='预测任务选项（forecasting task, options）:[M, S, MS]; '
                                                              'M:多变量预测多元（multivariate predict multivariate）, '
                                                              'S:单变量预测单变量（univariate predict univariate）, '
                                                              'MS:多变量预测单变量（multivariate predict univariate）')
parser.add_argument('--target', type=str, default='price', help='S或MS任务中的目标特征列名（target feature in S or MS task）')
parser.add_argument('--freq', type=str, default='d', help='时间特征编码的频率（freq for time features encoding）, '
                                                          '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
                                                          '你也可以使用更详细的频率，比如15分钟或3小时（you can also use more detailed freq like 15min or 3h）')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的位置（location of model checkpoints）')

# seq_len其实就是n个滑动窗口的大小，pred_len就是一个滑动窗口的大小
parser.add_argument('--seq_len', type=int, default=128, help='Informer编码器的输入序列长度（input sequence length of Informer encoder）原始默认为96')
parser.add_argument('--label_len', type=int, default=64, help='inform解码器的开始令牌长度（start token length of Informer decoder），原始默认为48')
parser.add_argument('--pred_len', type=int, default=30, help='预测序列长度（prediction sequence length）原始默认为24')
# pred_len就是要预测的序列长度（要预测未来多少个时刻的数据），也就是Decoder中置零的那部分的长度
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=1, help='编码器输入大小（encoder input size）')
parser.add_argument('--dec_in', type=int, default=1, help='解码器输入大小（decoder input size）')
parser.add_argument('--c_out', type=int, default=1, help='输出尺寸（output size）')
parser.add_argument('--d_model', type=int, default=512, help='模型维数（dimension of model）')
parser.add_argument('--n_heads', type=int, default=8, help='（num of heads）')
parser.add_argument('--e_layers', type=int, default=2, help='编码器层数（num of encoder layers）')
parser.add_argument('--d_layers', type=int, default=1, help='解码器层数（num of decoder layers）')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='堆栈编码器层数（num of stack encoder layers）')
parser.add_argument('--d_ff', type=int, default=2048, help='fcn维度（dimension of fcn）')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='是否在编码器中使用蒸馏，使用此参数意味着不使用蒸馏'
                                                           '（whether to use distilling in encoder, using this argument means not using distilling）', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='用于编码器的注意力机制，选项：[prob, full]'
                                                             '（attention used in encoder, options:[prob, full]）')

# 时间特征编码【未知】
parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码，选项：[timeF, fixed, learned]'
                                                               '（time features encoding, options:[timeF, fixed, learned]）')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力'
                                                                    '（whether to output attention in ecoder）')
parser.add_argument('--do_predict', action='store_true',default=True, help='是否预测看不见的未来数据'
                                                              '（whether to predict unseen future data）')
parser.add_argument('--mix', action='store_false', help='在生成解码器中使用混合注意力'
                                                        '（use mix attention in generative decoder）', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='将数据文件中的某些cols作为输入特性'
                                                        '（certain cols from the data files as the input features）')
parser.add_argument('--num_workers', type=int, default=0, help='工作的数据加载器数量'
                                                               'data loader num workers')
parser.add_argument('--itr', type=int, default=20, help='次实验'
                                                       'experiments times')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批大小'
                                                               'batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='提前停止的连续轮数'
                                                            'early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='实验描述'
                                                           'exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='校正的学习率'
                                                              'adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练'
                                                           'use automatic mixed precision training', default=False)
parser.add_argument('--output', type=str, default='./output',help='输出路径')


# 想要获得最终预测的话这里应该设置为True；否则将是获得一个标准化的预测。
parser.add_argument('--inverse', action='store_true', help='逆标准化输出数据'
                                                           'inverse output data', default=True)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

# 进行parser的变量初始化，获取实例。
args = parser.parse_args()

# 判断GPU是否能够使用，并获取标识
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 判断是否使用多块GPU，默认不使用多块GPU
if args.use_gpu and args.use_multi_gpu:
    # 获取显卡列表，type：str
    args.devices = args.devices.replace(' ','')
    # 拆分显卡获取列表，type：list
    device_ids = args.devices.split(',')
    # 转换显卡id的数据类型
    args.device_ids = [int(id_) for id_ in device_ids]
    # 获取第一块显卡
    args.gpu = args.device_ids[0]

# 初始化数据解析器，用于定义训练模式、预测模式、数据粒度的初始化选项。
"""
字典格式：{数据主题：{data：数据路径，'T':目标字段列名,'M'：，'S'：，'MS':}}

'M:多变量预测多元（multivariate predict multivariate）'，
'S:单变量预测单变量（univariate predict univariate）'，
'MS:多变量预测单变量（multivariate predict univariate）'。
"""
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'chicken':{'data':'日均价.csv','T':'price','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}


# 判断在parser中定义的数据主题是否在解析器中
if args.data in data_parser.keys():
    # 根据args里面定义的数据主题，获取对应的初始化数据解析器info信息，type：dict
    data_info = data_parser[args.data]
    # 获取该数据主题的数据文件的路径
    args.data_path = data_info['data']
    # 从数据解析器中获取 S或MS任务中的目标特征列名。
    args.target = data_info['T']
    # 从数据解析器中 根据变量features的初始化信息 获取 编码器输入大小，解码器输入大小，输出尺寸
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

# 堆栈编码器层数，type：list
args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
# 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

now_time = datetime.datetime.now().strftime('%mM_%dD %HH:%Mm:%Ss').replace(" ","_").replace(":","_")

# 构建单次运行的存储路径：
run_name_dir_old = args.model+"_"+data_parser["chicken"]["data"][0]+"_"+now_time+"_"+args.data
run_name_dir = os.path.join(args.output,run_name_dir_old)
if not os.path.exists(run_name_dir):
    os.makedirs(run_name_dir)
# 单次运行的n个实验的模型存储的路径
run_name_dir_ckp = os.path.join(args.checkpoints, run_name_dir_old)
# if not os.path.exists(run_name_dir_ckp):
#     os.makedirs(run_name_dir_ckp)

# 获取模型实例
Exp = Exp_Informer

# 存储整个实验的info信息
info_file = os.path.join(run_name_dir, "info_{}_{}.json".format(args.model, args.data))
info_dict = dict()

# 获取page实例
page_loss = get_page_loss(args.itr)
page_pt=get_page_value(args.itr)

# 预测未来的那段时间的真实值
true_month = [7.696,8.467,8.149,8.566,8.288,8.090]
# true_month = [8.467,8.149,8.566,8.288,8.090,6.747]
# len = 82
# true_date = [8.518,8.508,8.699,8.200,8.460,8.299,8.183,8.012,8.200,8.137,8.000,8.179,8.134,8.348,8.391,8.366,
#         8.600,8.599,8.199,8.486,8.300,8.308,8.299,8.220,8.,8.200,8.200,8.199,8.200,8.299,8.211,8.200,8.059,
#         8.100,8.212,8.200,8.200,8,8.220,8.199,8.084,7.999,7.915,8.126,8.000,7.918,8.155,8.127,8.048,7.918,
#         7.999,7.814,7.623,7.800,7.599,7.727,7.499,7.353,7.468,7.300,7.230,7.100,6.632,6.859,6.600,6.700,6.700,
#         6.878,6.302,6.243,6.260,6.640,6.683,6.656,6.274,6.199,6.473,6.167,6.048,6.299,6.011,5.972,
#         ]

true_date = [8.518,8.508,8.699,8.200,8.460,8.299,8.183,8.012,8.200,8.137,8.000,8.179,8.134,8.348,8.391,8.366,
        8.600,8.599,8.199,8.486,8.300,8.308,8.299,8.220,8.,8.200,8.200,8.199,8.200,8.299
        ]

pred_dates = []
# 用来保存预测数据的字典
data_dict = dict()
true = []

if data_parser["chicken"]["data"] == "日均价.csv" or data_parser["chicken"]["data"] == "周均价.csv":
    args.batch_size = 32
    true = true_date
    data_dict["true"] = true_date
else:
    args.batch_size = 8
    true = true_month
    data_dict["true"] = true_month

is_show_label_sign = data_parser["chicken"]["data"][0]

# 要进行多少次实验，一次实验就是完成一个模型的训练-测试-预测 过程。默认2次
for ii in range(args.itr):
    run_ex_dir = os.path.join(run_name_dir,"第_{}_次实验记录".format(ii+1))
    if not os.path.exists(run_ex_dir):
        os.makedirs(run_ex_dir)
    # 添加实验info
    info_dict["实验序号"] = ii
    info_dict["model"] = args.model
    info_dict["data_them"] = args.data
    info_dict["编码器的输入序列长度【滑动窗口大小】"] = args.seq_len
    info_dict["预测序列长度"] = args.pred_len
    info_dict["时间特征编码的频率【数据粒度】freq"] = args.freq
    info_dict["dorpout"] = args.dropout
    info_dict["batch_size"] = args.batch_size
    info_dict["损失函数loss"] = args.loss
    info_dict["提前停止的连续轮数patience"] = args.patience

    # 实验设置记录要点，方便打印，同时也作为文件名字传入参数，setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii)
    # 设置实验，将数据参数和模型变量传入实例
    exp = Exp(args) # set experiments

    # 训练模型
    print('>>>>>>>start training :  {}  >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    model,info_dict,all_epoch_train_loss,all_epoch_vali_loss,all_epoch_test_loss,epoch_count = exp.train(setting,info_dict,run_name_dir_ckp,run_ex_dir)

    # 模型测试
    print('>>>>>>>testing :  {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    info_dict,test_pred,test_true = exp.test(setting,info_dict,run_ex_dir)
    # print(test_pred)
    # print(test_true)

    future_pred, pred_date = 0,0
    # 做预测
    if args.do_predict:
        print('>>>>>>>predicting :  {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # 模型预测未来
        future_pred,pred_date = exp.predict(setting,run_name_dir_ckp,run_ex_dir, True)
        pred_dates = pred_date
        # print("未来的预测值：",future_pred)
        # print("未来的时间范围：",pred_date)


    # 存储实验的info信息：
    with open(info_file, mode='a',encoding='utf-8') as f:
        json.dump(info_dict, f, indent=4, ensure_ascii=False)
    # 存储数据：
    df = pd.DataFrame(data={"price":future_pred,"time":pred_date},columns=["price","time"])
    df.to_csv(os.path.join(run_ex_dir,"第{}次实验_未来预测结果.csv".format(ii+1)),index=False,encoding='utf-8')

    # 存储预测结果到字典
    data_dict["实验{}".format(ii+1)] = future_pred

    # 可视化：
    line_loss = chart_loss(all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count, run_ex_dir, args, ii + 1)
    # chart_predict(pred_date,future_pred,run_ex_dir,args,ii+1)
    line_pt = chart_predict_and_true(pred_date,future_pred,true,run_ex_dir,args,ii+1,is_show_label_sign)

    # 将图表加入page
    page_loss.add(line_loss)
    page_pt.add(line_pt)

    # 清除cuda的缓存
    torch.cuda.empty_cache()

#可视化page
page_loss.render(os.path.join(run_name_dir,"训练-验证-损失可视化-test.html"))
page_loss.save_resize_html(source=os.path.join(run_name_dir,"训练-验证-损失可视化-test.html"),
                           cfg_file=os.path.join('./output/',"chart_config.json"),
                           dest=os.path.join(run_name_dir,"训练-验证-损失可视化.html"))

page_pt.render(os.path.join(run_name_dir,"predict-true-test.html"))
page_pt.save_resize_html(source=os.path.join(run_name_dir,"predict-true-test.html"),
                           cfg_file=os.path.join('./output/',"chart_config.json"),
                           dest=os.path.join(run_name_dir,"predict-true.html"))


# 存储字典文件
data_dict["date"] = pred_dates
df = pd.DataFrame(data_dict)
df.to_csv(os.path.join(run_name_dir,"{}次实验预测结果.csv".format(args.itr)),index=False,encoding='utf-8',sep=',')




# https://blog.csdn.net/fluentn/article/details/115392229

# 输入格式：时间列名是date

"""
第一个任务：添加自定义评估指标函数 【ok】 和 存储预测结果函数 ；   难度评估：简单
第二个任务：添加自定义可视化函数；   难度评估：简单
第三个任务：修改源码对训练集、测试集进行预测之后将预测结果存储到本地；   难度评估：中等偏上

"""

























