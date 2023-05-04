import numpy as np
import pandas as pd
import math

df = pd.read_csv('CardioPhenoLarge.csv')
df.fillna('0', inplace=True)


def clean(x):
    if x != '0':
        x = x.replace("'", " ").strip('[').strip(']').strip().replace(' ', '').split(',')
    return x


df['p41270'] = df['p41270'].apply(clean)  # clean list of icd10 codes from csv format
df['listICD'] = [','.join(map(str, l)) for l in df['p41270']]  # convert list of idc10 codes to a string

HeartDiseaseCodesAll = \
        ['I00', 'I010', 'I011', 'I018', 'I019', 'I020', 'I029', 'I050', 'I051', 'I052', 'I058', 'I059', 'I060', 'I061',
         'I062', 'I068', 'I069', 'I070', 'I071', 'I072', 'I078', 'I079', 'I080', 'I081', 'I083', 'I088', 'I089',
         'I090', 'I091', 'I098', 'I099',
         'I10', 'I110', 'I119', 'I120', 'I129', 'I130', 'I131', 'I132', 'I139', 'I150', 'I151', 'I152', 'I158', 'I159',
         'I200', 'I201', 'I208', 'I209', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219', 'I220', 'I221', 'I228', 'I229',
         'I230', 'I231', 'I232', 'I233', 'I235', 'I236', 'I238', 'I240', 'I241', 'I248', 'I249',
         'I250', 'I251', 'I252', 'I253', 'I254', 'I255', 'I256', 'I258', 'I259',
         'I260', 'I269', 'I270', 'I271', 'I272', 'I278', 'I279', 'I280', 'I281', 'I288', 'I289',
         'I300', 'I301', 'I308', 'I309', 'I310', 'I312', 'I313', 'I318', 'I319', 'I320', 'I321', 'I328', 'I330', 'I339',
         'I340', 'I341', 'I342', 'I348', 'I349', 'I350', 'I352', 'I358', 'I359', 'I360', 'I361', 'I362', 'I368', 'I369',
         'I370', 'I371', 'I372', 'I378', 'I379', 'I38', 'I390', 'I391', 'I393', 'I394', 'I398', 'I400', 'I401', 'I408',
         'I409', 'I410', 'I411', 'I412', 'I418', 'I420', 'I421', 'I422', 'I423', 'I424', 'I424', 'I425', 'I426', 'I427',
         'I428', 'I429', 'I430', 'I431', 'I432', 'I438', 'I440', 'I441', 'I442', 'I443', 'I444', 'I445', 'I446', 'I447',
         'I450', 'I451', 'I452', 'I453', 'I454', 'I455', 'I456', 'I458', 'I459', 'I460', 'I461', 'I469',
         'I470', 'I471', 'I472', 'I479', 'I480', 'I481', 'I482', 'I483', 'I484', 'I489',
         'I490', 'I491', 'I492', 'I493', 'I494', 'I495', 'I498', 'I499', 'I500', 'I501', 'I509',
         'I510', 'I511', 'I512', 'I513', 'I514', 'I515', 'I516', 'I517', 'I518', 'I519', 'I528',
         'I600', 'I601', 'I602', 'I603', 'I604', 'I605', 'I605', 'I606', 'I607', 'I608', 'I609',
         'I610', 'I611', 'I612', 'I613', 'I614', 'I615', 'I616', 'I618', 'I619', 'I620', 'I621', 'I629',
         'I630', 'I631', 'I632', 'I634', 'I635', 'I636', 'I638', 'I639', 'I64', 'I650', 'I651', 'I652', 'I653', 'I658',
         'I659', 'I660', 'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I670', 'I671', 'I672', 'I673', 'I674', 'I675',
         'I676', 'I677', 'I678', 'I679', 'I680', 'I689', 'I690', 'I691', 'I692', 'I693', 'I694', 'I698',
         'I7000', 'I7001', 'I7010', 'I7011', 'I7020', 'I7021', 'I7080', 'I7081', 'I710', 'I711', 'I712', 'I713', 'I714',
         'I715', 'I716', 'I718', 'I719', 'I720', 'I721', 'I722', 'I723', 'I724', 'I725', 'I726', 'I728', 'I729',
         'I730', 'I731', 'I738', 'I739', 'I740', 'I741', 'I742', 'I743', 'I744', 'I745', 'I748', 'I749',
         'I770', 'I771', 'I772', 'I773', 'I774', 'I775', 'I776', 'I778', 'I779', 'I780', 'I781', 'I788', 'I789',
         'G951', 'H341', 'H342', 'O100', 'O101', 'O102', 'O109', 'S0660', 'S0661', 'Z951', 'Z955'
         ]

pattern = '|'.join(HeartDiseaseCodesAll)
df['HeartDisease'] = np.where(df['listICD'].str.contains(pattern), 1, 0)  # 1 if in list of disease codes, 0 if healthy
df.drop(columns=['p41270', 'listICD'], inplace=True)
print(df['HeartDisease'].sum())  # total number of ppl in dataset diagnosed with cardiovascular disease
print(df.groupby(['p31']).sum(numeric_only=True))  # number of female (0) and male (1) ppl diagnosed with heart disease
print(df['p31'].count())
print(df['p31'].sum())
df.to_csv('HeartDiseaseLarge.csv', index=False)