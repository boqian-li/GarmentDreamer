import os


# bad_ids = ['tee_sleeveless_762', 'tee_sleeveless_270', 'tee_sleeveless_1276', 'tee_sleeveless_3248', 'tee_sleeveless_1777', 'tee_132',
#  'tee_sleeveless_1213', 'tee_sleeveless_2362', 'tee_sleeveless_2524', 'tee_sleeveless_940', 'tee_sleeveless_2535', 'tee_sleeveless_2664',
#  'tee_sleeveless_2648', 'tee_sleeveless_431', 'tee_sleeveless_1132', 'jumpsuit_sleeveless_2004', 'jumpsuit_sleeveless_1205', 'jumpsuit_sleeveless_3341',
#  'jumpsuit_sleeveless_2001', 'jumpsuit_sleeveless_2914', 'Trousers_553', 'jumpsuit_sleeveless_1619', 'Trousers_289', 'Dress_2023', 'Dress_714', 'Dress_1591',
#  'Dress_63', 'Trousers_1605', 'Dress_305', 'Trousers_1228', 'jumpsuit_sleeveless_3763', 'Trousers_399',
#  'Trousers_342', 'Trousers_1579', 'Trousers_896', 'Trousers_662', 'Dress_219',
#  'Dress_842', 'Dress_1605', 'Trousers_359', 'Trousers_120', 'Dress_900', 'Trousers_374',
#  'Trousers_660', 'Trousers_664', 'Trousers_1471', 'Trousers_411', 'Trousers_607', 'Dress_32', 'Trousers_1021',
#  'Trousers_791', 'Dress_1368', 'Trousers_1566', 'Trousers_1457', 'Dress_740', 'jumpsuit_sleeveless_2203', 'Trousers_718',
#  'Trousers_1242', 'Trousers_898', 'Trousers_1265', 'Dress_1407', 'Trousers_1329', 'Trousers_1106', 'Trousers_69', 
#  'Trousers_116', 'Trousers_243', 'Trousers_547', 'Trousers_1286', 'Trousers_1621', 'Trousers_1422', 'Trousers_1482',
#  'Trousers_1577', 'Trousers_1675', 'Trousers_1206', 'Dress_2020', 'Trousers_583', 'Trousers_368', 'Trousers_1685',
#  'Trousers_119', 'Trousers_1490', 'Trousers_262', 'Trousers_322', 'jumpsuit_sleeveless_1898', 'Trousers_904', 'Dress_1824',
#  'Trousers_805', 'Dress_623', 'Trousers_57', 'Dress_1693', 'Trousers_188', 'Trousers_493', 'Dress_1699', 'Trousers_1342',
#  'Trousers_1216', 'Trousers_303', 'Trousers_1549', 'Dress_412', 'Trousers_1533', 'Trousers_1017', 'Dress_924', 'Dress_989',
#  'Trousers_643', 'Trousers_1278', 'Trousers_1581', 'Dress_1246', 'Trousers_988', 'Trousers_1508', 'Trousers_651', 'Dress_1074', 
#  'Dress_1116', 'Dress_1691', 'Dress_690', 'Trousers_1636', 'Trousers_1294', 'Dress_1959', 'Dress_1053', 'Trousers_174', 'Trousers_277',
#  'Trousers_1175', 'Trousers_247', 'Trousers_670', 'Trousers_206', 'Trousers_893', 'Dress_509', 'Trousers_125', 'Trousers_846', 'Trousers_1062', 
#  'Trousers_1033', 'Trousers_1625', 'Trousers_1054', 'Dress_642', 'Dress_1026', 'Trousers_421', 'Dress_1111', 'Trousers_943', 'Trousers_164',
#  'Trousers_1079', 'Trousers_965', 'Trousers_202', 'Trousers_591', 'Trousers_912', 'Dress_328', 'Trousers_1297', 'Dress_255', 'Dress_619',
#  'Dress_1976', 'Trousers_968', 'Trousers_993', 'Trousers_1041', 'Trousers_297', 'Trousers_1615', 'Trousers_166', 'Trousers_389', 'Trousers_146',
#  'Trousers_541', 'Trousers_1463', 'Trousers_576', 'Dress_504', 'Trousers_425', 'Dress_1406', 'Trousers_6', 'Trousers_250']

root2 = '/home/boqian/Desktop/exported_codes_train'
categories = {}

for file in os.listdir(root2):
    if file.endswith('pt'):
        file_path = os.path.join(root2, file)
        category = '_'.join(file.split('_')[:-1])
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1

print(categories)