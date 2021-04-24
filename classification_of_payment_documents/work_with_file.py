from cmath import log
from collections import defaultdict
import openpyxl

wb = openpyxl.load_workbook('example_table.xlsx')

sheet = wb.active
rows = sheet.max_row
cols = sheet.max_column

name = []
description = []
cnt = 0

for i in range(2, rows + 1):
    cnt = cnt + 1;
    for j in range(1, cols + 1):
        if (j == 5):
            cell = sheet.cell(row = i, column = j)
            name.append(str(cell.value))
        if (j == 7):
            cell = sheet.cell(row=i, column=j)
            description.append(str(cell.value))

# cnt = 0
# for i, j in zip(name, description):
#     print(i, j)
#     cnt += 1;
#     if (cnt > 30):
#         break