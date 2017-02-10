N = 50000000
a = [(1, 1) for i in range(N)]

b = [x+y for ix, x, y in zip(range(len(a)), a, a) if ix % 1000 == 0]
# print(len(b))

# b = []
# for i in range(N):
#     if i % 1000 == 0:
#         b.append(a[i]+a[i])

print(len(b))