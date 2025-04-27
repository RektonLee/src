# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
# import matplotlib.pyplot as plt
 
# fig = plt.figure()
# plt.plot(range(10))
# plt.savefig("dummy_name.png")

hh="d"
try:
    if hh == "d":
       raise ValueError("❌ 发生错误")
    print("✅ 没有错误")
except ValueError as e:
    print(e)
    # 这里可以添加更多的错误处理逻辑