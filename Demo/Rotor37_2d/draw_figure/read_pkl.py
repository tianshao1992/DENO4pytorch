import pickle

F=open(r'D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\work_train_FNO\FNO_0\x_norm.pkl','rb')
# D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\work_train_FNO\FNO_0

content=pickle.load(F)
print(content)