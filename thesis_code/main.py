import layer
import actfun

#Activation function
fun = actfun.sigmoid

#Layer initialization
lay = layer.dense(5, 2, fun)

#Sample input batch
inp = [[0,1,3,4,2], [5,3,2,1,6]]

#Output of a forward pass
out = lay.forward(inp)

print(out)
