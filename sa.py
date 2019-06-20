import random
import math
import numpy as np
import csv
#import time
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])
interconnectfile = "testcasehm2.csv"
posfile = "areaplan.csv"
area = "area.csv"
def init_graph(k, N,M, graph,blocks):
	for node in xrange(1,k+1):
		for a in xrange(0,blocks[node-1]):
			value = False
			while(not value):
				i = np.random.randint(0,N)
				j = np.random.randint(0,M)
				if(graph[i,j]==0):
					graph[i,j] = node
					value = True
	print graph
	return graph


def next_state(k, N,M,graph):
	n = np.random.randint(int(N/2),N)
	for i in xrange(1,n):
		val = False
		while(not val):
			new_graph = np.array(graph)
			x1 = np.random.randint(0,N)
			x2 = np.random.randint(0,N)
			y1 = np.random.randint(0,M)
			y2 = np.random.randint(0,M) 
			while(x1==x2 and y1==y2):
				x2 = np.random.randint(0, N)
				y2 = np.random.randint(0, M)
			if(graph[x1,y1]>0 and graph[x2,y2]>0):
				new_graph[x1, y1] = graph[x2,y2]
				new_graph[x2, y2] = graph[x1, y1]
				val = True
	return new_graph,x1,x2,y1,y2

def cost_calc(k,N,M, graph, interconnects,pin_pos):
	cost = 0
	for i in xrange(0,N):
		for j in xrange(0,M):
			element1 = graph[i,j]
			if(element1 > 0):
				value = False
				if( pin_pos.size != 0):
					for p in xrange(0,pin_pos.shape[1]):
						x = pin_pos[0,p]
						y = pin_pos[1,p]
						cost = cost + (np.absolute(x-i) +np.absolute(y-j))*interconnects[element1-1,k+p]
				m = i
				for l in xrange(j+1,M):
					element2 = graph[m,l]
					if(element2>0):
						if(element1< element2):
							temp1 = element1-1
							temp2 = element2-1
							cost = cost + (np.absolute(m-i) +np.absolute(l-j))*interconnects[temp1,temp2]
						if (element1> element2):
							temp1 = element2-1
							temp2 = element1-1
							cost = cost + (np.absolute(m-i) +np.absolute(l-j))*interconnects[temp1,temp2]
						if (element1 == element2 and value ==False):
							cost = cost + (np.absolute(m-i) +np.absolute(l-j))*10000
							value =  True
				for m in xrange(i+1,N):
					for l in xrange(0,M):
						element2 = graph[m,l]
						if(element2>0):
							if(element1< element2):
								temp1 = element1-1
								temp2 = element2-1
								cost = cost + (np.absolute(m-i) +np.absolute(l-j))*interconnects[temp1,temp2]
							if (element1> element2):
								temp1 = element2-1
								temp2 = element1-1
								cost = cost + (np.absolute(m-i) +np.absolute(l-j))*interconnects[temp1,temp2]
							if (element1 == element2 and value ==False):
								cost = cost + (np.absolute(m-i) +np.absolute(l-j))*10000
								value =  True
		
	return cost      
def cost_change(k,N, graph, interconnects,pin_pos,x1,x2,y1,y2):
	delta = 0
	change1 = graph[x2,y2]
	change2 = graph[x1,y1]
	for i in xrange(0,N):
		for j in xrange(0,N):
			element1 = graph[i,j]
			if(element1 > 0):
				temp1 = element1-1
				temp2 = change1-1
				if (element1> change1):
					temp1 = change1-1
					temp2 = element1-1
				delta = delta + (np.absolute(x2-i) +np.absolute(y2-j))*interconnects[temp1,temp2] - (np.absolute(x1-i) +np.absolute(y1-j))*interconnects[temp1,temp2]
				temp1 = element1-1
				temp2 = change2-1
				if (element1> change2):
					temp1 = change2-1
					temp2 = element1-1
				delta = delta + (np.absolute(x1-i) +np.absolute(y1-j))*interconnects[temp1,temp2] - (np.absolute(x2-i) +np.absolute(y2-j))*interconnects[temp1,temp2]
	if( pin_pos.size != 0):
		for p in xrange(0,pin_pos.shape[1]):
			x = pin_pos[0,p]
			y = pin_pos[1,p]
			delta = delta + (np.absolute(x-x2) +np.absolute(y-y2))*interconnects[change1-1,k+p]-(np.absolute(x-x1) +np.absolute(y-y1))*interconnects[change1-1,k+p]+(np.absolute(x-x1) +np.absolute(y-y1))*interconnects[change2-1,k+p]-(np.absolute(x-x2)+ np.absolute(y-y2))*interconnects[change2-1,k+p]
	return delta		
def sim_anneal(k,N,M,graph,interconnects,pin_pos,blocks):
	T = 10000000
	alpha = 0.998
	threshold = 0.00000001
	graph_curr = init_graph(k,N,M,graph,blocks)
	mincost = cost_calc(k,N,M, graph_curr, interconnects,pin_pos)
	costhist = []
	n_iter = 0
	while(T>threshold):
		#start_time = time.time()
		tempgraph,x1,x2,y1,y2 = next_state(k, N,M, graph_curr)
		#delta = cost_change(k,N, tempgraph, interconnects,pin_pos,x1,x2,y1,y2)
		cost_new = cost_calc(k,N,M, tempgraph, interconnects,pin_pos)
		delta = cost_new - mincost
		if (delta<0):
			graph_curr = tempgraph
			#mincost = mincost+delta
			mincost = cost_new
		else:
			p = np.exp(-delta / T)
			if(np.random.random()<p):
				graph_curr = tempgraph
				#mincost = mincost+delta
				mincost = cost_new
		T = alpha*T
		#print "--- %s seconds ---" % (time.time() - start_time)
		costhist.append(mincost)
		n_iter =n_iter+1
	return graph_curr,mincost,costhist,n_iter

def main():	
	block = []	
	with open(interconnectfile, 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
		hiearchies = list(csvreader.next())
    		connects = list(csv.reader(csvfile,delimiter=','))
	with open(posfile, 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
    		posgrid = list(csv.reader(csvfile,delimiter=','))
	with open(area, 'r') as csvfile: 
		csvreader = csv.reader(csvfile)
		block = list(csvreader.next())
	b=0
	for row in connects:
		connects[b] = ['0' if x==' ' else x for x in connects[b]]
		b=b+1
	pin = []
	k = len(block)
	print k
	N = len(posgrid)
	M = len(posgrid[0])
	graph = np.zeros([N,M])
	for i in xrange(0,N):
		for j in xrange(0,M):
			if(posgrid[i][j]=='pin'):
				pin.append([i,j])
				graph[i,j]= -1
			elif(posgrid[i][j]=='empty'):
				graph[i,j]= -2
	pin_pos = np.array(pin, dtype=np.float)
	pin_pos = pin_pos.transpose()
	interconnects = []
	interconnects = np.array(connects, dtype=np.float)
	blocks = np.array(block,dtype =np.int)
	#interconnects = np.random.rand(k+p,k+p)*10000
	#interconnects = np.triu(interconnects, 1)
	finalgraph,mincost,costhist,n_iter = sim_anneal(k,N,M,graph,interconnects,pin_pos,blocks)
	print "Final Cost", mincost
	print "Final Grid"
	finalgrid = np.empty([N, N], dtype="S25")
	pn = 1
	for i in xrange(0,N):
		for j in xrange(0,M):
			elem = int(finalgraph[i,j])
			if (elem >0):
				finalgrid[i,j] = hiearchies[elem-1]
			elif(elem>-2):
				finalgrid[i,j] = "pin"+str(pn)
				pn=pn+1
			else:
				finalgrid[i,j] = ' '
	np.savetxt("placement.csv", finalgrid, delimiter=",",fmt="%s")
 	fig,ax = plt.subplots(figsize=(24,16))
	ax.set_ylabel('Cost')
	ax.set_xlabel('Iterations')
	_=ax.plot(range(n_iter),costhist,'b.')
	plt.show()
if __name__ == '__main__':
	main()







         
