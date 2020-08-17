n,m=map(int,input().split())
g=[[] for j in range(n)]
for i in range(m):
    u,v,w=map(int,input().split())
    u=u-1
    v=v-1
    g[u].append(v)
    g[v].append(u)
    print(u,v)
print(g)

def cyc(v, visited, parent): 
    visited[v] = True
    for i in g[v]: 
        if visited[i] == False: 
            if cyc(i, visited, v) == True: 
                return True
            elif i != parent: 
                return True
        return False       
def check(): 
    visited = [False] * n
    if cyc(0, visited, -1) == True: 
        return False
    for i in range(n): 
        if visited[i] == False: 
            return False
    return True
print(check())
#1 2 1
#1 3 2
#2 3 3