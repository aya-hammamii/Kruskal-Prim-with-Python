from flask import Flask,render_template,request
import pandas as pd
from math import expm1
import numpy as np 


app = Flask(__name__, static_url_path='/static')




@app.route('/',methods=['GET', 'POST'])
def home():
    if "submit" in request.form and request.method=="POST":
        if request.files['file'].filename == '':
            return render_template('main.html', _anchor="section-statut",message="Erreur !",response="Aucun fichier sélectionné !")
        else:
            file = request.files['file']
            l=matrice(file)
            #print(l)
            h=[]
            t=()
            for i in l:
                for j in range (3):
                    t=tuple(i)
                h.append(t)


            arcs = [[int(x) for x in lst] for lst in h]
           # print("Liste des arcs au format (début, fin, longueur): ")


            """
            # Liste des arcs au format (début, fin, longueur)
            arcs = [
                (0, 1, 7), (1, 2, 8), (0, 3, 5), (1, 3, 9), (1, 4, 7), (2, 4, 5),
                (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)
            ]

            """
            arcs_initial=arcs

            arcs.sort(key=lambda x: x[2])

            nombre_sommets = request.values.get('nombre_sommets')

            nombre_sommets = int(nombre_sommets)
            
            k=Kruskal(arcs, nombre_sommets)
            ###########Affichage de la graphe initial
            import pandas as pd
            df_initial = pd.DataFrame (arcs_initial, columns = ['source','dest','weight'])
            import networkx as nx
            G0=nx.from_pandas_edgelist(df_initial,
                         source='source',
                         target='dest',
                         edge_attr='weight')
            import matplotlib.pyplot as plt
            pos=nx.spring_layout(G0)
            nx.draw(G0, pos, with_labels=True, font_weight='bold')
            edge_weight = nx.get_edge_attributes(G0,'weight')
            nx.draw_networkx_edge_labels(G0, pos, edge_labels = edge_weight)
            plt.savefig("C:/Users/ayaha/Algo_Project/static/images/graphe_initial.png")
            # clearing the current plot
            plt.clf()
            

            ###########Affichage de MSP
            df = pd.DataFrame (k, columns = ['source','dest','weight'])
            import networkx as nx
            G=nx.from_pandas_edgelist(df,
                                     source='source',
                                     target='dest',
                                     edge_attr='weight')
            pos=nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, font_weight='bold')
            edge_weight = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_weight)
            plt.savefig("C:/Users/ayaha/Algo_Project/static/images/MSP.png")
            # clearing the current plot
            plt.clf()

            return render_template('main.html', _anchor="section-statut",message='Success !'+'\n' + 'L\'arbre initial:' + str(arcs)+'\n', response='L\'arbre du cout minimal'+ str(k) )
        
        
    return render_template('main.html', _anchor="section-statut")






############################## Kruskal##############################################
def matrice(file):
    matrix=file.readlines()
    return [i.strip().split() for i in matrix]
class EnsembleDisjoint:
    """
    Cette classe permet de gérer les ensmebles disjoints. Deux éléments sont
    considérés dans le même ensemble s'ils ont le même parent.
    """
    parent = {}

    # Création de n ensemble disjoints, état de départt de notre graphe
    def __init__(self, N):
        for i in range(N):
            self.parent[i] = i

    # Fonction qui permet de retrouver le parent le plus lointain
    def get_parent(self, k):
        if self.parent[k] == k:
            return k

        return self.get_parent(self.parent[k])

    # Union de deux ensembles jusque là disjoints
    def Union(self, a, b):
        x = self.get_parent(a)
        y = self.get_parent(b)

        self.parent[x] = y

        
        
def Kruskal(arcs, nombre_sommets):
    """
    Construction de l'arbre couvrant minimum à l'aide de l'algorithme de Kruskal
    Les paramètres sont :
        - Les arcs du graphe au format (début, fin, longueur)
        - Le nombre de sommets dans le graph
    """

    Arbre_minimum = []
    ed = EnsembleDisjoint(nombre_sommets)
    index = 0

    while len(Arbre_minimum) != nombre_sommets - 1:

        (src, dest, weight) = arcs[index]
        index = index + 1

        x = ed.get_parent(src)
        y = ed.get_parent(dest)

        if x != y:
            Arbre_minimum.append((src, dest, weight))
            ed.Union(x, y)

    return Arbre_minimum










########################################## Prim


def primsAlgorithm(vertices,adjacencyMatrix):

        
            # Creating another adjacency Matrix for the Minimum Spanning Tree:
            mstMatrix = [[0 for column in range(vertices)] for row in range(vertices)]
            
                    # Defining a really big number:
            positiveInf = float('inf')
             # This is a list showing which vertices are already selected, so we don't pick the same vertex twice and we can actually know when stop looking
            selectedVertices = [False for vertex in range(vertices)]
            # While there are vertices that are not included in the MST, keep looking:
            while(False in selectedVertices):
                # We use the big number we created before as the possible minimum weight
                minimum = positiveInf

                # The starting vertex
                start = 0

                # The ending vertex
                end = 0

                for i in range(0,vertices):
                    # If the vertex is part of the MST, look its relationships
                    if selectedVertices[i]:
                        # Again, we use the Symmetric Matrix as an advantage:
                        for j in range(0+i,vertices):
                            # If the vertex analyzed have a path to the ending vertex AND its not included in the MST to avoid cycles)
                            if (not selectedVertices[j] and adjacencyMatrix[i][j]>0):  
                                # If the weight path analyzed is less than the minimum of the MST
                                if adjacencyMatrix[i][j] < minimum:
                                    # Defines the new minimum weight, the starting vertex and the ending vertex
                                    minimum = adjacencyMatrix[i][j]
                                    start, end = i, j

                # Since we added the ending vertex to the MST, it's already selected:
                selectedVertices[end] = True

                # Filling the MST Adjacency Matrix fields:
                mstMatrix[start][end] = minimum

                # Initially, the minimum will be Inf if the first vertex is not connected with itself, but really it must be 0:
                if minimum == positiveInf:
                    mstMatrix[start][end] = 0

                # Symmetric matrix, remember
                mstMatrix[end][start] = mstMatrix[start][end]

            # Show off:
            print(mstMatrix)


    
    


#################

@app.route('/prim',methods=['GET', 'POST'])
def prim():
    if "submit" in request.form and request.method=="POST":
        if request.files['file'].filename == '':
            return render_template('main2.html', _anchor="section-statut",message="Erreur !",response="Aucun fichier sélectionné !")
        else:
            file = request.files['file']



            l=matrice(file)
            print(l)
            adjacencyMatrix = [[int(x) for x in lst] for lst in l]         

            vertices = request.values.get('vertices')
            vertices=int(vertices)

            primsAlgorithm(vertices,adjacencyMatrix)


            n0=len(adjacencyMatrix) # nombre de lignes
            m0=len(adjacencyMatrix[0]) # nombre de colonnes
            import networkx as nx   
            import matplotlib.pyplot as plt
            G0 = nx.Graph()
            for i in range(n0):
                for j in range(m0):
                    if adjacencyMatrix[i][j] > 0:
                        G0.add_edge(i, j, weight=adjacencyMatrix[i][j])
            pos = nx.nx_pydot.graphviz_layout(G0)
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            nx.draw_networkx(G0, pos=pos, with_labels='true', font_weight='bold')
            edge_weight = nx.get_edge_attributes(G0,'weight')
            nx.draw_networkx_edge_labels(G0, pos, edge_labels = edge_weight)
            plt.savefig("C:/Users/ayaha/Algo_Project/static/images/Graphe_initial_Prim.png")
            plt.show()
            # clearing the current plot
            plt.clf() 

            
            

                        ######################
            n=len(mstMatrix) # nombre de lignes
            m=len(mstMatrix[0]) # nombre de colonnes
            import networkx as nx   
            import matplotlib.pyplot as plt
            G = nx.Graph()
            for i in range(n):
                for j in range(m):
                    if mstMatrix[i][j] > 0:
                        G.add_edge(i, j, weight=mstMatrix[i][j])
            pos = nx.nx_pydot.graphviz_layout(G)
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            nx.draw_networkx(G, pos=pos, with_labels='true', font_weight='bold')
            edge_weight = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_weight)
            plt.savefig("C:/Users/ayaha/Algo_Project/static/images/MSP_Prim.png")
            # clearing the current plot
            plt.clf()

            return render_template('main2.html', _anchor="section-statut",message='Success !'+'\n' + 'Matrice initial:' + str(adjacencyMatrix)+'\n', response='Matrice du cout minimal' + str(adjacencyMatrix) )
        
        
    return render_template('main2.html', _anchor="section-statut") 
    
    

    
    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
    
    
    
            
        