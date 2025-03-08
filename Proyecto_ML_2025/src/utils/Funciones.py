import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




def snake(col):
        col = re.sub(r'[\s\-]', '_', col) 
        col = re.sub(r'([a-z])([A-Z])', r'\1_\2', col)  
        col = re.sub(r'_{2,}', '_', col)  
        return col.lower().strip('_')  



def k_optimo(X_train, y_train, X_test, y_test , k_min=1, k_max=100, k_step= 2):
    """
       Esta funcion busca el k mas optimo para el modelo KNN

    Parametrosd
    X_train
    y_train
    X_test
    k_min
    k_max
    k_step 
#+
    Devuelve una tupla con el K y la precision
    """#+
    precision = []

    for k in range(k_min, k_max, k_step):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acuracy = accuracy_score(y_test, y_pred)
        precision.append((k,acuracy))

    mejor_k = max(precision,key=lambda x: x[1])

    return mejor_k, precision