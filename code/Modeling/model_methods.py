import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

class ModelingML():
    
    def __init__(self):
        pass
    
    
    
    def classification_model(self, model, scaler, x_train, x_test, y_train,y_test):
        '''
        Esta função é reponsavel por criar a pipeline para os modelos de machine learning.
        
        Args:
            model: O modelo de machine learning.
            scaler: O escalonador dos dados
            x_train: O conjunto de treino sem a váriavel alvo.
            x_test: O conjunto de teste sem a variável alvo.
            y_train: O conjunto de treino com a variável alvo.
            y_test: O conjunto de teste com a variável alvo.
        
        Returns:
            y_pred: São os valores previstos após o treinamento do conjunto de dados.
            accuracy: Valor da acurácia do modelo de machine learning.
            cm: Valores da matriz de confusão do modelo
            fpr: Taxa de falsos positivos do modelo.
            tpr: Taxa de verdadeiros positivos do modelo.
            auc: Valor da métrica que avalia a qualidade do modelo de classificação.
        '''
        
        # cria pipeline
        pipeline = make_pipeline(scaler, model)
        
        # ajusta modelo aos dados de treino
        pipeline.fit(x_train, y_train)
        
        # fazendo previsões nos dados de teste
        y_pred = pipeline.predict(x_test)
        
        # calculando métricas de validação de modelo
            # Acurácia
        accuracy = accuracy_score(y_test,y_pred)
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        return y_pred, accuracy, cm, fpr, tpr, auc
    
    
    
  
    
    
    
    def metrics_validation(self, y_test, y_pred, accuracy, cm):
        '''
        Esta função é responsável por calcular as métricas de validação do modelo.
        
        Args:
            y_test: O conjunto de teste com a variável alvo.
            y_pred: São os valores previstos após o treinamento do conjunto de dados.
            accuracy: Valor da acurácia do modelo de machine learning.
            cm: Valores da matriz de confusão do modelo
        
        Returns:
            Exibe a plotagem de valores e gráficos da validação.
        '''
        
        # exibindo F1-score e Recall
        print(classification_report(y_test,y_pred))
        print('-'*60)

        # exibindo Acurácia
        print('A acurácia do modelo de Regressão Logística é:', accuracy)
        print('-'*60)

        # Exibindo Matrix de confusão
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=('cividis'), cbar=False)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de confusão')
        
        return plt.show()



    def cross_val(self, fpr, tpr, auc):
        '''
        Esta função é responável por exibir o gráfico de validação cruzada.
        
        Args:
            fpr: Taxa de falsos positivos do modelo.
            tpr: Taxa de verdadeiros positivos do modelo.
            auc: Valor da métrica que avalia a qualidade do modelo de classificação.
            
        Returns:
            Polta o gráfico de validação cruzada
        '''
        # exibindo validação crusada
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'Regressão logística (AUC = {auc:.2f})')
        plt.plot([0,1],[0,1], linestyle = '--', color='gray', label = 'Random')
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curva ROC')
        plt.legend()
        
        return plt.show()