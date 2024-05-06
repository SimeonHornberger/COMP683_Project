from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,classification_report
import flowkit as fk
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def getTimestamp(filename):
    endingRemoved = filename.replace(".fcs","")
    timestamp = ""
    if endingRemoved.endswith("24H"):
        timestamp = "24H"
    elif endingRemoved.endswith("48H"):
        timestamp = "48H"
    elif endingRemoved.endswith("72H"):
        timestamp = "72H"
    elif endingRemoved.endswith("120H"):
        timestamp = "120H"
    elif endingRemoved.endswith("7D"):
        timestamp = "7D"
    elif endingRemoved.endswith("14D"):
        timestamp = "14D"
    elif endingRemoved.endswith("90D"):
        timestamp = "90D"
    elif endingRemoved.endswith("1Y"):
        timestamp = "1Y"
    elif "Control" in endingRemoved:
        timestamp = "Healthy"
    else:
        timestamp = None
    return timestamp


def getTimestampBinaryClassification(filename):
    endingRemoved = filename.replace(".fcs","")
    timestamp = ""
    if "Control" in endingRemoved:
        timestamp = "Healthy"
    else:
        timestamp = "Stroke"
    return timestamp

def getTimestampCoarseClassification(filename):
    endingRemoved = filename.replace(".fcs","")
    timestamp = ""
    weekOrUnder = endingRemoved.endswith("24H") or endingRemoved.endswith("48H") or endingRemoved.endswith("72H") or endingRemoved.endswith("120H") or endingRemoved.endswith("7D")
    overWeek = endingRemoved.endswith("14D") or endingRemoved.endswith("90D") or endingRemoved.endswith("1Y")
    if weekOrUnder:
        timestamp = "weekOrUnder"
    elif overWeek:
        timestamp = "overWeek"
    else:
        timestamp = "Healthy"
    return timestamp

def getTimestamp4(filename):
    endingRemoved = filename.replace(".fcs","")
    timestamp = ""
    threeDaysAndUnder = endingRemoved.endswith("24H") or endingRemoved.endswith("48H") or endingRemoved.endswith("72H")
    between3and7 = endingRemoved.endswith("120H") or endingRemoved.endswith("7D")
    between7and90 = endingRemoved.endswith("14D") or endingRemoved.endswith("90D")
    between90andYear = endingRemoved.endswith("1Y")
    healthy = "Control" in endingRemoved
    if threeDaysAndUnder:
        timestamp="threeDaysAndUnder"
    elif between3and7:
        timestamp="between3and7"
    elif between7and90:
        timestamp="between7and90"
    elif between90andYear:
        timestamp="between90andYear"
    elif healthy:
        timestamp="healthy"
    else:
        timestamp=None
    return timestamp


def convertFCStoDataframe(filename):
    timestamp = getTimestamp(filename)
    sample = fk.Sample(filename, ignore_offset_error=True)
    dm = sample._get_raw_events()
    CNames = np.array(sample.pns_labels)
    columnNames = ["115In_CD45",
                   "139La_CD66",
                   "141Pr_CD7",#
                   "142Nd_CD19",#
                   "143Nd_CD45RA",
                   "144Nd_CD11b",
                   "145Nd_CD4",
                   "146Nd_CD8a",#
                   "147Sm_CD11c",
                   "148Nd_CD123",#
                   "149Sm_CREB",
                   "150Nd_STAT5",
                   "151Eu_p38",
                   "152Sm_TCRgd",
                   "153Eu_STAT1",
                   "154Sm_STAT3",
                   "155Gd_S6",
                   "156Gd_CD24",
                   "157Gd_CD38",
                   "158Gd_CD33",
                   "159Tb_MAPKAPK2",
                   "160Gd_Tbet",
                   "162Dy_FoxP3",
                   "164Dy_IkB",
                   "165Ho_CD16",
                   "166Er_NFkB",
                   "167Er_ERK",
                   "168Er_pSTAT6",
                   "169Tm_CD25",
                   "170Er_CD3",#
                   "171Yb_CD27",
                   "172Yb_IgM",
                   "173Yb_CCR2",
                   "174Yb_HLADR",
                   "175Lu_CD14",#
                   "176Lu_CD56"]
    toKeep = [CNames.tolist().index(name) for name in columnNames]
    dm_trans = np.arcsinh(1./5 * dm[:,toKeep])
    CNames = CNames[toKeep]
    df = pd.DataFrame(dm_trans,columns=CNames)
    df_reduced = df.head(10000)
    #Impute zero values with mean for that feature
    for column in df_reduced:
        if column != "time":
            mean = df_reduced[column].mean()
            df_reduced[column] = df_reduced[column].replace(0, mean)
    #Scale data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_reduced)
    scaled_df = pd.DataFrame(scaled_features, columns=df_reduced.columns)
    scaled_df["time"] = timestamp
    return scaled_df

def main():
    df_24H = convertFCStoDataframe("SCAN_1012_24H.fcs")
    df_48H = convertFCStoDataframe("SCAN_1005_48H.fcs")
    df_72H = convertFCStoDataframe("SCAN_1005_72H.fcs")
    df_120H = convertFCStoDataframe("SCAN_1003_120H.fcs")
    df_7D = convertFCStoDataframe("SCAN_1003_7D.fcs")
    df_14D = convertFCStoDataframe("SCAN_1001_14D.fcs")
    df_90D = convertFCStoDataframe("SCAN_1006_90D.fcs")
    df_1Y = convertFCStoDataframe("SCAN_1001_1Y.fcs")
    df_Healthy = convertFCStoDataframe("SCAN_Control_IFNa_LPS_1.fcs")
    df_list = [df_24H,df_48H,df_72H,df_120H,df_7D,df_14D,df_90D,df_1Y,df_Healthy]
    df_complete = pd.concat(df_list)

    X = df_complete.loc[:,df_complete.columns!="time"]
    y = df_complete["time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #test size was 0.33

    classifier = LogisticRegression(random_state=0)
    #classifier = KNeighborsClassifier(n_neighbors=30) #initially n=5
    #classifier = RandomForestClassifier(max_depth=20, random_state=0) #max_depth initially 2
    labels = ["24H","48H","72H","120H","7D","14D","90D","1Y","Healthy"]

    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    classificationReport = classification_report(y_test, y_pred, output_dict=True)
    #print(classificationReport)
    #plotConfusionMatrix(y_test,y_pred)
    recallLR = classificationReport["macro avg"]["recall"]
    print(recallLR)
    recallPlotValues = []
    for timestamp in labels:
        recallPlotValues.append(classificationReport[timestamp]["recall"])
    plt.scatter(labels,recallPlotValues)
    plt.ylim(0.5,1)
    plt.xlabel("Label")
    plt.ylabel("Recall")
    plt.title("Recall for label")
    plt.show()

    classifier = KNeighborsClassifier(n_neighbors=30) #initially n=5
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    classificationReport = classification_report(y_test, y_pred, output_dict=True)
    recallKNN = classificationReport["macro avg"]["recall"]
    print(recallKNN)
    recallPlotValues = []
    for timestamp in labels:
        recallPlotValues.append(classificationReport[timestamp]["recall"])
    plt.scatter(labels,recallPlotValues)
    plt.ylim(0.5,1)
    plt.xlabel("Label")
    plt.ylabel("Recall")
    plt.title("Recall for label")
    plt.show()

    classifier = RandomForestClassifier(max_depth=20, random_state=0) #max_depth initially 2
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    classificationReport = classification_report(y_test, y_pred, output_dict=True)
    recallRF = classificationReport["macro avg"]["recall"]
    print(recallRF)
    recallPlotValues = []
    for timestamp in labels:
        recallPlotValues.append(classificationReport[timestamp]["recall"])
    plt.scatter(labels,recallPlotValues)
    plt.ylim(0.5,1)
    plt.xlabel("Label")
    plt.ylabel("Recall")
    plt.title("Recall for label")
    plt.show()

    barLabels = ["Logistic Regression","K-Nearest Neighbors","Random Forest"]
    values = [recallLR,recallKNN,recallRF]
    bars = plt.bar(barLabels,values,color=['blue', 'orange', 'green'])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom')
    plt.xlabel("Model")
    plt.ylabel("Macro average recall")
    plt.title("Macro average recall for all models")
    plt.show()

def plotConfusionMatrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap='Blues', fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()

