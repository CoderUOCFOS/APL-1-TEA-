#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
from scipy.signal import find_peaks


# In[2]:


#calibration using sulphur spectra

df_sul=pd.read_csv("suphur aperture 5.asc",sep="\t",header=None,nrows=1600)
df_sul=df_sul.to_numpy()
print(df_sul)
raman_shift=df_sul[:,0]
sulphur_intensity=df_sul[:,1]

peaks, _ = find_peaks(sulphur_intensity,height=400)

# plt.plot(raman_shift,sulphur_intensity)
# plt.plot(raman_shift[peaks][0:6], sulphur_intensity[peaks][0:6], 'x')

#print(raman_shift[peaks][0:6])


# In[3]:


#calibration pt 2
# acctual_raman_peaks = [153,187,220,249,439,473]
acctual_raman_peaks = [153,187,220,249,439,473]
params = np.polyfit(raman_shift[peaks][0:6],acctual_raman_peaks,1)

# raman_peaks_new=np.polyval(params,raman_shift[peaks][0:6])
# plt.plot(raman_shift[peaks][0:6],acctual_raman_peaks,'r*')
# plt.plot(raman_shift[peaks][0:6],raman_peaks_new,'b-')
# plt.xlabel("Experimental Sulphur Peaks (cm-1)")
# plt.ylabel("Actual Sulphur Peaks (cm-1)")
# plt.title("Calibration spectrum")


# In[4]:


#importing the data files 
files=[]
basepath=r"C:\Users\lakis\OneDrive\Desktop\UOC LMS\3rd Year\Advanced Physics lab 1\Research\TEA\Final day\Final day 532"
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        files=files+[entry]
        files.sort()
data_files=files[0:22]
print(data_files)


# In[5]:


#creating a numpy array with the data (22x1600) (no store bought samples)
intensity=np.zeros((22,1600))

count=0
for file_name in data_files:
    df=pd.read_csv(file_name,sep="\t",header=None,nrows=1600)
    df=df.to_numpy()
    if count==0:
        r_shift=df[:,0]

    df=df[:,1].reshape(1,1600)
    intensity[count,:]=df
    count=count+1

# r_shift=pd.DataFrame(r_shift)
# print(r_shift)
#calibration of instruemnt
r_shift_new=np.polyval(params,r_shift)
r_shift=pd.DataFrame(r_shift_new)
print(r_shift)


# In[6]:


#blank correction
intensity_samples=intensity[0:21,:]
intensity_blnk=intensity[21,:]
# print(intensity_samples)
# print(intensity_blnk)

#intensity values after the blank correction (22x1600)
intensity_corr=intensity_samples-intensity_blnk


# In[7]:


#the region of each tea sample (0-west high, 1- uva med,2-low,3-west med,4-uva high,5-nuwara eliya,6-storebought
df_label=np.array([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,3,4,0,0])


# In[8]:


#scaling the data
scaler=StandardScaler(with_mean=True,with_std=True)
intensity_corr_norm=scaler.fit_transform(intensity_corr)
print(intensity_corr_norm.shape)


df_corr = pd.DataFrame(data=intensity_corr_norm, index=df_label)
print(df_corr.shape)


# In[45]:


pca=PCA(n_components=5)
X_red=pca.fit_transform(df_corr)
np.cumsum((pca.explained_variance_ratio_)*100)


# In[10]:


#plotting the explained variances

features_pca=range(1,6)
plt.figure(facecolor="white",figsize=(6,6))
plt.bar(features_pca,pca.explained_variance_ratio_*100)
plt.step(range(1,6),np.cumsum(pca.explained_variance_ratio_*100),where='mid',label="CEV",color="red")
plt.xlabel("Principal Component",fontweight="bold")
plt.ylabel("% explained variance",fontweight="bold")
plt.title("% explained variance vs PCs",fontweight="bold")
plt.legend(["Cumulative % explained variance","Componentwise % explained variance"],fontsize=8,loc="center right")
plt.savefig("% explained variance",bbox_inches="tight",dpi=400)


# In[11]:


#saving components to a datframe
df_PC=pd.DataFrame(X_red)
print(np.cumsum(pca.explained_variance_ratio_*100))

#plotting the first two PC s
plt.figure(figsize=(6, 6),facecolor="white") 
scatter1=plt.scatter(X_red[:, 0],X_red[:, 1],c=df_label,s=15)
plt.legend(handles=scatter1.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10)
plt.title("First two PCs (99.10 % CEV)",fontweight="bold")
plt.xlabel("Principal Component 1",fontweight="bold")
plt.ylabel("Principal Component 2",fontweight="bold")
plt.savefig("First two PCs",bbox_inches="tight",dpi=400)


# In[12]:


plt.rcParams.update({'font.size': 14})
fig = plt.figure(1, figsize=(6,6),facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.grid(True)

scatter=ax.scatter(
    X_red[:, 0],
    X_red[:, 1],
    X_red[:, 2],
    c=df_label,
    s=15,
)

plt.legend(handles=scatter.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10)

ax.set_title("First three PCs (99.53% CEV)",fontweight="bold")
ax.set_xlabel("PC1",fontweight="bold")
ax.set_ylabel("PC2",fontweight="bold")
ax.set_zlabel("PC3",fontweight="bold")
plt.savefig("First three PCs",bbox_inches="tight",dpi=400)


# In[13]:


#LDA classification of the dataset

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(n_components=None)
x_lda=lda.fit_transform(df_corr,df_label)
print(np.cumsum(lda.explained_variance_ratio_*100))


# In[40]:


#LDA 2d plot
plt.figure(figsize=(6,6),facecolor="white")
scatter3=plt.scatter(x_lda[:,0],x_lda[:,1],c=df_label,s=15)
plt.legend(handles=scatter3.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10,loc="center left")
plt.title('First 2 LDA components (81.57% CEV)',fontweight="bold")
plt.xlabel('Component 1',fontweight="bold")
plt.ylabel('Component 2',fontweight="bold")
plt.savefig("First two LDA components",bbox_inches="tight",dpi=400)


# In[41]:


#LDA 3d plot
fig4 = plt.figure(1, figsize=(6,6),facecolor="white")
ax4 = fig4.add_subplot(111, projection="3d")

scatter4=ax4.scatter(
    x_lda[:, 0],
    x_lda[:, 1],
    x_lda[:, 2],
    c=df_label,
    s=15,
)

plt.legend(handles=scatter4.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10,loc="upper left")

ax4.set_title("First 3 LDA components (92.38% CEV)",fontweight="bold")
ax4.set_xlabel("C1",fontweight="bold")
ax4.set_ylabel("C2",fontweight="bold")
ax4.set_zlabel("C3",fontweight="bold")
plt.savefig("First 3 LDA components",bbox_inches="tight",dpi=400)


# In[17]:


#plotting the CEV for LDA
plt.figure(figsize=(6,6),facecolor="white")
features2=range(1,5)
plt.bar(features2,lda.explained_variance_ratio_*100)
plt.xlabel("LDA component",fontweight="bold")
plt.ylabel("% explained variance",fontweight="bold")
plt.title("% explained variance vs LDA Components",fontweight="bold")

plt.step(range(1,5),np.cumsum(lda.explained_variance_ratio_*100),where='mid',label="CEV",color="red")
plt.legend(["Cumulative % explained variance","Componentwise % explained variance"],fontsize=8,loc="center right")
plt.savefig("% explained variance for LDA",bbox_inches="tight",dpi=400)


# In[18]:


#non-linear dimensionality red (Kernal PCA)
from sklearn.manifold import TSNE
tsne=TSNE(n_components=3,random_state=1)
x_tsne=tsne.fit_transform(df_PC)


# In[19]:


#TSNE 2d plot
plt.figure(figsize=(6,6),facecolor="white")
scatter4=plt.scatter(x_tsne[:,0],x_tsne[:,1],c=df_label,s=15)
plt.legend(handles=scatter4.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10)
plt.title('TSNE with 2 components',fontweight="bold")
plt.xlabel('Component 1',fontweight="bold")
plt.ylabel('Component 2',fontweight="bold")
plt.savefig("TSNE 2 components",bbox_inches="tight",dpi=400)


# In[20]:


#TSNE 3d plot
fig5 = plt.figure(1, figsize=(6,6),facecolor="white")
ax5 = fig5.add_subplot(111, projection="3d")

scatter5=ax5.scatter(
    x_tsne[:, 0],
    x_tsne[:, 1],
    x_tsne[:, 2],
    c=df_label,
    s=15,
)

plt.legend(handles=scatter5.legend_elements()[0],labels=["Western High","Uva Med","Low Grown","Western Med","Uva High"],fontsize=10)

ax5.set_title("TSNE for TEA data for 3 components",fontweight="bold")
ax5.set_xlabel("C1",fontweight="bold")
ax5.set_ylabel("C2",fontweight="bold")
ax5.set_zlabel("C3",fontweight="bold")
plt.savefig("TSNE 3 components",bbox_inches="tight",dpi=400)


# In[21]:


#classify based on grade
#0-BOP, 1-BOPF, 2-PEK,3-BOP1, 4-PF1,5-FF1,6-PD,7-FBOP,8-BOPS,9-FBOP1

df_labels_grades=[0,1,2,0,2,1,3,4,8,1,5,0,3,4,9,1,7,6,5,7,0]

df_labels_grades2=["WH_BOP","UM_BOPF","L_PEK","WM_BOP","UH_PEK","WH_BOPF","UM_BOP1",'L_PF1',"WM_BOPS","UH_BOPF","WH_FF1","UM_BOP","L_BOP1","WM_PF1","UH_FBOP1","WH_BOPF","UM_FBOP","WM_PD","UH_FF1","WH_FBOP","WH_BOP"]


# In[22]:


#LDA classification of the dataset according to grade

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda2=LinearDiscriminantAnalysis(n_components=None)
x_lda2=lda2.fit_transform(df_corr,df_labels_grades)
print(np.cumsum(lda2.explained_variance_ratio_))


# In[23]:


#LDA , 2 components, based on grades

from matplotlib.colors import ListedColormap
colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray', 'black']
# Create a custom colormap
cmap1 = ListedColormap(colors)

plt.figure(figsize=(6,6),facecolor="white")
scatter6=plt.scatter(x_lda2[:,0],x_lda2[:,1],c=df_labels_grades,s=15,cmap=cmap1)
plt.legend(handles=scatter6.legend_elements()[0],labels=["BOP","BOPF","PEK","BOP1","PF1","FF1","PD","FBOP","BOPS","FBOP1"],fontsize=6)
plt.title('First 2 LDA components: Grades (81.79% CEV)',fontweight="bold")
plt.xlabel('Component 1',fontweight="bold")
plt.ylabel('Component 2',fontweight="bold")
plt.savefig("LDA (tea grades) 2 components",bbox_inches="tight",dpi=400)


# In[24]:


#LDA 3 componenets:Grades,3d plot

# %matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
fig5 = plt.figure(1, figsize=(6,6),facecolor="white")
ax5 = fig5.add_subplot(111, projection="3d")

scatter5=ax5.scatter(
    x_lda2[:, 0],
    x_lda2[:, 1],
    x_lda2[:, 2],
    c=df_labels_grades,
    s=15,
    cmap=cmap1,
    
)

plt.legend(handles=scatter5.legend_elements()[0],labels=["BOP","BOPF","PEK","BOP1","PF1","FF1","PD","FBOP","BOPS","FBOP1"],fontsize=8,loc="upper left")

ax5.set_title("First 3 LDA components:Grades (89.22% CEV)",fontweight="bold")
ax5.set_xlabel("C1",fontweight="bold")
ax5.set_ylabel("C2",fontweight="bold")
ax5.set_zlabel("C3",fontweight="bold")
plt.savefig("LDA (tea grades) 3 components",bbox_inches="tight",dpi=400)


# In[25]:


#plotting the CEV for LDA
plt.figure(figsize=(6,6),facecolor="white")
features3=range(1,10)
plt.bar(features3,lda2.explained_variance_ratio_*100)
plt.xlabel("LDA components",fontweight="bold")
plt.ylabel("% explained variance",fontweight="bold")
plt.title("% explained variance vs LDA Components:Grades",fontweight="bold")

plt.step(range(1,10),np.cumsum(lda2.explained_variance_ratio_*100),where='mid',label="CEV",color="red")
plt.savefig("% CEV_grades_LDA",bbox_inches="tight",dpi=400)


# In[26]:


#plotting PCA scatterplot (Grades)

#plotting the first two PC s
plt.figure(figsize=(6, 6),facecolor="white") 
scatter6=plt.scatter(X_red[:, 0],X_red[:, 1],c=df_labels_grades,cmap=cmap1,s=15)

plt.legend(handles=scatter6.legend_elements()[0],labels=["BOP","BOPF","PEK","BOP1","PF1","FF1","PD","FBOP","BOPS","FBOP1"],fontsize=6)

plt.title("First 2 PC directions: Grades (99.10% CEV)",fontweight="bold")
plt.xlabel("PC1",fontweight="bold")
plt.ylabel("PC2",fontweight="bold")
plt.savefig("First 2 PCs_grades",bbox_inches="tight",dpi=400)


# In[27]:


plt.rcParams.update({'font.size': 14})
fig7 = plt.figure(1, figsize=(6,6),facecolor="white")
ax7 = fig7.add_subplot(111, projection="3d")

scatter7=ax7.scatter(
    X_red[:, 0],
    X_red[:, 1],
    X_red[:, 2],
    c=df_labels_grades,
    s=15,
    cmap=cmap1
)

plt.legend(handles=scatter7.legend_elements()[0],labels=["BOP","BOPF","PEK","BOP1","PF1","FF1","PD","FBOP","BOPS","FBOP1"],fontsize=6)

ax7.set_title("First three PC directions: Grades (99.53% CEV)",fontweight="bold")
ax7.set_xlabel("PC1",fontweight="bold")
ax7.set_ylabel("PC2",fontweight="bold")
ax7.set_zlabel("PC3",fontweight="bold")
plt.savefig("First 3 PCs_grades",bbox_inches="tight",dpi=400)


# In[31]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering


# In[29]:


def optimal_clusters_spectral(X,Y):
    'X: dataset, Y-list of n_componenets to run through'
    scores=np.zeros(len(Y))
    j=0
    for i in Y:
        model_i=SpectralClustering(n_clusters=i, affinity='nearest_neighbors')
        model_i.fit(X)
        clusters_i = model_i.labels_
        score_i=silhouette_score(X, clusters_i)
        scores[j]=score_i
        j=j+1
    print(scores)
        
    max_score_index=np.argmax(scores)
    print(f"The max score is {scores[max_score_index]} and it is when n={Y[max_score_index]}")
    
    return i


# In[32]:


i=optimal_clusters_spectral(x_lda,[2,3,4,5,6,7,8,9,10])
j=optimal_clusters_spectral(x_lda2,[2,3,4,5,6,7,8,9,10])


# In[33]:


#spectral clustering on the LDA dataset

from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D



# Create a spectral clustering model with 3 clusters
model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')

# Fit the model to the data
model.fit(x_lda2)

# Get the cluster assignments for each data point
clusters = model.labels_

# Create a 3D figure
fig7 = plt.figure(figsize=(12,12),facecolor="white")
ax7 = fig7.add_subplot(111, projection='3d')

# Create a 3D scatter plot of the data colored by cluster assignment

colors2=["red","blue","green"]
colors2=ListedColormap(colors2)
ax7.scatter(x_lda2[:, 0], x_lda2[:, 1], x_lda2[:, 2], c=clusters,cmap=colors2,s=15)

#to label each point
for i, label in enumerate(df_labels_grades2):
    ax7.text(x_lda2[:, 0][i], x_lda2[:, 1][i], x_lda2[:, 2][i], label,fontsize=8)



# Set the axis labels
ax7.set_title("Clustered data (LDA: Grades-Spectral)",fontweight="bold",fontsize=16,pad=1)
ax7.set_xlabel('Component 1',fontweight="bold",fontsize=14)
ax7.set_ylabel('Component 2',fontweight="bold",fontsize=14)
ax7.set_zlabel('Component 3',fontweight="bold",fontsize=14)
ax7.view_init(elev=4, azim=115)
plt.savefig("Clustered data_LDA_Grades_spectral.png",bbox_inches="tight",dpi=400)






# In[34]:


# Create a spectral clustering model with 2 clusters
model2 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')

# Fit the model to the data
model2.fit(x_lda)

# Get the cluster assignments for each data point
clusters2 = model2.labels_

# Create a 3D figure
fig8 = plt.figure(figsize=(12,12),facecolor="white")
ax8 = fig8.add_subplot(111, projection='3d')

# Create a 3D scatter plot of the data colored by cluster assignment

colors2=["red","blue"]
colors2=ListedColormap(colors2)
ax8.scatter(x_lda[:, 0], x_lda[:, 1], x_lda[:, 2], c=clusters2,cmap=colors2)

#to label each point
for i, label in enumerate(df_labels_grades2):
    ax8.text(x_lda[:, 0][i], x_lda[:, 1][i], x_lda[:, 2][i], label,fontsize=10)



# Set the axis labels
ax8.set_title("Clustered data (LDA: Locations-Spectral)",fontweight="bold",fontsize=16)
ax8.set_xlabel('Component 1',fontweight="bold",fontsize=14)
ax8.set_ylabel('Component 2',fontweight="bold",fontsize=14)
ax8.set_zlabel('Component 3',fontweight="bold",fontsize=14)
ax8.view_init(elev=4, azim=115)
plt.savefig("Clustered data_LDA_Locations_spectral.png",bbox_inches="tight",dpi=400)
# ax7.view_init(elev=95, azim=-89)


# In[35]:


#determining the optimimum cluster number for k means

from sklearn.cluster import KMeans

def optimal_clusters(X,Y):
    'X: dataset, Y-list of n_componenets to run through'
    scores=np.zeros(len(Y))
    j=0
    for i in Y:
        model_i=KMeans(n_clusters=i)
        model_i.fit(X)
        clusters_i = model_i.labels_
        score_i=silhouette_score(X, clusters_i)
        scores[j]=score_i
        j=j+1
    print(scores)
        
    max_score_index=np.argmax(scores)
    print(f"The max score is {scores[max_score_index]} and it is when n={Y[max_score_index]}")
    
    return i
        
        


# In[36]:


i=optimal_clusters(x_lda[:,0:3],[2,3,4,5,6,7,8,9,10])
j=optimal_clusters(x_lda2[:,0:3],[2,3,4,5,6,7,8,9,10])



# In[38]:


# Create a kmeans clustering model with 2 clusters
model1 = KMeans(n_clusters=3)

# Fit the model to the data
model1.fit(x_lda[:,0:3])

# Get the cluster assignments for each data point
clusters1 = model1.labels_

# Create a 3D figure
fig9 = plt.figure(figsize=(12,12),facecolor="white")
ax9 = fig9.add_subplot(111, projection='3d')

# Create a 3D scatter plot of the data colored by cluster assignment

colors2=["red","blue","green"]
colors2=ListedColormap(colors2)
ax9.scatter(x_lda[:, 0], x_lda[:, 1], x_lda[:, 2], c=clusters1,cmap=colors2,s=15)

#to label each point
for i, label in enumerate(df_labels_grades2):
    ax9.text(x_lda[:, 0][i], x_lda[:, 1][i], x_lda[:, 2][i], label,fontsize=8)



# Set the axis labels
ax9.set_title("Clustered data (LDA: Locations)",fontweight="bold",fontsize=16,pad=1)
ax9.set_xlabel('Component 1',fontweight="bold",fontsize=14)
ax9.set_ylabel('Component 2',fontweight="bold",fontsize=14)
ax9.set_zlabel('Component 3',fontweight="bold",fontsize=14)
ax9.view_init(elev=4, azim=115)

plt.savefig("Clustered data_LDA_Locations.png",bbox_inches="tight",dpi=400)


# In[39]:


# Create a kmeans clustering model with 6 clusters
model2 = KMeans(n_clusters=6)

# Fit the model to the data
model2.fit(x_lda2[:,0:3])

# Get the cluster assignments for each data point
clusters2 = model2.labels_

# Create a 3D figure
fig10 = plt.figure(figsize=(12,12),facecolor="white")
ax10 = fig10.add_subplot(111, projection='3d')

# Create a 3D scatter plot of the data colored by cluster assignment

colors3=["red","blue","green","brown","black","yellow"]
colors3=ListedColormap(colors3)
ax10.scatter(x_lda2[:, 0], x_lda2[:, 1], x_lda2[:, 2], c=clusters2,cmap=colors3,s=15)

#to label each point
for i, label in enumerate(df_labels_grades2):
    ax10.text(x_lda2[:, 0][i], x_lda2[:, 1][i], x_lda2[:, 2][i], label,fontsize=8)



# Set the axis labels
ax10.set_title("Clustered data (LDA: Grades)",fontweight="bold",fontsize=16,pad=1)
ax10.set_xlabel('Component 1',fontweight="bold",fontsize=14)
ax10.set_ylabel('Component 2',fontweight="bold",fontsize=14)
ax10.set_zlabel('Component 3',fontweight="bold",fontsize=14)
ax10.view_init(elev=4, azim=115)

plt.savefig("Clustered data_LDA_Grades.png",bbox_inches="tight",dpi=400)

