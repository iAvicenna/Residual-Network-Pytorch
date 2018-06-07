
################################################################################
#This is a classification code (works on GPU) to classify images from CIFAR10. #
#This database contains images of size 3 x 32 x 32 (50000 training and 10000   #
#test examples) with 10 classes ('plane', 'car', 'bird', 'cat', 'deer', 'dog', #
#'frog', 'horse', 'ship', 'truck'). The architecture is a residual network     #
#using basic blocks. Using a single basic block achieves accuracy up to %80    #
#in several minutes.  Using multiple basic blocks (ResNet-18 architecture)     #
#achieves %92-93 percent accuracy in about 200 cycles. The code has been       # 
#mainly tested in google colaboratory. Therefore command line arguements are   #
#not available yet. The original paper for this work is in:                    # 
#arxiv.org/abs/1512.03385                                                      #
#                                                                              #
#Notes:                                                                        #
#1- Accuracy, recall, precision, F1 norm for each class are printed.           #
#2- A manual scheduler function is implemented to change the learning rate.    #
#   Exponentially decreasing or stepwise options are available.                #
#3- This code has been tested on pytorch 0.3. In 0.4 the way the scalars are   # 
# handled is changed so you need to change .sum() to .sum().item(),  .data[0]  #
# to .item()                                                                   #
################################################################################


# install some packages which are not available in google colab
# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
!pip install gputil
!pip install psutil
!pip install humanize


#torch, numpy and matplot related packages
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init

#other packages
import pickle
import time
import psutil
import humanize
import os
import GPUtil as GPU



#These two functions convert an object from height x width x color
#dimensions to color x height x width and vice versa. They are used
#during whitening.
class HWCtoCHW(object):

    def __call__(self, tensor):
        return tensor.transpose(0, 1).transpose(1, 2).contiguous()
      
class CHWtoHWC(object):  

    def __call__(self, tensor):
        return tensor.transpose(1, 2).transpose(0, 1).contiguous()      

def computeZCAMAtrix(dataname):

    #This function computes the ZCA matrix for a set of observables X where
    #rows are the observations and columns are the variables (M x C x W x H matrix)
    #C is number of color channels and W x H is width and height of each image
    #It is input is the data set name and output whitening matrix, per channel
    #mean and std. Everything is calculated after normalizing to [0,1] scale.
    
    if(dataname=='CIFAR10'):
        root = 'cifar10/' 
       
    
        temp= datasets.CIFAR10(root = root,
                               train = True,
                               download = True)
        
  
    #normalize the data to [0 1] range
    #subtract mean and normalize std
    temp.train_data=temp.train_data/255
    mean=(temp.train_data.mean(axis=(0,1,2)))
    std=(temp.train_data.std(axis=(0,1,2))) 
    temp.train_data=np.multiply(1/std,np.add(temp.train_data,-mean))
    

    
    
    #reshape data from M x C x W x H to M x N where N=C x W x H 
    X =  temp.train_data
    
    X = X.reshape(-1, 3072)
    
    # compute the covariance 
    cov = np.dot(np.transpose(X,(1,0)),X)   # cov is (N, N)
    
    # singular value decomposition
    U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,1) V is (N,N)
    # build the ZCA matrix which is (N,N)
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
  


    return (torch.from_numpy(zca_matrix).float(), mean, std)  

  
#This function just gets mean and std of a dataset  
#normalized to [0,1] range
def getMeanAndStd(dataname):

   
    if(dataname=='CIFAR10'):
        root = 'cifar10/' 
       
    
        temp= datasets.CIFAR10(root = root,
                               train = True,
                               download = True)
        
  
    #normalize the data to [0 1] range
    #subtract mean and normalize std
    temp.train_data=temp.train_data/255
    mean=(temp.train_data.mean(axis=(0,1,2)))
    std=(temp.train_data.std(axis=(0,1,2))) 

    return (mean, std)  

#This function is used to set specific parameters per layer.
#At this particular code it is used to set weight decays for biases to 0.
def setParams(network,state):
  
  params_dict = dict(network['model'].named_parameters())
  params=[]
  weights=[]

  
  for key, value in params_dict.items():

      if key[-4:] == 'bias':
          
          params += [{'params':value,'weight_decay':0.0}]
          
      else: 
          
          params +=  [{'params': value, 'weight_decay':state['weight decay']}]
  

  return params
  

#given batch_size and data set name (currently only CIFAR10)  
#defined data loaders
def getData(batch_size,dataname,Z,mean,std): 
  
  
    if(batch_size<=0):
        batch_size=256

    if(dataname=='CIFAR10'):
        root = 'cifar10/'
        
        
        #this transformation is used to augment data,normalize it and then whiten
        #it if state['whitening'] is set to whitening
        
        if state['whitening'] != 'None' :
            transform_train = transforms.Compose(
            [                  
              transforms.RandomRotation(5),
              transforms.RandomHorizontalFlip(),
              transforms.RandomCrop(32, padding=4),
              transforms.ToTensor(),
              transforms.Normalize(mean , std),  
              HWCtoCHW(),
              transforms.LinearTransformation(Z),   
              CHWtoHWC(),                         
              ])

        #for test set we do not apply the random transformations
            transform_test = transforms.Compose(
            [     
              transforms.ToTensor(),
              transforms.Normalize(mean , std),  
              HWCtoCHW(),
              transforms.LinearTransformation(Z),   
              CHWtoHWC(),
              ])
          
        else:   
            transform_train = transforms.Compose(
            [                  
              transforms.RandomRotation(5),
              transforms.RandomHorizontalFlip(),
              transforms.RandomCrop(32, padding=4),
              transforms.ToTensor(),
              transforms.Normalize(mean , std),                         
              ])

            transform_test = transforms.Compose(
            [     
              transforms.ToTensor(),
              transforms.Normalize(mean , std),  
              ])
        
        
        #get the training and test sets
        training_set = datasets.CIFAR10(root = root,
                                  transform = transform_train,
                                  train = True,
                                  download = True)
        
        test_set = datasets.CIFAR10(root = root,
                                  transform = transform_test,
                                  train = False,
                                  download = True)
 
    else:
        print('Currently only CIFAR10 is allowed, terminating program')
        sys.exit()
        

    
    #DataLoaders are used to iterate over the database images in batches rather
    #one by one. gradients and minimizations are carried out over the whole batch rather
    #than one by one. this decreases variation in computation of gradients.
    training_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  )

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                             )

    
    return (training_set,test_set,training_loader,test_loader)


    
#track ram usage in google colab
def ramUsage ():
  
    !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
    

    
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]
    def printm():
        process = psutil.Process(os.getpid())
        print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
   
    printm()


# A scheduler function used to change learning rate based on given conditions.
# Presently two conditions are available: step-wise and exponential.
def schedule (e,state,optimizer):
  
    if(state['scheduler']=='step'):  

        if(e in state['schedule']):
            print('New learning rate: ')
              
            index=state['schedule'].index(e)
            for param_group in network['optimizer'].param_groups:
                param_group['lr']*=state['gamma'][index]
                state['learning rate']*=state['gamma'][index]
                     
            print('%.5f'% param_group['lr'])
                  
                  
    elif(state['scheduler']=='exponential'):
              
     
        print('New learning rate: ')
        
        for param_group in network['optimizer'].param_groups:
            param_group['lr']=np.exp(-state['decay factor']*e)*state['learning rate']
            
        print('%.5f'% param_group['lr'])  
            
    else:
      
         print('Warning: Unknown scheduler name. Learning rate will not be changed.')
        
        
    
    
#this function is used to test the accuracy of the model     
#over the test set. it also computes precision, recall and accuracy over each 
#class.
def test(network,state,isCuda,data):
  
    network['model'].eval()
    #calculate the accuracy of our model over the whole test set in batches
    correct = 0
    precision =[0,0,0,0,0,0,0,0,0,0]
    recall =[0,0,0,0,0,0,0,0,0,0]
    predicted_positives=[0,0,0,0,0,0,0,0,0,0]
    actual_positives=[0,0,0,0,0,0,0,0,0,0]
    F1=[0,0,0,0,0,0,0,0,0,0]
    confusion_matrix=np.zeros((10,10))
    
    for x, y in data['test loader']:
       
        
      
        if(isCuda==1):
            x, y = Variable(x).cuda(), y.cuda()
        if(isCuda==0):
            x, y = Variable(x), y
          
        h =  network['model'].forward(x)
        
        pred = h.data.max(1)[1]
        
        c = (pred == y).squeeze()
        correct += pred.eq(y).sum()

        
        for k in range (0,len(c)):
            precision[y[k]]+= c[k]
            recall[y[k]]+=c[k]
            predicted_positives[pred[k]]+=1
            actual_positives[y[k]]+=1
            confusion_matrix[y[k],pred[k]]+=1
            
    
    for k in range (0,10):
              
        confusion_matrix[k,:]=confusion_matrix[k,:]/actual_positives[y[k]]
      
        if(predicted_positives[k]>0):
            precision[k]=precision[k]/predicted_positives[k]
        else: 
            precision[k]=0
            
        if(actual_positives[k]>0):    
            recall[k]=recall[k]/actual_positives[k]
        else:
            recall[k]=0
            
        if(precision[k]+recall[k]>0):
            F1[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
        else:
            F1[k]=0
        

    return (correct/len(data['test set']), precision, recall,F1,confusion_matrix)

#the following function prints accuracy scores for each class and confuson matrix
#as table compatible with github

def printTable (state,classes,e):
  
    print('| Class  | Score                                       |')
    print('|--------', end="") 
    print('|---------------------------------------------|') 
    
    for k in range (0, 10):
        print('| %s  | Precision:%.2f, Recall: %.2f, F1 norm: %.2f | ' %(classes[k], state['precision'][e+1][k], state['recall'][e+1][k], state['F1 norm'][e+1][k]) )
		   
      
    print('\n')  
    print('| class  ', end="")
    for k in range (0, 10):
        print('| %s  '%classes[k], end="")
    print('|')  
    for k in range (0, 11):
        print('|--------', end="") 
    print('|')    

    for k in range (0, 10):                
        print('| %s  |'%classes[k], end="")
        for s in range (0, 10):  
            print(' %.5f|' %state['confusion matrix'][k,s], end="")
        print('')         

#this are the two types of the basic blocks in a residual network. The residual network
#in this code is built by concatenating several such blocks together.
#Basic blocks are of the form x -> D(x) + F(x), where D(x) is x downsampled
#to the same dimensions as F(x) by a single convolution and F(x) is collection of 
#successive operations involving several convolutions and batchnorms.
class BasicResBlock1(nn.Module):
    def __init__(self, input, output, downsample, stride=1):
       super(BasicResBlock1, self).__init__()
       
       self.conv1 = torch.nn.Conv2d(input,output,kernel_size=3,stride=stride,padding=1, bias=False)
       self.batchNorm1 = torch.nn.BatchNorm2d(output)
       self.conv2 = torch.nn.Conv2d(output,output,kernel_size=3,padding=1, stride=1, bias=False)
       self.downsample=downsample
       
       #applied to the residual to downsample
       
      
       
        
    def forward(self,x1):       
      
       residual = self.downsample(x1)
     
  
       x2 = self.conv1(x1)
       x2 = self.batchNorm1(x2)
       x2 = F.relu(x2,inplace=True) 
       x2 = self.conv2(x2)
       
      
       x2+= residual

      
     
       return x2
       
class BasicResBlock2(nn.Module):
    def __init__(self, input, output):
       super(BasicResBlock2, self).__init__()
       
       self.conv1 = torch.nn.Conv2d(input,output,kernel_size=3,stride=1,padding=1, bias=False)
       self.batchNorm1 = torch.nn.BatchNorm2d(input)
       self.conv2 = torch.nn.Conv2d(output,output,kernel_size=3,padding=1, stride=1, bias=False)   
       self.batchNorm2 = torch.nn.BatchNorm2d(output) 
       self.batchNorm3 = torch.nn.BatchNorm2d(output) 
        
    def forward(self,x1):       
       
        
       residual = x1
        
       
       x2 = self.batchNorm1(x1)
       x2 = F.relu(x2,inplace=True)  
       x2 = self.conv1(x1);
        
       x2 = self.batchNorm2(x2)
       x2 = F.relu(x2,inplace=True)  
       x2 = self.conv2(x2)
       

       x2+= residual
        
       x2 = self.batchNorm3(x2)  
       x2 = F.relu(x2, inplace=True)
      
     
       return x2       
  

#Below we define the residual network class.
class ResNet(nn.Module):
    def __init__(self,width, number_of_blocks):
        super(ResNet, self).__init__()
        
        #these are the inital layers applied before basic blocks
        
        self.conv1 = torch.nn.Conv2d(3,width,kernel_size=3,stride=1,padding=1, bias=False)         
        self.batchNorm1 = torch.nn.BatchNorm2d(width) 
        self.relu1 = nn.ReLU(inplace=True)
        

        #resLayer1 is the basic block for the residual network that is formed by
        #concatenating several basic blocks of increasing dimensions together.
        self.downsample1=torch.nn.Conv2d(width,2*width,kernel_size=1,stride=1,bias=False)  
        self.downsample2=torch.nn.Conv2d(2*width,4*width,kernel_size=1,stride=2,bias=False)
        self.downsample3=torch.nn.Conv2d(4*width,8*width,kernel_size=1,stride=2,bias=False)
        
        self.resLayer1=[]
        self.resLayer1.append(BasicResBlock1(width,2*width,self.downsample1,1))
        for x in range (0, number_of_blocks[0]) :      #stage1
            self.resLayer1.append(BasicResBlock2(2*width,2*width))
        self.resLayer1=nn.Sequential(*self.resLayer1)
        
        self.resLayer2=[]
        self.resLayer2.append(BasicResBlock1(2*width,4*width,self.downsample2,2)) #stage2
        for x in range (0, number_of_blocks[1]) :
            self.resLayer2.append(BasicResBlock2(4*width,4*width))
        self.resLayer2=nn.Sequential(*self.resLayer2)
        
        self.resLayer3=[]
        self.resLayer3.append(BasicResBlock1(4*width,8*width,self.downsample3,2)) #stage3
        for x in range (0, number_of_blocks[2]) :
            self.resLayer3.append(BasicResBlock2(8*width,8*width))
        self.resLayer3=nn.Sequential(*self.resLayer3)   

        
        self.avgpool1 = torch.nn.AvgPool2d(8,stride=1)
        
        #define the final linear classifier layer
        self.full1=nn.Linear(8*width,10)
        
        #weight initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out')
                
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant(m.weight, 1)
                torch.nn.init.constant(m.bias, 0)

            elif isinstance(m, nn.Linear):   
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out')
                torch.nn.init.constant(m.bias, 0)
        
    #define the forward run for the input data x    
    def forward(self, x):
    
        #initial layers before basic blocks    
        x = self.conv1(x)  
        x = self.batchNorm1(x)
        x = self.relu1(x)
  
        
        
        #residual layers and then average pooling
        x = self.resLayer1(x);
        x = self.resLayer2(x);
        x = self.resLayer3(x);
        #x = self.resLayer4(x);
     
        x = self.avgpool1(x)
     
    
        
        #linear classifier layer (since we
        #use CrossEntropyLoss for the loss function
        #which already has logsoftmax incorporated inside
        #we dont have any activation function here.)
        x = x.view(x.size(0), -1)
    
        
        x = self.full1(x)
        return x 
        

#this is the training function. 
#rnn is the defined network. cost_criterion is the cost function
#optimizer is the gradient descent method we choose, lr is the learning rate
#gamma is the learning rate decay factor and schedule is the list of cycles
#which tells when the learning date should decay.

def train(network,state,isCuda,data):
  
    costs=[] #store the cost each epoch
    plt.ion()
    fig = plt.figure()
    
    pred=0 #predictions of the model over the training set
    correct=0 #correct predictions

    average_cost=0
    #some temporary arrays used in recording accuracy, recall etc
    tempA=0;
    tempB=[0,0,0,0,0,0,0,0,0,0];
    tempC=[0,0,0,0,0,0,0,0,0,0];
    tempD=[0,0,0,0,0,0,0,0,0,0];
    file_output = open('ResNetOutput', 'w')
  
    for e in range(state['cycles']): #cycle through the database many times

        print('\n\n Cycle: ',e+1)
 
              
        t1 = time.clock()

        schedule (e,state,network['optimizer'])
      
      
        
          
        #put network in train mode
        network['model'].train()
         
        #following for loop cycles over the training set in batches
        #of batch_number=128 using the training_loader object
        for i, (x, y) in enumerate(data['training loader'] ,0):
        

         
         
            #here x,y will store data from the training set in batches 
            if(isCuda==1):
                x, y = Variable(x.cuda()), Variable(y.cuda())
            if(isCuda==0):
                x, y = Variable(x), Variable(y)
                
            h = network['model'].forward(x) #calculate hypothesis over the batch

            
            cost = network['cost criterion'](h, y) #calculate cost the cost of the results
            
            network['optimizer'].zero_grad() #set the gradients to 0
            cost.backward() # calculate derivatives wrt parameters
            network['optimizer'].step() #update parameters
            
            average_cost=average_cost+cost.data[0] #add the cost to the average cost
            

            pred = h.data.max(1)[1]   #calculate correct predictions over this batch
            correct += pred.eq(y.data).sum()
            del x, y, h, cost
        
        
        t2 = time.clock()
        print('Training time for cycle %d is %.2f  '%(e,t2-t1), end="")
        
        
        
        state['training accuracy'].append(correct/len(data['training set'])) #training accuracy within one cycle
        pred=0
        correct=0
        average_cost=average_cost*state['batch size']/len(data['training set']) #average training cost within one cycle
        
       
        #get precision, recall, F1 norm, test accuracy
        (tempA, tempB, tempC, tempD, state['confusion matrix']) = test(network,state,isCuda,data)
    
    
        
    
        state['test accuracy'].append(tempA)
        state['precision'] = np.vstack((state['precision'],tempB))
        state['recall'] = np.vstack((state['recall'],tempC))
        state['F1 norm'] = np.vstack((state['F1 norm'],tempD))
        
        #compute best test accuracy so far
        if(state['test accuracy'][-1]>state['best test accuracy']):
            state['best test accuracy']=state['test accuracy'][-1]
            torch.save(network['model'].state_dict(), 'mynet_trained')
            

        #compute average test accuracy over the last 10 cycles    
        len1 = len(state['test accuracy'])   
        if(len1>9):          
            state['average test accuracy'].append(sum(state['test accuracy'][len1-10:len1])/10)
        t3 = time.clock()  
        print('Cost calculation time is %.2f'%(t3-t2), end="")   
          
        #plot results                                           
        if(e%3==2):
            len1=len(state['test accuracy'])
            len2=len(state['training accuracy'])  
            len3=len(state['average test accuracy'])
            plt.xlabel("Cycles")
            plt.ylabel("Accuracy")
            plt.title("Test and Train Accuracy vs Cycles")
            
            axes = plt.gca()
            axes.set_xlim([0,state['cycles']])
            axes.set_ylim([0,1])
            
            plt.plot(range(0,len1), state['test accuracy'],'g',label="test accuracy")
            plt.plot(range(1,len2+1), state['training accuracy'], 'b', label="training accuracy")
            if(e>9):
                plt.plot(range(10,len3+10), state['average test accuracy'], 'r--', label="average test accuracy")
            
            plt.legend()
            plt.show()
            

            if(e>9):
                file_output.write('\tBest test accuracy: %.5f, current test accuracy: %.5f, average test accuracy %.5f, training accuracy: %.5f (cost:%.5f)' %(state['best test accuracy'] ,state['test accuracy'][-1],state['average test accuracy'][-1],state['training accuracy'][-1],average_cost))       
                print('\tBest test accuracy: %.5f, current test accuracy: %.5f, average test accuracy %.5f, training accuracy: %.5f (cost:%.5f)' %(state['best test accuracy'] ,state['test accuracy'][-1],state['average test accuracy'][-1],state['training accuracy'][-1],average_cost)) 
            else:
                file_output.write('\tBest test accuracy: %.5f, current test accuracy: %.5f, training accuracy: %.5f (cost:%.5f)' %(state['best test accuracy'] ,state['test accuracy'][-1],state['training accuracy'][-1],average_cost))       
                print('\tBest test accuracy: %.5f, current test accuracy: %.5f, training accuracy: %.5f (cost:%.5f)' %(state['best test accuracy'] ,state['test accuracy'][-1],state['training accuracy'][-1],average_cost)) 

            ramUsage()
  
        if(e%20==19):
            printTable(state,network['classes'],e)
          
            for k in range (0, 10):
		             file_output.write('\t Class: %s, Precision:%.2f, Recall: %.2f, F1 norm: %.2f  ' %(network['classes'][k], state['precision'][e+1][k], state['recall'][e+1][k], state['F1 norm'][e+1][k]) )

            
            
        average_cost=0   
            
    f = open("mynet_state.pkl","wb")
    pickle.dump(state,f)
    f.close
    file_output.close


#the main function :p old habits die hard
  
if __name__ == '__main__':
  

    isCuda=torch.cuda.is_available() #check if CUDA is available

    #these are the master variables which contain all most all the other variables as lists
    state={} #this is the list that contains state of the system including all the parameters of the program as well as accuracy values and such
    network={} #this is the list that contains the network and functions related to the network (optimizer and cost function)
    data={} #this is the list that contains test and training datasets and dataloaders
    
    state['whitening']='None' #set to 'None' if you dont want whitening.
    
    #compute data whitening matrix and get the loaders etc
    state['dataname']='CIFAR10'

    
    Z=torch.Tensor()
    if(state['whitening']=='ZCA'):
        (Z,mean,std)=computeZCAMAtrix(state['dataname']) #calculate whitening matrix
    else:
        (mean,std)=getMeanAndStd(state['dataname'])
        
    state['batch size']=256
    (data['training set'],data['test set'],data['training loader'],data['test loader']) = getData(state['batch size'],state['dataname'],Z,mean,std)  
    
    #set parameters
    state['learning rate']=0.1 #initial learning rate
    state['momentum']=0.9  #momentum variable for the gradient descent
    state['weight decay']=0.0005  #weight decay rate
    state['cycles'] = 300 #number of cycles that the training runs over the database
    state['params'] = [] #used to define per layer parameters such as different weight decays or learning parameters for each layer

        
    state['number of blocks']=[1, 1, 1] #number of extra blocks in each stage of residual network (on top of the transition block)
    state['width']= 16 #width of the initial convolutional layer
    state['scheduler'] = 'step'  #step or exponential scheduler is available
    
    if(state['scheduler']=='step'):
        state['schedule']=[60, 120, 160] #at each cycle in this list lr -> gamma*lr
        state['gamma']=[0.2, 0.2, 0.2] #the multplicative factor used to decrease learning rate as lr -> gamma*lr    
    if(state['scheduler']=='exponential'):    
        state['decay factor']=0.01 #decay of learning rate if exponential is chosen
    
    state['training accuracy']=[] #list of training accuracies over time
    state['test accuracy']=[] #accuracy of the model over the test set
    state['average test accuracy']=[] #tail of the Cesaro sum of test accuracy
    state['average training accuracy']=[] #Cesaro sum of training accuracy
    state['best test accuracy']=0 #best test accuracy over all cycles
    state['precision']=[]  #list of true positives/predicted positives for each category over time of the form training accuracy[category][cycle]
    state['recall']=[]   #list of true positives/actual positives for each category over time of the form recall [category][cycle]
    state['F1 norm']= []  #F1 norm for each category which is 2*precision*recall/(precision+recall)
    state['confusion matrix']=np.zeros((10,10))

    
    #initialize network variables
    if( isCuda==1):
        network['model']= ResNet(state['width'],state['number of blocks']).cuda()
    else:
        network['model']= ResNet(state['width'],state['number of blocks'])
        
    #print(network['model'])    
    
    state['params']=setParams(network,state)
    network['cost criterion']=  torch.nn.CrossEntropyLoss() #cost function
    network['optimizer'] =  torch.optim.SGD(state['params'], momentum=state['momentum'], lr=state['learning rate'],nesterov=True) #optimizer
    #[{'params': resLayer4.bias, 'weight_decay': 0}, {'params': resLayer3.bias, 'weight_decay': 0}, {'params': resLayer2.bias, 'weight_decay': 0},   {'params': resLayer1.bias, 'weight_decay': 0}, {'params': batchNorm1.bias, 'weight_decay': 0},{'params': full1.bias, 'weight_decay': 0} ,{'params': batchNorm1.bias, 'weight_decay': 0} ]
    
    if(state['dataname']=='CIFAR10'):        
        network['classes'] = ('plane', 'car  ', 'bird ', 'cat  ', 'deer ', 'dog  ', 'frog ', 'horse', 'ship ', 'truck') #classes in CIFAR10
    
    
    
    #calculate the initial success of the model
    (temp,state['precision'],state['recall'],state['F1 norm'],state['confusion matrix'])=test(network,state,isCuda,data)
    state['test accuracy'].append(temp)

    ramUsage ()
    train(network,state,isCuda,data) #start training
   
    
