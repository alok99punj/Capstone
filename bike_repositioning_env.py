#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Explanation of the environment made for repositioning of e-bikes in a dock based bike sharing service based out of Bangalore 
#### The e-bikes are repositioned in a proactive way, the e-bikes are repositioned to maintain the balance between supply and demand of bikes in a station 
#### The e-bikes are repositioned using trucks ( having a capacity of 20 bikes), the capacity of each station considered is 30 bikes each 
#### the stattion acts as an agent in this environment
#### the state vector of the env contains the following - the bikes at each station  , the pickup and dropoff vectors and the charge level of each bikes
#### the size and shape of each vector is mentioned below 
####   ##Assumptions
# 1. the capacity of the truck is 20 
# 2. the capacity of the station is 30 
# 3. the truck follows a fixed path in a round robbin fashion 
# 4. the truck completes one side fixed path trip in one time slot 
# 5. the truck is initially empty 
# 6. the station loads only 50% of its capacity into the truck 
# 7. the truck unloads bikes until the the station reaches 30 bikes
# 8. 3-4 kW and 2-3 kW charge bikes are loaded to the truck first( if the requirement not met then low charge bikes uploaded)
# 9. unloading is done in a similar manner by unloading the high charge ones first 
# 10. the bike vec has been considered for a time slot after removing the pickups and dropoffs of that particular station 


### the truck moves along a fixed path mentioned by the vector(num_b) which is constructed after taking into consideration the distance and time to complete a one way trip
### rewards are based on the following- 
### if after repositioning the station has a balnced supply and demand then -> positive reward awarded 
### else -> negetive reward 
### if the charge levels of bikes( high + mid) = the number of pickups then positive reward awarded 
### else -> negetive 
### if a station calls a truck for loading bikes but fails to deliver then -> negetive reward 
### else -> positive 
### if a station calls a truck for unloading but fails to unload( happens when sattion has max bikes) then -> negetive reward awarded 
### else -> positive reward awarded 


# In[2]:


# importing necessary libraries 
import random 
import gym 
import pandas as pd 
import numpy as np 


# In[3]:


#####   These declarations will be replaced by the dataset values #####

### info on the shape of the variables declared below 
## num_b(x,y) -> x is the number of timeslots , y is the number of stations
## charge_b(x,y,z) -> x is the number of timeslots, y is the number of stations and z is the number of vehicles at a charge level
## pickup_b and dropoff_b -> similar to num_b


# In[4]:


num_b=np.array([[10,4,30,15,20],[15,23,4,22,1],[12,2,3,12,3]])#### initializing the array which contains the number of bikes at each of the 5 stations at any given point in time_slot(excluding the pickups and dropoffs)
charge_b=np.array([[[5,2,3],[4,0,0],[10,10,10],[10,5,0],[20,0,0]],[[15,0,0],[20,3,0,],[4,0,0],[20,2,0],[1,0,0]],[[12,0,0],[2,0,0],[2,1,0],[12,0,0],[2,1,0]]])
pickup_b=np.array([[5 ,3 ,1 ,24,1],[20,1,3,23,12],[1,23,1,32,1]])
dropoff_b=np.array([[10,23,12,34,2],[23,3,1,3,1],[32,3,2,1,3]])
#### these declarations are for the 3 time slots of 30 min(say)


# In[5]:


### verifying the shape of each array 
print("num_b",num_b.shape)### the x-axis denotes the number of time slots and the y-axis denotes the number of stations 
print("charge_b",charge_b.shape)### the x-axis denotes the number of time slots , the y - axis denotes the number of stations and the z - axis denotes teh number of bikes at the three different charge levels 
print("load_b",pickup_b.shape)### the x axis denotes the number of time slots and the y - axis denotes the number of stations 
print("unload_b",dropoff_b.shape)### the x- axis denotes the number of time slots and the y - axis denotes the number of stations 

#### note:: This declaration will be replaced with the dataset values which achal provides 


# In[6]:


s_v=np.array([num_b,pickup_b,dropoff_b])### defining the state vector 
#print(s_v.shape)
#### charge vector contains a different shape will be take care off individually 
##print(s_v)
#arr=[[11,1,3],[12,3,1],[13,23,1]]
print(s_v)## station number and then charge level is denoted by j


# In[7]:


class Bike_rep_env:
    def __init__(self,num_b,charge_b,pickup_b,dropoff_b,s_v):
        self.num_b=num_b
        self.charge_b=charge_b
        self.pickup_b=pickup_b
        self.dropoff_b=dropoff_b
        self.total_reward=0
        self.action_history=[]
        self.total_timeslots=0
        self.s_v=s_v
        self.timeslots=3
        self.initial=s_v## preserving the initial state
        self.initial_charge=charge_b ## preserving the initial charge state 
    ##Assumptions
# 1. the capacity of the truck is 20 
# 2. the capacity of the station is 30 
# 3. the truck follows a fixed path in a round robbin fashion 
# 4. the truck completes one side fixed path trip in one time slot 
# 5. the truck is initially empty 
# 6. the station loads only 50% of its capacity into the truck 
# 7. the truck unloads bikes until the the station reaches 30 bikes
# 8. 3-4 kW and 2-3 kW charge bikes are loaded to the truck first( if the requirement not met then low charge bikes uploaded)
# 9. unloading is done in a similar manner by unloading the high charge ones first 
# 10. the bike vec has been considered for a time slot after removing the pickups and dropoffs of that particular station 




    def nextState(self,action,truck,truck_charge,flag):
        if(self.total_timeslots%2!=0):
            for i in range(len(action)):
           # print(truck_charge)
                #print(truck)
                if action[i]==0:### No action
                    continue
                
                
                if action[i]==1: ### action of loading bikes into the truck
                    if (int(self.s_v[0][self.total_timeslots][i]*0.50))>0 and (truck+int(s_v[0][self.total_timeslots][i]*0.50)<=20):
                        truck=truck+int(self.s_v[0][self.total_timeslots][i]*0.50)    ### updating the truck after the load action 
                        self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]-int(self.s_v[0][self.total_timeslots][i]*0.50) ## updating the state vector after the load action
                        ### update the charge levels of bikes in the station and the truck 
                        num=int(self.s_v[0][self.total_timeslots][i]*0.50)
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
                    else:
                        flag=True
                if action[i]==2:
                    if (truck >0 and truck + self.s_v[0][self.total_timeslots][i]<=30):
                        self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]+truck ## updating the station after the unload operation
                        truck = 0  ### updating the truck after the unload operation
                        #### updating the charge levels of the truck and the station after the unload operation 
                        num=0
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
                    elif(self.s_v[0][self.total_timeslots][i]+truck>30):
                        self.s_v[0][self.total_timeslots][i]=30    ## updating the state vector after an unload operation 
                        truck=truck-(30-self.s_v[0][self.total_timeslots][i]) ## updating the truck after the unload operation
                        num=30-self.s_v[0][self.total_timeslots][i]
                        u=1
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
                    else:
                        flag=True
        
                    
        if(self.total_timeslots%2==0):
            for i in range(len(action)-1,-1,-1):
                #print(truck_charge)
                if action[i]==0:### No action
                    continue
                
                
                
                
                if action[i]==1: ### action of loading bikes into the truck
                    if (int(self.s_v[0][self.total_timeslots][i]*0.50))>0 and (truck+int(s_v[0][self.total_timeslots][i]*0.50)<=20):
                        truck=truck+int(self.s_v[0][self.total_timeslots][i]*0.50)    ### updating the truck after the load action 
                        self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]-int(self.s_v[0][self.total_timeslots][i]*0.50) ## updating the state vector after the load ac
                        #### Update the charge of teh station and the truck after the load operation 
                        num=int(self.s_v[0][self.total_timeslots][i]*0.50)
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
                    else:
                        flag=True
        
                if action[i]==2:
                    if (truck >0 and truck + self.s_v[0][self.total_timeslots][i]<=30):
                        self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]+truck ## updating the station after the unload operation
                        truck = 0  ### updating the truck after the unload operation 
                    #### Updating the charge of the truck and the station after the unload operation 
                        num=0
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
           
                    elif(self.s_v[0][self.total_timeslots][i]+truck>30):
                        self.s_v[0][self.total_timeslots][i]=30    ## updating the state vector after an unload operation 
                        truck=truck-(30-self.s_v[0][self.total_timeslots][i])
                        num=30-self.s_v[0][self.total_timeslots][i]
                        u=1
                        truck_charge=self.charge_vec_update(truck_charge,action[i],num,i,u)
                    else:
                        flag=True
           
        
        return self.s_v,truck,truck_charge,flag    ### this function returns the the updated state vector and the the updated truck value 

             
    def charge_vec_update(self,truck_charge,sta_action,num,sta_num,u):
        if sta_action == 0:
            return truck_charge 
        if sta_action ==1:
            if(num<=self.charge_b[self.total_timeslots][sta_num][0]):### checking for high charge 
                self.charge_b[self.total_timeslots+1][sta_num][0]=self.charge_b[self.total_timeslots][sta_num][0]-num
                truck_charge[0]=truck_charge[0]+num
            
            
            if(num>self.charge_b[self.total_timeslots][sta_num][0]):
                self.charge_b[self.total_timeslots+1][sta_num][0]=0
                truck_charge[0]=truck_charge[0]+self.charge_b[self.total_timeslots][sta_num][0]
                change=num-self.charge_b[self.total_timeslots][sta_num][0]
            #print(change)
                if(change<=self.charge_b[self.total_timeslots][sta_num][1]):
                    self.charge_b[self.total_timeslots+1][sta_num][1]=self.charge_b[self.total_timeslots][sta_num][1]-change
                
                    truck_charge[1]=truck_charge[1]+change
                
                if(change>self.charge_b[self.total_timeslots][sta_num][1]):
                    self.charge_b[self.total_timeslots+1][sta_num][1]=0
                    truck_charge[1]=truck_charge[1]+self.charge_b[self.total_timeslots][sta_num][1]
                
                    change2=change-self.charge_b[self.total_timeslots][sta_num][1]
                    if(change2>0):
                        truck_charge[2]=truck_charge[2]+change2
                        if(change2> self.charge_b[self.total_timeslots][sta_num][2]):
                            self.charge_b[self.total_timeslots+1][sta_num][2]=change2-self.charge_b[self.total_timeslots][sta_num][2]
                        else:
                            self.charge_b[self.total_timeslots+1][sta_num][2]=self.charge_b[self.total_timeslots][sta_num][2]-change2
                    else:
                        self.charge_b[self.total_timeslots+1][sta_num][2]=self.charge_b[self.total_timeslots][sta_num][2]
        if sta_action==2:
        
        #### put a condition in the code to check if the charge vector of the station  is less than 30 after adding bikes after a unload 
            if u==0:                 
         
                self.charge_b[self.total_timeslots+1][sta_num][0]=self.charge_b[self.total_timeslots][sta_num][0]+truck_charge[0]
                self.charge_b[self.total_timeslots+1][sta_num][1]=self.charge_b[self.total_timeslots][sta_num][1]+truck_charge[1]
                self.charge_b[self.total_timeslots+1][sta_num][2]=self.charge_b[self.total_timeslots][sta_num][2]+truck_charge[2]
                truck_charge[0]=0
                truck_charge[1]=0
                truck_charge[2]=0
            if u==1:
    
                if(num<=truck_charge[0]):
                    truck_charge[0]=truck_charge[0]-num
                    self.charge_b[self.total_timeslots+1][sta_num][0]=self.charge_b[self.total_timeslots][sta_num][0]+num
                
                if(num>truck_charge[0]):
                
                    self.charge_b[self.total_timeslots+1][sta_num][0]=self.charge_b[self.total_timeslots][sta_num][0]+truck_charge[0]
                    diff=num-truck_charge[0]
                    truck_charge[0]=0
                    if(diff<truck_charge[1]):
                        truck_charge[1]=0
                        self.charge_b[self.total_timeslots+1][sta_num][1]=self.charge_b[self.total_timeslots][sta_num][1]+diff
                    if(diff>truck_charge[1]):
                    
                        self.charge_b[self.total_timeslots+1][sta_num][1]=self.charge_b[self.total_timeslots][sta_num][1]+truck_charge[1]
                        diff2=diff-truck_charge[1]
                        truck_charge[1]=0
                        if(diff2>0):
                            self.charge_b[self.total_timeslots+1][sta_num][2]=self.charge_b[self.total_timeslots][sta_num][2]+diff
                            truck_charge[2]=0 
        
        
        
            
        return truck_charge
                    
    
    def step(self,action,truck,truck_charge):
        ### there are 3 actions considered -- the actions taken by the agent which is the station are as follows-- (No action ,load_bikes, unload_bikes)
            # action 0 - No action 
            # action 1- load bikes into a truck 
            # action 2 - unload bikes from a truck 
            ## agent -- is a station 
            ### any action begins from  the time slot 1 onwards 
            self.action_history.append(action)## storing the actions taken in a particular time slot (array ds used to store action history)
            ## action is a single dimensional array as it gives actions for each of the n stations simultaneously 
            ## counting the number of time slots executed 
            #self.total_timeslots+=1### this gives the current time slot on which the action is taken 
            info = None
            flag=False
            
            state,truck_,charge_,flag=self.nextState(action,truck,truck_charge,flag)## calling the next state function to update the state vector according to the action given to the step function
            ### the length of the action, bike_vec and charge_vec for a given time slot is = number of stations 
            ## gives back the state vector which contains( number of bikes at each station, the total number of pickups and number of dropoffs) and the charge_b vector which contains the number of bikes at a particular stattion with certain charge level 
            if(self.total_timeslots<(self.timeslots-1)):
                self.total_timeslots+=1
            info =None
            ## calculating the reward for actions taken for a particular time slot 
            ##considering the number of bikes in a station, wheather the demand of the station is met and the charge level of the bikes present in the station 
            reward=0
            for i in range(len(action)):
                ### checking if each station is balanced in the map 
                if(state[1][self.total_timeslots][i]<=(state[0][self.total_timeslots][i]+state[2][self.total_timeslots][i])):
                    reward+=2## giving a positive reward for each balanced station after repositioning 
                else:
                    reward-=2
                ### considering the number of bikes having a charge level between high(3-4kW) and mid (1-2kW) should be equal to or greater than the pickups at that time slot 
                if(state[1][self.total_timeslots][i]<=(self.charge_b[self.total_timeslots][i][0]+self.charge_b[self.total_timeslots][i][1])):
                    reward+=1
                else:
                    reward-=1
                if(flag==True):  ##### taking into consideration the bounce off of truck jab station mai bikes na ho during a load  
                    reward-=1    ### and jab station full ho jaye tab during the unload 
                if(flag==False):
                    reward+=1
                    
            self.total_reward+=reward ## updating the total reward to be accomplished at the end of x episodes 
            
            ## defining the done variable to end an episode for a number of time slots 
            
            if(self.total_timeslots==(self.timeslots-1)):## the max timeslot for ending an episode 
                done=True
                self.total_timeslots=0
            else:
                done=False
            
            return state,reward,done,(truck_,charge_)


        
    
    def reset(self):
        initial_timeslot=random.randint(0,self.timeslots-1)## initializing the initial timeslot(done randomly for now)
        self.action_history=[0]*(len(self.action_history))## setting the action_history vector to zero 
        self.total_reward=0 ## setting the total reward after an episode to zero 
        self.s_v=self.initial
        self.charge_b=self.initial_charge
        return self.s_v,self.charge_b### returning the original state vector and charge levels as present in the dataset 
    
    
        
        
    
                                                       
                                                       
                     
                                                       
                                                       
                                                       
                                                       
                                                       
               
                                            
                                                       
            
                
                                      
                
        
        


# In[8]:


## calling the bike_rep_env with arandomized action pair 
### action size - 3 (0 -> No action,1 -> Load,2 -> unload)
action_size=2
### creating an object for the class bike_rep_env


env=Bike_rep_env(num_b,charge_b,pickup_b,dropoff_b,s_v)
a_v=[]

#init_state,init_charge=env.reset()## reset the env and the charge levels of the dataset
#print(init_state,init_charge)
action=[1,1,1,0,2]
truck=0
truck_charge=[0,0,0]
total=0
## executing the agent 


# In[9]:


k=1
for j in range(10):
    while(True):
            for i in range(5):
                action=random.randint(0,action_size)### generating random actions for a particular timeslot 
                a_v.append(action)
            state,reward,done,truck_prop=env.step(a_v,truck,truck_charge)
            total+=reward
            a_v=[]
            
    
            if done:
                break
    
    print("Episode",k,"Reward",total)
    total=0
    k+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




