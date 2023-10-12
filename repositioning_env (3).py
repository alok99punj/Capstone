#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing necessary libraries 
import random 
import gym 
import pandas as pd 
import numpy as np 


# In[5]:


#####   These declarations will be replaced by the dataset values #####

### info on the shape of the variables declared below 
## num_b(x,y) -> x is the number of timeslots , y is the number of stations
## charge_b(x,y,z) -> x is the number of timeslots, y is the number of stations and z is the number of vehicles at a charge level
## pickup_b and dropoff_b -> similar to num_b


# In[6]:


num_b=np.array([[10,4,30,15,20],[15,23,4,22,1],[12,2,3,12,3]])#### initializing the array which contains the number of bikes at each of the 5 stations at any given point in time_slot
charge_b=np.array([[[5,2,3],[4,0,0],[10,10,10],[10,5,0],[20,0,0]],[[15,0,0],[20,3,0,],[4,0,0],[20,2,0],[1,0,0]],[[12,0,0],[2,0,0],[2,1,0],[12,0,0],[2,1,0]]])
pickup_b=np.array([[5 ,3 ,1 ,24,1],[20,1,3,23,12],[1,23,1,32,1]])
dropoff_b=np.array([[10,23,12,34,2],[23,3,1,3,1],[32,3,2,1,3]])
#### these declarations are for the 3 time slots of 30 min(say)


# In[7]:


### verifying the shape of each array 
print("num_b",num_b.shape)### the x-axis denotes the number of time slots and the y-axis denotes the number of stations 
print("charge_b",charge_b.shape)### the x-axis denotes the number of time slots , the y - axis denotes the number of stations and the z - axis denotes teh number of bikes at the three different charge levels 
print("load_b",pickup_b.shape)### the x axis denotes the number of time slots and the y - axis denotes the number of stations 
print("unload_b",dropoff_b.shape)### the x- axis denotes the number of time slots and the y - axis denotes the number of stations 

#### note:: This declaration will be replaced with the dataset values which achal provides 


# In[8]:


s_v=np.array([num_b,pickup_b,dropoff_b])### defining the state vector 
#print(s_v.shape)
#### charge vector contains a different shape will be take care off individually 
##print(s_v)
#arr=[[11,1,3],[12,3,1],[13,23,1]]
print(s_v)## station number and then charge level is denoted by j


# In[9]:


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
        
        
        
    def getState(self,action,bike_vec,charge_vec):
        for i in range(len(action)):
            if action[i]==0:
                continue
            if action[i]==1:
                if(self.s_v[0][self.total_timeslots][i]-bike_vec[i]>0):
                    self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]-bike_vec[i]
                for j in range(3):
                    if(self.charge_b[self.total_timeslots][i][j]-charge_vec[i][j]>0):
                        self.charge_b[self.total_timeslots+1][i][j]=self.charge_b[self.total_timeslots][i][j]-charge_vec[i][j]
            if action[i]==2:
                if(self.s_v[0][self.total_timeslots][i]+bike_vec[i]<=30):
                    self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]+bike_vec[i]
                if(self.s_v[0][self.total_timeslots][i]+bike_vec[i]>30):
                    self.s_v[0][self.total_timeslots+1][i]=self.s_v[0][self.total_timeslots][i]+(30-s_v[0][self.total_timeslots][i])
                for j in range(3):
                    ## check this part so that the overall count doesnot cross 30 
                    #if(charge_vec[i][j]+charge_b[total_timeslots][i][j]<=30):
                    if(sum(charge_vec[i])+sum(self.charge_b[self.total_timeslots][i]<=30)):
                        self.charge_b[self.total_timeslots+1][i][j]=charge_vec[i][j]+self.charge_b[self.total_timeslots][i][j]
        return self.s_v,self.charge_b
#a,b= getState([1,2,0,1,1],[12,23,1,12,31],[[10,2,1],[12,2,1],[3,1,2],[12,3,1],[12,20,1]])
## for this input the unit test output was [ 3 30  3 10  3] for the t+1 state 

    
    
    
    
    def step(self,action,bike_vec,charge_vec):
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
            state,charge=self.getState(action,bike_vec,charge_vec)## calling the next state function to update the state vector according to the action given to the step function
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
                if(state[1][self.total_timeslots][i]<=(charge[self.total_timeslots][i][0]+charge[self.total_timeslots][i][1])):
                    reward+=1
                else:
                    reward-=1
            self.total_reward+=reward ## updating the total reward to be accomplished at the end of x episodes 
            
            ## defining the done variable to end an episode for a number of time slots 
            
            if(self.total_timeslots==(self.timeslots-1)):## the max timeslot for ending an episode 
                done=True
                self.total_timeslots=0
            else:
                done=False
            
            return state,reward,done,info
    
    
    def reset(self):
        initial_timeslot=random.randint(0,self.timeslots-1)## initializing the initial timeslot(done randomly for now)
        self.action_history=[0]*(len(self.action_history))## setting the action_history vector to zero 
        self.total_reward=0 ## setting the total reward after an episode to zero 
        self.s_v=self.initial
        self.charge_b=self.initial_charge
        return self.s_v,self.charge_b### returning the original state vector and charge levels as present in the dataset 
    
    
        
        
        
        
        
        
                
                
            
                    
                
            
            
            
        


# In[10]:


## calling the bike_rep_env with arandomized action pair 
### action size - 3 (0 -> No action,1 -> Load,2 -> unload)
action_size=2
### creating an object for the class bike_rep_env


env=Bike_rep_env(num_b,charge_b,pickup_b,dropoff_b,s_v)


init_state,init_charge=env.reset()## reset the env and the charge levels of the dataset
#print(init_state,init_charge)


## executing the agent 


charge=[0]*15
charge=np.array(charge)
charge=charge.reshape(5,3)
total=0
a_v=[]
bike_vec=[]
total=0

  
            

    



# In[12]:


#### the reward for 10 episodes are calculated with randomized action pair which will be replaced by the algo 


# In[13]:


k=1
for j in range(10):
    while(True):
            for i in range(5):
                action=random.randint(0,action_size)### generating random actions for a particular timeslot 
                a_v.append(action) ### this is the action vector 
                for j in range(3):
                    charge[i][j]=random.randint(0,10)
                bike_vec.append(sum(charge[i]))
            state,reward,done,info=env.step(a_v,bike_vec,charge)
            total+=reward
            a_v=[]
            bike_vec=[]
    
            if done:
                break
    
    print("Episode",k,"Reward",total)
    total=0
    k+=1


# In[ ]:




