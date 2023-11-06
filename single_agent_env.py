# -*- coding: utf-8 -*-
"""single_agent_env.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZLw2Z_CjMfNn-3UUosgMQukj6x4XfkUJ
"""

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

# importing necessary libraries
import random
import gym
import pandas as pd
import numpy as np

#####   These declarations will be replaced by the dataset values #####

### info on the shape of the variables declared below
## num_b(x,y) -> x is the number of timeslots , y is the number of stations
## charge_b(x,y,z) -> x is the number of timeslots, y is the number of stations and z is the number of vehicles at a charge level
## pickup_b and dropoff_b -> similar to num_b

num_b=np.array([10,15,12])#### initializing the array which contains the number of bikes at each of the 5 stations at any given point in time_slot(excluding the pickups and dropoffs)
charge_b=np.array([[5,2,3],[15,0,0],[12,0,0]])
pickup_b=np.array([5,20,1])
dropoff_b=np.array([10,23,32])
#### these declarations are for the 3 time slots of 30 min(say)

### verifying the shape of each array
print("num_b",num_b.shape)### the x-axis denotes the number of time slots and the y-axis denotes the number of stations
print("charge_b",charge_b.shape)### the x-axis denotes the number of time slots , the y - axis denotes the number of stations and the z - axis denotes teh number of bikes at the three different charge levels
print("load_b",pickup_b.shape)### the x axis denotes the number of time slots and the y - axis denotes the number of stations
print("unload_b",dropoff_b.shape)### the x- axis denotes the number of time slots and the y - axis denotes the number of stations

#### note:: This declaration will be replaced with the dataset values which achal provides

s_v=np.array([num_b,pickup_b,dropoff_b])### defining the state vector
#print(s_v.shape)
#### charge vector contains a different shape will be take care off individually
##print(s_v)
#arr=[[11,1,3],[12,3,1],[13,23,1]]
print(s_v)## station number and then charge level is denoted by j
print(s_v[0][1])

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


           # print(truck_charge)
                #print(truck)


                if action<0: ### action of loading bikes into the truck
                    if (int(self.s_v[0][self.total_timeslots]))>=abs(action) and (truck+ action <=20):
                        truck=truck+abs(action)    ### updating the truck after the load action
                        self.s_v[0][self.total_timeslots+1]=self.s_v[0][self.total_timeslots]+action ## updating the state vector after the load action
                        ### update the charge levels of bikes in the station and the truck
                        num=abs(action)
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action,num,0,u)
                    else:
                        flag=True
                if action >= 0:
                    if ((truck - action) >=0 and self.s_v[0][self.total_timeslots] + action<=30):
                        self.s_v[0][self.total_timeslots+1]=self.s_v[0][self.total_timeslots]+action## updating the station after the unload operation
                        truck = truck - action ### updating the truck after the unload operation
                        #### updating the charge levels of the truck and the station after the unload operation
                        num=truck
                        u=0
                        truck_charge=self.charge_vec_update(truck_charge,action,num,0,u)
                    elif(self.s_v[0][self.total_timeslots]+action>30):
                        self.s_v[0][self.total_timeslots+1]=30    ## updating the state vector after an unload operation
                        truck=truck-(30-self.s_v[0][self.total_timeslots]) ## updating the truck after the unload operation
                        num=30-self.s_v[0][self.total_timeslots]
                        u=1
                        truck_charge=self.charge_vec_update(truck_charge,action,num,0,u)
                    else:
                        flag=True




                return self.s_v,truck,truck_charge,flag    ### this function returns the the updated state vector and the the updated truck value


    def charge_vec_update(self,truck_charge,sta_action,num,sta_num,u):

        if sta_action <0:
            if(num<=self.charge_b[self.total_timeslots][0]):### checking for high charge
                self.charge_b[self.total_timeslots+1][0]=self.charge_b[self.total_timeslots][0]-num
                truck_charge[0]=truck_charge[0]+num


            if(num>self.charge_b[self.total_timeslots][0]):
                self.charge_b[self.total_timeslots+1][0]=0
                truck_charge[0]=truck_charge[0]+self.charge_b[self.total_timeslots][0]
                change=num-self.charge_b[self.total_timeslots][0]
            #print(change)
                if(change<=self.charge_b[self.total_timeslots][1]):
                    self.charge_b[self.total_timeslots+1][1]=self.charge_b[self.total_timeslots][1]-change

                    truck_charge[1]=truck_charge[1]+change

                if(change>self.charge_b[self.total_timeslots][1]):
                    self.charge_b[self.total_timeslots+1][1]=0
                    truck_charge[1]=truck_charge[1]+self.charge_b[self.total_timeslots][1]

                    change2=change-self.charge_b[self.total_timeslots][1]
                    if(change2>0):
                        truck_charge[2]=truck_charge[2]+change2
                        if(change2> self.charge_b[self.total_timeslots][2]):
                            self.charge_b[self.total_timeslots+1][2]=change2-self.charge_b[self.total_timeslots][2]
                        else:
                            self.charge_b[self.total_timeslots+1][2]=self.charge_b[self.total_timeslots][2]-change2
                    else:
                        self.charge_b[self.total_timeslots+1][2]=self.charge_b[self.total_timeslots][2]
        if sta_action>=0:

        #### put a condition in the code to check if the charge vector of the station  is less than 30 after adding bikes after a unload


                if(num<=truck_charge[0]):
                    truck_charge[0]=truck_charge[0]-num
                    self.charge_b[self.total_timeslots+1][0]=self.charge_b[self.total_timeslots][0]+num

                if(num>truck_charge[0]):

                    self.charge_b[self.total_timeslots+1][0]=self.charge_b[self.total_timeslots][0]+truck_charge[0]
                    diff=num-truck_charge[0]
                    truck_charge[0]=0
                    if(diff<truck_charge[1]):
                        truck_charge[1]=0
                        self.charge_b[self.total_timeslots+1][1]=self.charge_b[self.total_timeslots][1]+diff
                    if(diff>truck_charge[1]):

                        self.charge_b[self.total_timeslots+1][1]=self.charge_b[self.total_timeslots][1]+truck_charge[1]
                        diff2=diff-truck_charge[1]
                        truck_charge[1]=0
                        if(diff2>0):
                            self.charge_b[self.total_timeslots+1][2]=self.charge_b[self.total_timeslots][2]+diff
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
            rew_arr=[]

                ### checking if each station is balanced in the map
            if(state[1][self.total_timeslots]<=(state[0][self.total_timeslots]+state[2][self.total_timeslots])):
                    reward+=2## giving a positive reward for each balanced station after repositioning
                    rew_arr.append(reward)
            else:
                    reward-=2
                    rew_arr.append(reward)
                ### considering the number of bikes having a charge level between high(3-4kW) and mid (1-2kW) should be equal to or greater than the pickups at that time slot
            if(state[1][self.total_timeslots]<=(self.charge_b[self.total_timeslots][0]+self.charge_b[self.total_timeslots][1])):
                    reward+=1
                    rew_arr.append(reward)

            else:
                    reward-=1
                    rew_arr.append(reward)

                # if(flag==True):  ##### taking into consideration the bounce off of truck jab station mai bikes na ho during a load
                #     reward-=1    ### and jab station full ho jaye tab during the unload
                # if(flag==False):
                #     reward+=1

            self.total_reward+=reward ## updating the total reward to be accomplished at the end of x episodes

            ## defining the done variable to end an episode for a number of time slots
            s1=state[0][self.total_timeslots]
            s2=state[1][self.total_timeslots]
            s3 = state[2][self.total_timeslots]
            state=[s1,s2,s3]
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
        s1=self.initial[0][0]
        s2=self.initial[1][0]
        s3 = self.initial[2][0]
        state=[s1,s2,s3]

        return state,self.charge_b[0]### returning the original state vector and charge levels as present in the dataset


    def seed(self,seed_value):
      random.seed(seed_value)

## calling the bike_rep_env with arandomized action pair
### action size - 3 (0 -> No action,1 -> Load,2 -> unload)

### creating an object for the class bike_rep_env


env=Bike_rep_env(num_b,charge_b,pickup_b,dropoff_b,s_v)
a_v=[]
state , charge= env.reset()
#init_state,init_charge=env.reset()## reset the env and the charge levels of the dataset
#print(init_state,init_charge)
action=[1,1,1,0,2]
truck=0
truck_charge=[0,0,0]
total=0
## executing the agent
print(state)
print(charge)

state,reward,done,truck_prop=env.step(-8,truck,truck_charge)
truck_prop

k=1
for j in range(10):
    while(True):


            action=random.randint(-20,30)### generating random actions for a particular timeslot
            state,reward,done,truck_prop=env.step(action,truck,truck_charge)
            #print(state)
            total+=reward



            if done:
                break

    print("Episode",k,"Reward",total)
    total=0
    k+=1
