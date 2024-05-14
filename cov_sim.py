import mesa

# Data visualization tools.
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import random

import gc

import csv

import Kingston_Info
import argparse

import json

def parse_arguments():

    parser = argparse.ArgumentParser(description="Configure simulation parameters")

    #Residence populations. The number of students in residence at each post-secondary institution (default ratio roughly corresponds to real-life numbers)
    parser.add_argument("--queens_residence_pop", type=int, default=100, help="Enter the Queen's residence population")
    parser.add_argument("--slc_residence_pop", type=int, default=50, help="Enter the SLC residence population")
    parser.add_argument("--rmc_residence_pop", type=int, default=40, help="Enter the RMC residence population")

    #Total post-secondary school populations
    parser.add_argument("--queens_pop", type=int, default=250, help="Enter the total Queen's population")
    parser.add_argument("--slc_pop", type=int, default=60, help="Enter the total SLC population")
    parser.add_argument("--rmc_pop", type=int, default=40, help="Enter the total RMC population")

    #Kingston population. Roughly the total number of agents that will be in the simulation (can be up to 4 more due to home population generation)
    parser.add_argument("--kingston_pop", type=int, default=1000, help="Enter the total population")

    #Number of residences at each post-secondary institution (except for SLC, which only has 1 residence)
    parser.add_argument("--queens_residences", type=int, default=10, help="Enter the number of Queen's residences - max of 30")
    parser.add_argument("--rmc_residences", type=int, default=3, help="Enter the number of RMC residences - max of 7")

    #Penalties associated with the simulation and planner actions
    parser.add_argument("--mask_penalty_all", type=float, default=-10, help="Enter the mask penalty factor for all agents")
    parser.add_argument("--vaccine_penalty_all", type=float, default=-10, help="Enter the vaccine penalty factor for all agents")
    parser.add_argument("--mask_penalty_students", type=float, default=-5, help="Enter the mask penalty factor for students")
    parser.add_argument("--vaccine_penalty_students", type=float, default=-5, help="Enter the vaccine penalty factor for students")
    parser.add_argument("--non_icu_penalty", type=float, default=-8000, help="Enter the non-ICU penalty factor")
    parser.add_argument("--icu_penalty", type=float, default=-8000, help="Enter the ICU penalty factor")

    #The factors that will be multiplied with transmission chance
    parser.add_argument("--mask_factor", type=float, default=0.8, help="Enter the factor that wearing a mask multiplies transmission rate by")
    parser.add_argument("--vaccine_factor", type=float, default=0.4, help="Enter the factor that being vaccinated multiplies transmission rate by")

    #The chance an agent wears a mask
    parser.add_argument("--mask_chance", type=float, default=0.7, help="Enter the chance that an agent wears a mask")

    #The total number of non-ICU and ICU beds
    parser.add_argument("--non_icu_beds", type=int, default=2, help="Enter the total number of non-ICU beds")
    parser.add_argument("--icu_beds", type=int, default=1, help="Enter the total number of ICU beds")

    #The number of time steps for the simulation
    parser.add_argument("--horizon", type=int, default=100, help="Enter the desired number of time steps (horizon)")
    
    #Defines the way the simulation is run
    parser.add_argument("--mode", type=str, default="Init", help="Enter the desired mode (Init if you are creating new problem files, Test if you are drawing from existing problem files)")
    parser.add_argument("--iters", type=int, default=20, help="Enter the number of iterations you wish to run")
    parser.add_argument("--trials", type=int, default=20, help="Enter the number of trials per iteration you wish to run")

    return parser.parse_args()

#Agent class
#Stores what class they are in, and their mask/vaccination status
#Also tracks how long they have been in each class and how many agents they have spread disease to
class CovidAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, agent_info_all, model):
        #A child of the model class
        super().__init__(unique_id, model)

        self.age_bracket = agent_info_all[0][2]

        self.store_1 = agent_info_all[0][5]
        self.store_2 = agent_info_all[0][6]

        self.isolating = False

        #Assigns them as infectious from the start depending on some probability (10% here)
        prob_infectious = np.random.uniform(0,1)
        if (prob_infectious <= 0.1):
            self.susceptible = False
            self.exposed = False
            self.infectious = True
            self.recovered = False
            self.total_time_in_class = np.random.normal(16, 4)
        
        else:
            self.susceptible = True
            self.exposed = False
            self.infectious = False
            self.recovered = False
            self.total_time_in_class = 999

        #Same deal with masking and vaccinating
        prob_masked = np.random.uniform(1,2)
        if (prob_masked <= 0.7):
            self.masked_factor = 0.8
        else:
            self.masked_factor = 1

        prob_vaccinated = np.random.uniform(1,2)

        #Set to 1 by default
        self.vaccinated_factor = 1

        #Vaccination rates based on age bracket
        if self.age_bracket == 0:
            if (prob_vaccinated <= 0.251):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 1:
            if (prob_vaccinated <= 0.771):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 2:
            if (prob_vaccinated <= 0.819):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 3:
            if (prob_vaccinated <= 0.851):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 4:
            if (prob_vaccinated <= 0.883):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 5:
            if (prob_vaccinated <= 0.885):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 6:
            if (prob_vaccinated <= 0.940):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 7:
            if (prob_vaccinated <= 0.982):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 8:
            if (prob_vaccinated <= 0.990):
                self.vaccinated_factor = 0.4

        self.time_in_class = 0

        self.spread_to = 0

    #When the scheduler takes a step, agents do nothing. The agent behaviour is performed in the "location" agents
    def step(self):
        
        pass


#Location class is where the bulk of the mechanics lie
#Each location is stepped through, and on each step, every agent at a given location has their status updated based on other agents at that location
class CovidLocation(mesa.Agent):

    def __init__(self, unique_id, loc_info, agents_belonging, model):
        #Child of the model class
        super().__init__(unique_id, model)
        
        #Extracts some key info about the location
        self.loc_type = loc_info[1]
        self.loc_id = unique_id
        self.agents_at_loc = agents_belonging

        #Sets the initial values for number of agents of a given class at a location
        self.susceptible_count = 0
        self.exposed_count = 0
        self.infectious_count = 0
        self.recovered_count = 0
        for item in self.agents_at_loc:
            if item.susceptible:
                self.susceptible_count += 1
            elif item.exposed:
                self.exposed_count += 1
            elif item.infectious:
                self.infectious_count += 1
            else:
                self.recovered_count += 1

    #Updates the class of a single agent
    def update_class(self, current_agent):

        #This will be set to "True" if an agent is successfully infected by another agent
        newly_exposed = False

        #print("trying for location " + str(self.unique_id) + " with " + str(self.susceptible_count) + " susceptible agents" + ", agent " + str(current_agent.unique_id) + " with status " + 
        #      str((current_agent.susceptible, current_agent.exposed, current_agent.infectious, current_agent.recovered)))

        #Captures the mechanics of one agent being exposed to the disease by another
        if current_agent.susceptible:
            #print("agent " + str(current_agent.unique_id) + " being tested at location " + str(self.unique_id)
            #      + ". " + str(self.susceptible_count) + " susceptible and " + str(self.infectious_count) + " infectious")
            
            #For all agents at the same location as the agent in question, test to see if they infect the agent in question
            for agent_at_same_loc in self.agents_at_loc:


                if (self.model.day_of_week == 12 and agent_at_same_loc.store_1 == self.loc_id) or (self.model.day_of_week == 14 and agent_at_same_loc.store_2 == self.loc_id) or (not (self.model.day_of_week in (12, 14))):

                    #If an agent is infectious and not equalt to the current agent (although since the agent cannot be both susceptible and infectious, this would never happen)
                    if ((agent_at_same_loc.infectious) and (agent_at_same_loc != current_agent) and (not agent_at_same_loc.isolating)):

                        #The odds of an agent contracting the disease from another agent is (R0/(time in infectious class)) / (# of susceptibles at the location)
                        #This makes it so that each agent will infect approximately 3.32 agents, lining up with the definition of R0
                        #We multiply by the mask and vaccination factors if applicable for the agents in question
                        #Note that the infectious agent does not have their infectivity reduced by being vaccinated - if they are infected, they are spreading it the same (this is an assumption)
                        odds = (agent_at_same_loc.masked_factor * current_agent.masked_factor) * (current_agent.vaccinated_factor) * (3.32 / 16) / (self.susceptible_count)

                        #If the infection is successful, then the agent in question becomes infected
                        against = np.random.uniform(0,1)
                        if (against < odds):
                            #Add 1 to the number of agents infected
                            agent_at_same_loc.spread_to += 1
                            newly_exposed = True
            
            #Update the agent in question's status and reset their class timer
            if newly_exposed:
                #print("Success!")
                current_agent.susceptible = False
                current_agent.exposed = True
                current_agent.time_in_class = 0

                #Set the time the agent will be in the exposed class
                current_agent.total_time_in_class = np.random.normal(9, 2)

        #If the agent in question is instead exposed and at the end of their exposd period, update their status and reset their class timer
        elif current_agent.exposed and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.exposed = False
            current_agent.infectious = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the infectious class
            current_agent.total_time_in_class = np.random.normal(16, 4)

            #Check to see if agent isolates
            isolating_prob = np.random.uniform(0, 1)
            if isolating_prob <= 0.3:
                current_agent.isolating = True

        #Same for infectious
        #Update the model.spread_count_list list with the total number of agents they successfully infected
        elif current_agent.infectious and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.infectious = False
            current_agent.recovered = True
            #print("Agent " + str(current_agent.unique_id) + " spread to " + str(current_agent.spread_to) + " agents")
            current_agent.model.spread_count_list.append(current_agent.spread_to)
            current_agent.spread_to = 0
            current_agent.time_in_class = 0

            #Set the time the agent will be in the recovered class
            current_agent.total_time_in_class = 14

            current_agent.isolating = False
        
        #Same for recovered
        elif current_agent.recovered and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.recovered = False
            current_agent.susceptible = True
            current_agent.time_in_class = 0

            #Set to -1, since an agent is susceptible until infected
            current_agent.total_time_in_class = 999

        #print("after: agent " + str(current_agent.unique_id) + " with status " + 
        #      str((current_agent.susceptible, current_agent.exposed, current_agent.infectious, current_agent.recovered)))

    #Stepping through the locations
    def step(self):

        #First, do a count of the number of agents in each class present at a location
        #This must be done first, because agents are technically at all locations at once, and so if an agent is at locations x and y, but location x updates the status of the agent, its status might not be reflected in the current count for location y
        agent_list_here = []

        susceptible_per = 0
        exposed_per = 0
        infectious_per = 0
        recovered_per = 0

        for item in self.agents_at_loc:

            if (item.susceptible):
                susceptible_per += 1
            elif (item.exposed):
                exposed_per += 1
            elif (item.infectious):
                infectious_per += 1
            else:
                recovered_per += 1

        self.susceptible_count = susceptible_per
        self.exposed_count = exposed_per
        self.infectious_count = infectious_per
        self.recovered_count = recovered_per

        #We track the number of agents in each class once more after updating their classes for the sake of accurate reporting on each step
        #If we were to NOT care about the progress of the disease at each time step, we could theoretically just update it once at the start of the step() function
        susceptible_per = 0
        exposed_per = 0
        infectious_per = 0
        recovered_per = 0

        #For each agent at the location, update their status if it is the location's turn to step (dependent on the time of day)
        for item in self.agents_at_loc:

            if ((self.model.day_of_week % 2 == 1 and self.loc_type == "home") or (self.model.day_of_week in (2,4,6,8,10) and self.loc_type == "job")
            or (self.model.day_of_week == 12 and self.loc_type == "store" and item.store_1 == self.loc_id) or (self.model.day_of_week == 14 and self.loc_type == "store" and item.store_2 == self.loc_id)):

                #Update the agent's class and the time that they have been in the class
                self.update_class(item)
                item.time_in_class += 1

            #Append the agent and their class info
            agent_list_here.append((item.unique_id, item.susceptible, item.exposed, item.infectious, item.recovered))

            if (item.susceptible):
                susceptible_per += 1
            elif (item.exposed):
                exposed_per += 1
            elif (item.infectious):
                infectious_per += 1
            else:
                recovered_per += 1
        
        self.susceptible_count = susceptible_per
        self.exposed_count = exposed_per
        self.infectious_count = infectious_per
        self.recovered_count = recovered_per

        #print("location " + str(self.unique_id))
        #print(agent_list_here)
        #print(self.susceptible_count)
        #print(self.exposed_count)
        #print(self.infectious_count)
        #print(self.recovered_count)


        #print('location ' + str(self.loc_id) + ' here, agents I own are ' + str(agent_list_here))


#Model class, responsible for creating and stepping through all agents
class CovidModel(mesa.Model):


    def __init__(self, agents_info, location_info):
        #Extracts some basic info
        self.num_agents = len(agents_info)
        self.num_locations = len(location_info)

        #Creates the schedule - agents are stepped through randomly, although it does not matter in our case
        self.schedule = mesa.time.RandomActivation(self)

        self.running = True
        
        #Initialize to 1, indicating a Monday where all agents are at home
        self.day_of_week = 1

        #Total class counts are set to 0
        self.total_sus_count = 0
        self.total_exp_count = 0
        self.total_inf_count = 0
        self.total_rec_count = 0

        self.spread_count_list = []

        #Warm up the agent class?
        a = CovidAgent(agents_info[0][0][0], agents_info[0], self)

        agent_list = []

        # Create agents
        for i in range(0, self.num_agents):
            a = CovidAgent(agents_info[i][0][0], agents_info[i], self)
            agent_list.append((agents_info[i][0][0], a))
            self.schedule.add(a)
        
        #Create location agents with agent agents as objects of each location agent if the agent agent is at the location agent
        for i in range (0, self.num_locations):
            agents_at_loc = []

            for agent_temp in agent_list:
                if agent_temp[0] in location_info[i][2]:
                    agents_at_loc.append(agent_temp[1])

            l = CovidLocation(location_info[i][0], location_info[i], agents_at_loc, self)
            self.schedule.add(l)

        #self.datacollector = mesa.DataCollector(
        #    agent_reporters={"Location": "current_location"}
        #)

    #Step through the model (not stepped through by the scheduler)
    def step(self):
        #self.datacollector.collect(self)

        #Steps through location and agent agents
        self.schedule.step()

        #If we have reached Sunday night, reset to Monday morning
        if (self.day_of_week <= 13):
            self.day_of_week += 1
        else:
                self.day_of_week = 1

        #Count the number of agents in each class on each time step
        sus_count = 0
        exp_count = 0
        inf_count = 0
        rec_count = 0

        for obj in gc.get_objects():
            if isinstance(obj, CovidAgent):
                if (obj.susceptible):
                    sus_count += 1
                elif (obj.exposed):
                    exp_count += 1
                elif (obj.infectious):
                    inf_count += 1
                elif (obj.recovered):
                    rec_count += 1

        self.total_sus_count = sus_count
        self.total_exp_count = exp_count
        self.total_inf_count = inf_count
        self.total_rec_count = rec_count

        print("Susceptible: " + str(self.total_sus_count) + ", Exposed: " + str(self.total_exp_count) + 
              ", Infectious: " + str(self.total_inf_count) + ", Recovered: " + str(self.total_rec_count))

args_sim = parse_arguments()

susceptible_counts_total = []
exposed_counts_total = []
infectious_counts_total = []
recovered_counts_total = []
time_step_total = []

info = Kingston_Info.main_kingston_geo()
location_details = info[0]
agent_details = info[1]
organized_locs = info[2]

#print(agent_details)

for k in range (500):

    print("TRIAL: " + str(k))

    model = CovidModel(agent_details, location_details)
    model.reset_randomizer()

    susceptible_counts_trial = []
    exposed_counts_trial = []
    infectious_counts_trial = []
    recovered_counts_trial = []
    time_step_trial = []

    for i in range(500):

        model.step()

        susceptible_counts_trial.append(model.total_sus_count)
        exposed_counts_trial.append(model.total_exp_count)
        infectious_counts_trial.append(model.total_inf_count)
        recovered_counts_trial.append(model.total_rec_count)
        time_step_trial.append(i)

    susceptible_counts_total.append(susceptible_counts_trial)
    exposed_counts_total.append(exposed_counts_trial)
    infectious_counts_total.append(infectious_counts_trial)
    recovered_counts_total.append(recovered_counts_trial)
    time_step_total.append(time_step_trial)


data_dict = {
    "Susceptible": str(susceptible_counts_total),
    "Exposed": str(exposed_counts_total),
    "Infectious": str(infectious_counts_total),
    "Recovered": str(recovered_counts_total),
    "Time_step": str(time_step_total)
}


#json_object = json.dumps(data_dict, indent=4)    

#with open("real_geo_500_trials_SEIR.json", "w") as outfile:
#    outfile.write(json_object)

'''
susceptible_counts_total = []
exposed_counts_total = []
infectious_counts_total = []
recovered_counts_total = []
time_step_total = []

for k in range (500):

    print("TRIAL: " + str(k))

    new_info = Kingston_Info.shuffle_locations_uniform(organized_locs)

    location_details = info[0]
    agent_details = info[1]


    model = CovidModel(agent_details, location_details)
    model.reset_randomizer()

    susceptible_counts_trial = []
    exposed_counts_trial = []
    infectious_counts_trial = []
    recovered_counts_trial = []
    time_step_trial = []

    for i in range(500):

        model.step()

        susceptible_counts_trial.append(model.total_sus_count)
        exposed_counts_trial.append(model.total_exp_count)
        infectious_counts_trial.append(model.total_inf_count)
        recovered_counts_trial.append(model.total_rec_count)
        time_step_trial.append(i)

    susceptible_counts_total.append(susceptible_counts_trial)
    exposed_counts_total.append(exposed_counts_trial)
    infectious_counts_total.append(infectious_counts_trial)
    recovered_counts_total.append(recovered_counts_trial)
    time_step_total.append(time_step_trial)


data_dict = {
    "Susceptible": str(susceptible_counts_total),
    "Exposed": str(exposed_counts_total),
    "Infectious": str(infectious_counts_total),
    "Recovered": str(recovered_counts_total),
    "Time_step": str(time_step_total)
}
'''

#json_object = json.dumps(data_dict, indent=4)    

#with open("random_geo_500_trials_SEIR.json", "w") as outfile:
#    outfile.write(json_object)


#plt.plot(time_step_trial, infectious_counts_trial, color='red', linewidth=0.5)
#plt.plot(time_step_trial, susceptible_counts_trial, color='green', linewidth=0.5)
#plt.title('Susceptible/Infectious Agents Over Time')
#plt.xlabel("Time Steps")
#plt.ylabel("Number of Agents")
#plt.legend()

#save_name = "10000_agents.png"
#plt.savefig(save_name)
#plt.clf()
