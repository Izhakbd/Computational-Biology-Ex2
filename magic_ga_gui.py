import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import random

# GA Core
class MagicSquareGA:
    def __init__(self, n: int, fitness_func, inheritance_mode=None, mutation_rate=0.2, use_elitism = False, elitism_rate = 0.2):
        
        self.n = n # size of the matrix
        self.M = n * (n ** 2 + 1) // 2  # the sum of each row, column, and diagonal for a magic square
        self.pop_size = 100 # population size
        self.mutation_rate = mutation_rate # mutation rate
        self.num_of_mutations = 1 # number of mutations to apply to each individual
        self.fitness_func = fitness_func # fitness function
        self.population = [] # current population
        self.best = None # best individual
        self.best_fitness = float("inf")  # best fitness
        self.generation = 0 # current generation
        self.fitness_call_count = 0 # count how many times the fitness function was called
        self.best_fitness_history = [] # best fitness history for plotting over generations
        self.worst_fitness_history = [] # worst fitness history for plotting over generations
        self.avg_fitness_history = [] # average fitness history for plotting over generations
        self.no_improvement_counter = 0
        self.last_best_fitness = None
        self.inheritance_mode = inheritance_mode  # inheritance mode: None, Darwinian, or Lamarckian
        self.use_elitism = use_elitism
        self.elitism_rate = elitism_rate  # detrermine how many of the best individuals to carry over to the next generation
        

    # Generate a random individual to initialize the population
    # An individual is a flat list of numbers from 1 to n^2
    def random_individual(self):
        genes = list(range(1, self.n * self.n + 1))
        random.shuffle(genes)
        return genes
    
    # Initialize the algorithm with a random population, this method run only once at the algorithm start
    def initialize(self):
      # Initialize the population with random individuals
        self.population = [self.random_individual() for _ in range(self.pop_size)]
        # Calculate initial fitness for the population
        current_fitnesses = [self.fitness_func(ind) for ind in self.population]
        worst_fitness = max(current_fitnesses)
        avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
        best_fitness = min(current_fitnesses)

        
        self.best_fitness = best_fitness
        self.worst_fitness = worst_fitness
        self.avg_fitness = avg_fitness
        self.generation = 0
        self.fitness_call_count = 0 # Reset fitness call count
        self.best_fitness_history = [self.best_fitness] # Add initial best fitness to history
        self.worst_fitness_history = [self.worst_fitness] # Add initial worst fitness to history
        self.avg_fitness_history = [self.avg_fitness] # Add initial average fitness to history
        self.no_improvement_counter = 0
        self.last_best_fitness = self.best_fitness

    # Fitness function for regular magic square candidates (N=3 or 5)
    def regular_magic_square_fitness(self, individual):
        """Calculates the fitness score of a regular magic square candidate (N=3 or 5)."""
        self.fitness_call_count += 1 # Increment the fitness call count each time the fitness function is called
        n = self.n # size of the matrix
        magic_sum = self.M # the sum of each row, column, and diagonal for a magic square
        matrix = [individual[i * n:(i + 1) * n] for i in range(n)] # convert flat list to array of arrays which elements are rows of the matrix.
        fitness = 0

        # Add up the absolute differences from the magic sum for rows.
        for row in matrix:
            row_sum = sum(row)
            fitness += abs(row_sum - magic_sum)

        # Add up the absolute differences from the magic sum for columns.
        for col in range(n):
            col_sum = sum(matrix[row][col] for row in range(n))
            fitness += abs(col_sum - magic_sum)

        # Add diagonal sums to fitness
        diag1 = sum(matrix[i][i] for i in range(n))
        diag2 = sum(matrix[i][n - 1 - i] for i in range(n))
        fitness += abs(diag1 - magic_sum)
        fitness += abs(diag2 - magic_sum)

        return fitness


    # Most perfect magic square fitness for N = 4 only
    # This fitness function checks for 2x2 sub-squares and diagonal symmetry in addition to the regular fitness checks
    def most_perfect_magic_square_fitness(self, individual):
        """Fitness for most-perfect magic square (N=4 only). Includes 2×2 sub-squares and diagonal pair symmetry."""
        self.fitness_call_count += 1 # Increment the fitness call count each time the fitness function is called
        n = self.n
        half_sum = n * n + 1  # s = n² + 1
        matrix = [individual[i * n:(i + 1) * n] for i in range(n)]
        mpms_fitness = self.regular_magic_square_fitness(individual)  # renamed from fitness_score

        # Add up the absolute differences between every 2x2 sub-squares and the magic constant
        for row in range(n - 1):
            for col in range(n - 1):
                block = (
                    matrix[row][col] +
                    matrix[row][col + 1] +
                    matrix[row + 1][col] +
                    matrix[row + 1][col + 1]
                )
                mpms_fitness += abs(block - ((2 * n * n) + 2))  # compare to magic constant

        # Add up the absolute differences between main and anti-diagonal pairs and half of magic constant
        half = n // 2
        for i in range(half):
            main_diag_pair = matrix[i][i] + matrix[i + half][i + half]
            anti_diag_pair = matrix[i][n - 1 - i] + matrix[i + half][n - 1 - (i + half)]
            mpms_fitness += abs(main_diag_pair - half_sum)
            mpms_fitness += abs(anti_diag_pair - half_sum)

        return mpms_fitness

    # Get the N best individual in the current population by fitness
    def get_best_N_individuals_by_fitness(self, N):
        """Return the N individuals with the highest (worst) fitness in the current population."""
        return sorted(self.population, key=self.fitness_func)[:N]

    # Get the N worst individual in the current population by fitness
    def get_worst_N_individuals_by_fitness(self, N):
        """Return the N individuals with the highest (worst) fitness in the current population."""
        return sorted(self.population, key=self.fitness_func, reverse=True)[:N]

    # Selection based on inverse fitness - lower fitness means higher chance of selection
    def select_parents(self,population):
        weights = [1 / (1 + self.fitness_func(ind)) for ind in population]
        return random.choices(population, weights=weights, k=2)

    # Crossover two parents to create a child
    # This uses a simple one-point crossover method when the two parents are selected
    # The child is created by taking a random segment from the first parent and filling the rest with genes from the second parent
    def crossover(self, p1, p2):
        cut = random.randint(1, self.n * self.n - 2)
        head = p1[:cut]
        # Fill the tail with genes from p2 that are not in head from the cut point to the end of p2
        tail = [g for g in p2[cut:] if g not in head]
        # Now fill the rest from the *start* of p2 to preserve order
        remaining = [g for g in p2 if g not in head and g not in tail]
        # Return a child individual by combining the head and tail segments
        return head + tail + remaining 

    # Mutate an individual by swapping two numbers
    # This mutation is done with a certain probability (mutation_rate)
    #not all children are mutated
    def mutate(self, ind, num_mutations=1):
        for _ in range(num_mutations):
            i, j = random.sample(range(self.n * self.n), 2)
            ind[i], ind[j] = ind[j], ind[i]

    # Optimize an individual by trying all possible 2-cell swaps
    # This method iterates through all pairs of cells in the individual and swaps them
    # If the swap results in a lower fitness score, it updates the individual and breaks the loop
    # This is a local search optimization step to improve the individual
    def optimize(self, individual):

        """Try all possible 2-cell swaps in the individual to reduce fitness and break if found better."""
        best_fitness = self.fitness_func(individual) #initial fitness of the individual
        temp_optimizing_individual = individual[:]
        optimizing_individual = individual[:]

        number_of_cells = self.n * self.n
        
        for i in range(number_of_cells):
            for j in range(i + 1, number_of_cells):
                temp = temp_optimizing_individual[:] # Create a copy of the individual
                temp[i], temp[j] = temp[j], temp[i] # Swap the two cells in the copy
                fit = self.fitness_func(temp) # Calculate fitness of the modified individual
                if fit < best_fitness:
                    best_fitness = fit
                    temp_optimizing_individual = temp[:] # Assign the modified individual to temp_optimizing_individual for next iteration
                    optimizing_individual = temp_optimizing_individual[:]
                    break  # Break the inner loop if found a better fitness
        return optimizing_individual
        
        """Choose randomly two indexes and swap their value in the individual in order to try reduce individual fitness, iterating at most 100 times"""
        """This is the third optimization method i describe in report."""

        # optimizing_individual = individual[:]  
        # temp_optimizing_individual = individual[:]
        # improvement_count = 0  # Count how many times we improve the individual
        # for _ in range(100):
        #     temp = temp_optimizing_individual[:]  # Create a copy of the individual to work with
        #     i = random.randint(0, self.n * self.n - 1)  # Randomly select a cell index
        #     j = random.randint(0, self.n * self.n - 1)  # Randomly select another cell index
        #     while i == j:  # Ensure the two indices are different
        #         j = random.randint(0, self.n * self.n - 1)  # Re-select if they are the same
        #     temp[i], temp[j] = temp[j], temp[i]
        #     if self.fitness_func(temp) < self.fitness_func(temp_optimizing_individual):
        #         temp_optimizing_individual = temp[:]  # Update the temp optimizing individual if fitness improved
        #         optimizing_individual = temp[:]
        #         improvement_count += 1  # Increment the improvement count
        #     if improvement_count >= self.n:
        #         break   
        # return optimizing_individual
        

    # create a new generation of individuals
    # This method selects parents, performs crossover to create children, and applies mutation on part of the population.
    # After 500 generations, the algorithm stops and return the best individual found (it may not be a perfect magic square)
    def step(self):
        mode = self.inheritance_mode # inheritance mode: None, Darwinian, or Lamarckian
        # Step 1: create a separate optimized population for selection only
        if mode == "Darwinian":
            optimized_population = [self.optimize(ind) for ind in self.population]
        elif mode == "Lamarckian":
            optimized_population = [self.optimize(ind) for ind in self.population]
            self.population = optimized_population[:]  # Make the optimized population be the current population for crossover (an self.population attribute).

        current_fitnesses = [self.fitness_func(ind) for ind in self.population]
        worst_fitness = max(current_fitnesses)
        avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
        new_pop = []

        # Apply mutation to fixed number of individuals determined by mutation_rate
        num_mutations = int(self.pop_size * self.mutation_rate)
        for _ in range(num_mutations):
            individual = random.choice(self.population)  # Select a random individual from the population
            self.mutate(individual, self.num_of_mutations)
        
        if mode == "Darwinian":
            current_best = min(optimized_population, key=self.fitness_func) # The best individual in the current population based on optimized population
        else:  # if mode == "Lamarckian" or "None"
            current_best = min(self.population, key=self.fitness_func) # The best individual in the current population. In None mode based on original population ,and in Lamarckian mode based on optimized population.

        current_fit = self.fitness_func(current_best) # The fitness of the current best individual depending on inheritance mode

        # This section checks if no improvement in fitness has been made for 20 generations
        
        # If has improved in this generation, update the best individual and fitness
        if current_fit < self.best_fitness:
            self.best = current_best
            self.best_fitness = current_fit # Update the best fitness attribute if current fitness is better than the last best fitness
            self.no_improvement_counter = 0  # reset counter if fitness improved
            self.last_best_fitness = current_fit # update last best fitness
        # If no improvement in fitness in this generation
        else:
            self.no_improvement_counter += 1
            # If no improvement in 20 generations, increase mutation rate
            if self.no_improvement_counter >= 20:  
                self.num_of_mutations =  2  # Increase the number of mutations to apply to each individual (2 instead of 1)
                self.mutation_rate = 0.4  # Increase to 40% of the population
                self.elitism_rate = 0.3 # Increase to 30% of the population
                print(f"⚠️ No improvement for 20 generations — mutation rate increased to {self.mutation_rate:.4f}")            
                self.no_improvement_counter = 0  # Reset counter
                self.amplification_counter = 0  # To limit the generation's number of amplification value.

        # Reset rates after 20 generations of amplification
        if hasattr(self, 'amplification_counter'):
            self.amplification_counter += 1
            if self.amplification_counter >= 20:
                self.mutation_rate = 0.2
                self.elitism_rate = 0.2
                self.num_of_mutations = 1
                self.no_improvement_counter = 0
                del self.amplification_counter
                print(f"✅ Reset mutation_rate, elitism_rate, and num_of_mutations to original values.")

        # Keep track of fitness history for plotting.
        self.best_fitness_history.append(self.best_fitness)
        self.worst_fitness_history.append(worst_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Generate new generation from original or optimized population depending on inheritance mode
        if self.use_elitism:
            # Step 1: Copy top elite_count individuals unchanged
            elite_count = int(self.pop_size * self.elitism_rate)
            elites = sorted(self.population, key=self.fitness_func)[:elite_count]
            new_pop.extend(elites)

            # Step 2: Fill the rest of the population using selection/crossover/mutation
            for _ in range(self.pop_size - elite_count):
                p1, p2 = self.select_parents(self.population)
                child = self.crossover(p1, p2)
                new_pop.append(child)

        else:
            # No elitism: Generate the whole population from scratch in croosover
            for _ in range(self.pop_size):
                p1, p2 = self.select_parents(self.population)
                child = self.crossover(p1, p2)
                new_pop.append(child)

        self.population = new_pop # Update population property to the new generation who created.
        self.generation += 1

        return self.best_fitness == 0 or self.generation >= 1000

# GUI Class
class MagicSquareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magic Square Genetic Algorithm")
        self.geometry("1200x800")

        self.ga = None
        self.running = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_gui()

    # Setup GUI layout
    def setup_gui(self):
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        tk.Label(control_frame, text="Matrix Size (N):").pack(pady=5)
        self.n_var = tk.StringVar(value="3")
        self.n_combo = ttk.Combobox(control_frame, textvariable=self.n_var, values=["3", "4", "5", "8"], state="readonly")
        self.n_combo.pack(pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_algorithm)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_algorithm)
        self.stop_button.pack(pady=5)

        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_algorithm)
        self.reset_button.pack(pady=5)

        # controls for inheritance mode
        tk.Label(control_frame, text="Inheritance Mode:").pack(pady=(15, 0))
        self.inheritance_mode = tk.StringVar(value="None")  # default
        self.inheritance_combo = ttk.Combobox(
            control_frame,
            textvariable=self.inheritance_mode,
            values=["None", "Darwinian", "Lamarckian"],
            state="readonly"
        )
        self.inheritance_combo.pack(pady=5)


        # Elitism checkbox
        self.elitism_var = tk.BooleanVar(value=False)
        self.elitism_check = ttk.Checkbutton(control_frame, text="Use Elitism", variable=self.elitism_var)
        self.elitism_check.pack(pady=5)
    

        # Log box for output - present the current best individual and fitness
        self.log_box = tk.Text(control_frame, height=25, width=40)
        self.log_box.pack(pady=10)


        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Start GA
    def start_algorithm(self):
        n = int(self.n_var.get())
        
        if self.ga is None:
            # Determine the fitness function based on matrix size only if n = 4 or n = 8
            if n == 4 or n == 8:
                answer = messagebox.askquestion("Fitness Type", "Are you want to find most perfect magic square?")
                if answer == 'yes':
                    fitness_func = lambda ind: self.ga.most_perfect_magic_square_fitness(ind)
                    is_most_perfect = True
                else:
                    fitness_func = lambda ind: self.ga.regular_magic_square_fitness(ind)
                    is_most_perfect = False
            else:
                fitness_func = lambda ind: self.ga.regular_magic_square_fitness(ind)
                is_most_perfect = False

            
            # Initialize the GA with the selected parameters
            self.ga = MagicSquareGA(n=n, fitness_func=fitness_func, inheritance_mode=self.inheritance_mode.get(),use_elitism=self.elitism_var.get()) 
            self.ga.is_most_perfect = is_most_perfect  # <-- simple custom attribute
            self.ga.initialize()
        
        # If the GA stop and i want to run it again from the point i stoped only this line will be run
        self.running = True
        self.after(100, self.run_step)

    # Run one GA step
    def run_step(self):
        if not self.running:
            return
        done = self.ga.step()
        self.update_log()
        self.plot_fitness()
        if not done:
            self.after(50, self.run_step)
        else:
            self.running = False

    # Update the log output
    def update_log(self):
        self.log_box.delete(1.0, tk.END)
        self.log_box.insert(tk.END, f"Fitness Function Calls: {self.ga.fitness_call_count}\n\n")
        self.log_box.insert(tk.END, f"Generation: {self.ga.generation}\n")
        self.log_box.insert(tk.END, f"Best Fitness: {self.ga.best_fitness}\n")
        
        if self.ga.best is not None:
            self.log_box.insert(tk.END, f"Current best Individual:\n{self.format_matrix(self.ga.best)}\n")
        else:
            self.log_box.insert(tk.END, "No best individual yet.\n")

        # If the algorithm is running in most perfect magic square mode, add a note to the log
        if getattr(self.ga, 'is_most_perfect', False):
            self.log_box.insert(tk.END, "\nMost Perfect Magic Square Mode\n")
        # If the algorithm is running in elitism mode, add a note to the log
        if getattr(self.ga, 'use_elitism', False):
            self.log_box.insert(tk.END, "\nUsing Elitism\n")

       

        



    # Convert flat list to matrix string
    def format_matrix(self, flat):
        n = self.ga.n
        return "\n".join(
            " ".join(f"{flat[i * n + j]:2d}" for j in range(n)) for i in range(n)
        )

    # Plot fitness over generations
    def plot_fitness(self):
        self.ax.clear()
        self.ax.plot(self.ga.best_fitness_history, color="green", label="Best Fitness")
        self.ax.plot(self.ga.worst_fitness_history, color="red", label="Worst Fitness")
        self.ax.plot(self.ga.avg_fitness_history, color="darkblue", label="Average Fitness")
        self.ax.set_title("Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        # Force integer ticks
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
        self.ax.legend()
        self.canvas.draw()


    # Stop GA when button is pressed
    def stop_algorithm(self):
        self.running = False
    

    # Reset everything when button is pressed
    # This clears the log and plot, and resets the GA
    def reset_algorithm(self):
        self.running = False
        self.ga = None  # clear existing GA state
        self.log_box.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()

    # Properly shut down the application when the window is closed
    # This ensures that the GA stops running and the window closes cleanly
    def on_close(self):
        self.running = False
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = MagicSquareApp()
    app.mainloop()
