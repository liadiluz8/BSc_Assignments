#################################
# Your name: Liad Iluz
#################################

from array import array
import numpy as np
import matplotlib.pyplot as plt
import intervals

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_arr = np.random.uniform(low=0.0, high=1.0, size=m)
        x_arr = np.sort(x_arr)  # sorted x samples
        Y = np.array([0,1])
        p1 = np.array([0.2,0.8])
        p2 = np.array([0.9,0.1]) 
        f = lambda x : 0<=x<=0.2 or 0.4<=x<=0.6 or 0.8<=x<=1.0
        y_arr = np.array([np.random.choice(Y, p=p1) if f(x) \
            else np.random.choice(Y, p=p2) for x in x_arr])

        return np.array([x_arr,y_arr]).transpose() 

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        samples_range = np.arange(m_first, m_last+1, step, dtype=int)
        average_errors = np.zeros((samples_range.size, 2))
        n=0
        for sample_size in samples_range:
            for i in range(T):
                sample = self.sample_from_D(m=sample_size).transpose()
                intervals_ret, best_error_count = intervals.find_best_interval(sample[0,:], sample[1,:], k)
                true_error = Assignment2.calc_true_error(intervals_ret)
                empirical_error = best_error_count / sample_size
                average_errors[n] += [empirical_error, true_error]
            n+=1
        average_errors /= T # averaging
        
        # Plotting
        Assignment2.plot_errors(average_errors.transpose(), samples_range, "Number of Samples n")
        
        return average_errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_range = np.arange(k_first, k_last+1, step, dtype=int)
        sample = self.sample_from_D(m).transpose()
        ERM_results = [intervals.find_best_interval(sample[0,:], sample[1,:], k) for k in k_range]
        ERM_empirical_errors = [ERM_results[i][1]/m for i in range(len(ERM_results))]
        ERM_true_errors = [Assignment2.calc_true_error(ERM_results[i][0]) for i in range(len(ERM_results))]
        errors = np.array([ERM_empirical_errors,ERM_true_errors])

        # Plotting
        Assignment2.plot_errors(errors, k_range, "k")
        
        return k_range[np.argmin(ERM_true_errors)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        k_range = np.arange(1, 10+1, 1, dtype=int)
        test_errors, ret_intervals = np.array([]), []
        sample = self.sample_from_D(m).transpose()
        test_size = 0.2
        X_train, X_test, y_train, y_test = Assignment2.train_test_split(sample[0], sample[1], test_size)

        for k in k_range:
            intervals_res, best_error_count = intervals.find_best_interval(X_train, y_train, k)
            empirical_test_error = Assignment2.calc_sample_error(intervals_res, X_test, y_test)
            test_errors = np.append(test_errors, empirical_test_error)
            ret_intervals.append(intervals_res)
        
        best_k = k_range[np.argmin(test_errors)]
        
        return best_k

    #################################
    # Place for additional methods
    def calc_true_error(intervals : list[tuple]):
        U = [(0,0.2),(0.4,0.6),(0.8,1)]
        weight_U = Assignment2.calc_weight_intervals(U)
        weight_UI = Assignment2.calc_weight_intervals(intervals)
        weight_intersec = Assignment2.calc_weight_intersection_intervals(U, intervals)
        
        # Using the formula of the theoretical analysis in section a
        true_error = 0.2*weight_intersec + 0.9*(weight_UI - weight_intersec) + 0.8*(weight_U - weight_intersec) \
             + 0.1*(1 - weight_U - weight_UI + weight_intersec)
        return true_error

    def calc_weight_intervals(intervals : list[tuple]):
        weight = 0
        for interval in intervals:
            weight += interval[1] - interval[0]
        return weight
    
    def calc_weight_intersection_intervals(interv_1 : list[tuple], interv_2 : list[tuple]):
        weight = 0
        s1 = len(interv_1)
        s2 = len(interv_2)
        i1 = i2 = 0
        while i1 < s1 and i2 < s2:
            l1 = interv_1[i1][0]
            r1 = interv_1[i1][1]
            l2 = interv_2[i2][0]
            r2 = interv_2[i2][1]

            if l2 <= l1 <= r2 <= r1:
                weight += r2 - l1
                i2 += 1
            elif l1 <= l2 <= r1 <= r2:
                weight += r1 - l2
                i1 += 1
            elif l2 <= l1 <= r1 <= r2:
                weight += r1 - l1
                i1 += 1
            elif l1 <= l2 <= r2 <= r1:
                weight += r2 - l2
                i2 += 1
            elif l1 <= r1 <= l2 <= r2:
                i1 += 1
            elif l2 <= r2 <= l1 <= r1:
                i2 += 1

        return weight

    def plot_errors(errors, x_axis_range, x_axis_name):
        plt.title("Ex1 : Empirical and True Error with respect to "+x_axis_name)
        plt.xlabel(x_axis_name)
        plt.ylabel("Error")
        plt.plot(x_axis_range, errors[1], 'o')  # True
        plt.plot(x_axis_range, errors[0], 'o')  # Empirical
        plt.legend({'True error','Empirical error'})
        plt.show()

    def hypothesis_interval(intervals : list[tuple], x):
        for a,b in intervals:
            if a <= x <= b:
                return 1
        return 0
    
    def calc_sample_error(intervals : list[tuple], X_sample, y_sample):
        error = 0
        for i in range(y_sample.size):
            x = X_sample[i]
            y = y_sample[i]
            if Assignment2.hypothesis_interval(intervals, x) != y:
                error+=1
        return error/y_sample.size

    def train_test_split(X_sample, y_sample, test_size):
        """
            X_sample is sorted
        """
        sample_size = y_sample.size
        random_test_indices = np.sort(np.random.choice(np.arange(sample_size), \
            size=(int)(sample_size*test_size), replace=False))
        random_train_indicies = np.array([i for i in np.arange(sample_size) \
            if i not in random_test_indices])
        return X_sample[random_train_indicies], X_sample[random_test_indices], \
            y_sample[random_train_indicies], y_sample[random_test_indices]

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    average_errors = ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    best_k = ass.experiment_k_range_erm(1500, 1, 10, 1)
    best_k = ass.cross_validation(1500)

