def standardize(vector):
        M = max(vector)
        m = min(vector)
        std_vec = (vector - [m]*len(vector))/(M - m)
        return std_vec