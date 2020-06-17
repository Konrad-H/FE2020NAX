def standardize(vector):
        M_v = max(vector)
        m_v = min(vector)
        std_vec = (vector - [m_v]*len(vector))/(M_v - m_v)
        return std_vec