import random

class SamplingUtils:
    # NOTE: The code in this utility class is mostly AI generated, followed by
    # manual inspection for validity as well as tests.

    @staticmethod
    def count_completions(l, avail, Lmin, Lmax):
        """
        Count how many valid completions exist from the current state.
        
        l: current sequence length so far.
        avail: current “available” count.
        Lmin: minimum allowed total length.
        Lmax: maximum allowed total length.
        
        A valid complete sequence is one that ends with avail == 1 and
        with total length between Lmin and Lmax.
        
        Base case:
        - If l == Lmax, we cannot add any more numbers.
            In this case the sequence is valid only if avail == 1.
        
        For l < Lmax, there are two kinds of moves:
        1. Stop now (i.e. terminate the sequence) if allowed.
            You can only stop if l >= Lmin and avail == 1.
        2. Continue by choosing an integer k (from 0 to avail).
            When you add k, update avail as: new_avail = avail - k + 1.
        """
        if l == Lmax:
            return 1 if avail == 1 else 0

        total = 0
        # Option 1: Terminate the sequence now if allowed.
        if l >= Lmin and avail == 1:
            total += 1

        # Option 2: Extend the sequence.
        # You are allowed to choose any k between 0 and avail inclusive.
        for k in range(avail + 1):
            new_avail = avail - k + 1
            total += SamplingUtils.count_completions(l + 1, 
                                                     new_avail, 
                                                     Lmin, 
                                                     Lmax)
        return total

    def sample_state(seq, l, avail, Lmin, Lmax):
        """
        Recursively sample a complete sequence starting from the current state.
        
        seq: the sequence so far.
        l: current length (len(seq)).
        avail: current available count.
        
        There are two types of moves:
        - "Stop": if l >= Lmin and avail == 1  (i.e. the sequence can be terminated now)
        - "Continue": for each k from 0 to avail (if l < Lmax), add k as the next element.
        
        This function uses count_completions() to weight the choices so that every complete
        sequence (of length between Lmin and Lmax) is sampled uniformly.
        """
        # If we have reached the maximum allowed length, we must stop.
        if l == Lmax:
            if avail != 1:
                raise ValueError("Invalid state reached at maximum length!")
            return seq

        options = []  # Each option is a tuple: ("stop", None) OR ("continue", k)
        counts = []   # Number of valid completions that follow each option.

        # Option to stop (terminate now) is allowed only if:
        #   - The sequence is long enough (l >= Lmin), and
        #   - The current state is “complete” (avail == 1).
        if l >= Lmin and avail == 1:
            options.append(("stop", None))
            counts.append(1)  # There is exactly 1 way to stop.

        # Option to continue: try every possible next number k (from 0 to avail).
        for k in range(avail + 1):
            new_avail = avail - k + 1
            cnt = SamplingUtils.count_completions(l + 1, new_avail, Lmin, Lmax)
            options.append(("continue", k))
            counts.append(cnt)

        total_options = sum(counts)
        if total_options == 0:
            # This should not happen if the state is reachable.
            raise ValueError("No valid completions from state: length={}, avail={}".format(l, avail))

        # Randomly choose an option, weighted by the number of completions.
        r = random.randrange(total_options)
        for opt, cnt in zip(options, counts):
            if r < cnt:
                chosen = opt
                break
            r -= cnt

        if chosen[0] == "stop":
            # Terminate the sequence here.
            return seq
        else:
            # Continue by adding the chosen k to the sequence.
            k = chosen[1]
            new_seq = seq + [k]
            new_avail = avail - k + 1
            return SamplingUtils.sample_state(new_seq, 
                                              l + 1, 
                                              new_avail, 
                                              Lmin, 
                                              Lmax)

    def sample_sequence_variable(Lmin, Lmax):
        """
        Samples a valid sequence whose total length is between Lmin and Lmax.
        
        The sequence always starts with 0. After processing the first element (0),
        we have avail = 0 - 0 + 1 = 1 and length = 1.
        
        Then sample_state() takes over to decide, at each step, whether to stop or continue.
        """
        if Lmin < 1 or Lmax < Lmin:
            raise ValueError("Invalid length bounds.")
        # The sequence always begins with 0:
        # State: sequence = [0], length = 1, available count = 1.
        return SamplingUtils.sample_state([0], 1, 1, Lmin, Lmax)

    # Example usage:
    if __name__ == '__main__':
        # Let’s try generating 5 sequences where the length is between 4 and 7.
        Lmin, Lmax = 2, 5
        for i in range(5):
            seq = sample_sequence_variable(Lmin, Lmax)
            print(seq)
