def word_error_rate(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and 
    hypothesis sequences.

    Args:
        reference (list): List of words in the reference sequence.
        hypothesis (list): List of words in the hypothesis sequence.

    Returns:
        wer (float): Word Error Rate (WER) as a percentage.
    """
    # Create a matrix to store the edit distances
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution_cost = dp[i - 1][j - 1] + 1
                insertion_cost = dp[i][j - 1] + 1
                deletion_cost = dp[i - 1][j] + 1
                dp[i][j] = min(substitution_cost, insertion_cost, deletion_cost)

    # Calculate the WER as a percentage
    wer = (dp[n][m] / n) * 100.0
    return wer