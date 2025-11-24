# Modified iterative unmasking to save every possible choice
def iterative_unmasking_save_all(sequence, csv_file, init_mask, idx_to_aa):
    # Initialize dataset
    dataset = data.Load_Dataset(csv_file)
    summary = []

    # Loop through each starting position in the initial mask
    for start_pos in init_mask:
        current_sequence = sequence
        current_masklist = init_mask.copy()
        results = []
        confidence_scores = []

        # Ensure the first position to unmask is the given start position
        current_masklist.remove(start_pos)
        sample = dataset.__getitem__(0, sequence=current_sequence, masklist=current_masklist)

        # Unmask the initial start position
        updated_sequence, new_masklist, predicted_aa, confidence_score, nll = unmasking(
            sample,
            current_sequence,
            current_masklist,
            start_pos,
            idx_to_aa
        )

        # Store the results of the first iteration
        result = {
            'iteration': len(results) + 1,
            'position_unmasked': start_pos,
            'predicted_aa': predicted_aa,
            'confidence_score': confidence_score,
            'nll': nll,
            'updated_sequence': updated_sequence,
            'remaining_masks': new_masklist.copy(),
            'all_choices': []  # To store all possible choices for each iteration
        }
        results.append(result)
        confidence_scores.append(confidence_score)

        # Update the sequence and mask list for the next iterations
        current_sequence = updated_sequence
        current_masklist = new_masklist

        # Continue unmasking until all positions are unmasked
        while current_masklist:
            save_choice = []  # To store all options for this iteration

            # Loop through each possible position to unmask and save the outcome
            for pos in current_masklist:
                sample = dataset.__getitem__(0, sequence=current_sequence, masklist=current_masklist)

                # Unmask the selected position
                temp_sequence, temp_masklist, temp_predicted_aa, temp_confidence_score, temp_nll = unmasking(
                    sample,
                    current_sequence,
                    current_masklist,
                    pos,
                    idx_to_aa
                )
                # Save information about this choice
                choice_info = {
                    'unmask_pos': pos,
                    'predicted_aa': temp_predicted_aa,
                    'confidence_score': temp_confidence_score,
                    'nll': temp_nll,
                    'updated_sequence': temp_sequence
                }
                save_choice.append(choice_info)

            # Randomly select one of the available positions to proceed with
            selected_choice = random.choice(save_choice)

            # Update current state with the selected choice
            current_sequence = selected_choice['updated_sequence']
            current_masklist = [pos for pos in current_masklist if pos != selected_choice['unmask_pos']]
            confidence_scores.append(selected_choice['confidence_score'])

            # Store the result for the selected choice
            result = {
                'iteration': len(results) + 1,
                'position_unmasked': selected_choice['unmask_pos'],
                'predicted_aa': selected_choice['predicted_aa'],
                'confidence_score': selected_choice['confidence_score'],
                'nll': selected_choice['nll'],
                'updated_sequence': current_sequence,
                'remaining_masks': current_masklist.copy(),
                'all_choices': save_choice  # Store all options for analysis
            }
            results.append(result)

        # Calculate average confidence score for this run
        average_confidence = sum(confidence_scores) / len(confidence_scores)

        # Append the summary of the current run
        summary.append({
            'original_seq': sequence,
            'start_mask': start_pos,
            'final_sequence': current_sequence,
            'average_confidence': average_confidence,
            'results': results
        })

    return summary
