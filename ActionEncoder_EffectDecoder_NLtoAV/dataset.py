def load():
    global state_to_variable, output_to_variable, output_from_variable, get_states_and_actions, LanguageModule, Encoder, Decoder, loss, state_to_variable_batch, output_to_variable_batch, output_from_variable_batch, LSTMLanguageModule, get_session_ids, get_session_data
    from pretraindata.sg1sg2nlat import state_to_variable, output_to_variable, output_from_variable, get_states_and_actions, LanguageModule, Encoder, Decoder, loss, state_to_variable_batch, output_to_variable_batch, output_from_variable_batch, LSTMLanguageModule, get_session_ids, get_session_data