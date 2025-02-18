import pandas as pd

def test_concatenate_dataframe_list():
  #Definição do input a ser realizado
  input = [
      pd.DataFrame({'col1': [1,2], 'col2': [3,4]}),
      pd.DataFrame({'col1': [5,6], 'col2': [7,8]})
  ]
		
#definição do output esperado
expected_output = pd.concat([input[0], input[1]], ignore_index=True)

#chamada da função a ser testada
result = test_concatenate_dataframe_list([input[0], input[1]])

#utilização do assert
assert expected_output.equals(result)