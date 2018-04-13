def run(index, field_names, field_values, obj, func):
    if(index == len(field_names)):
        func(**obj)
    else:
        for i in range(len(field_values[index])):
            obj[field_names[index]] = field_values[index][i]
            run(index+1, field_names, field_values, obj, func)
    
def cross(obj, func):
    field_names = []
    field_values = []
    for i in obj:
        field_names.append(i)
        field_values.append(obj[i])
    run(0, field_names, field_values, {}, func)

#params_collection = {
#    'a': [3,54,5], 
#    'b': [7,6]
#}
#
#def testFunc(**kwargs):
#    print(kwargs)    # prints the dictionary of keyword arguments

#cross(params_collection, testFunc)