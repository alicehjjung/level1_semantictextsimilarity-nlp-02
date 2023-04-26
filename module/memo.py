
def memo(path, today, model_name, max_token_length, batch_size, max_epoch,learning_rate, scheduler=''):
    with open(path,'a') as f:
        text = '{}\t\t{}\t\t\t{}\t{}\t{}\t{}\t{}\n'.format('today','model_name', 'max_token_length', 'batch_size', 'max_epoch','learning_rate', 'scheduler')
        f.write(text)
        text = '{}\t{}\t{}\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}\n'.format(today, model_name, max_token_length, batch_size, max_epoch,learning_rate, scheduler)
        f.write(text)
