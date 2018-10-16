function q9()
lns = [.01,.02,.05,.1, .25, 1]
for lr = lns
  fprintf('Testing learning rate: %d \n', lr);
  a4_main(300, lr, .005, 1000)
end
  
end
