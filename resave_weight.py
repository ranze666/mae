import torch
import model_bootstrapped_mae

model = model_bootstrapped_mae.BootstrappedMAE()
checkpoint = torch.load('/home/wts/code/gaoyang_test/mae/output_dir/bootmae/run_20250307_041023/checkpoint-199.pth',map_location='cpu')

model.load_state_dict(checkpoint['model']) 
torch.save({
    'model':model.student_model.state_dict()
},'/home/wts/code/gaoyang_test/mae/output_dir/bootmae/run_20250307_041023/stu199.pth')